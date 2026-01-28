import torch
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import queue
import logging
import time
import io
import tempfile
import os

from lipsync import get_lip_sync_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AudioEngine")


class AudioEngine:
    """
    High-performance audio engine using:
    - faster-whisper for STT (4x faster than OpenAI Whisper)
    - Qwen3-TTS for TTS (0.6B-1.7B params, high quality, voice cloning)

    Optimized for RTX 3090 with CUDA acceleration.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Audio Engine running on: {self.device}")

        # --- STT Setup (faster-whisper) ---
        self._init_stt()

        # --- TTS Setup (Qwen3-TTS) ---
        self._init_tts()

        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.is_speaking = False

    def _init_stt(self):
        """Initialize faster-whisper for speech-to-text."""
        logger.info("Loading faster-whisper STT...")

        try:
            from faster_whisper import WhisperModel

            # Use large-v3-turbo for best speed/quality on RTX 3090
            # compute_type="float16" for optimal GPU performance
            self.stt_model = WhisperModel(
                "large-v3-turbo",
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8"
            )
            self.use_faster_whisper = True
            logger.info("Successfully loaded faster-whisper (large-v3-turbo)")

        except ImportError:
            logger.warning("faster-whisper not installed, falling back to transformers pipeline")
            self._init_fallback_stt()
        except Exception as e:
            logger.warning(f"Could not load faster-whisper: {e}")
            self._init_fallback_stt()

    def _init_fallback_stt(self):
        """Fallback to transformers pipeline if faster-whisper unavailable."""
        from transformers import pipeline

        self.stt_pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base.en",
            device=self.device
        )
        self.use_faster_whisper = False
        logger.info("Using fallback: transformers whisper-base.en")

    def _init_tts(self):
        """Initialize Qwen3-TTS for text-to-speech."""
        logger.info("Loading Qwen3-TTS...")

        try:
            from qwen_tts import Qwen3TTSModel

            # Load Qwen3-TTS model (0.6B for faster inference, 1.7B for higher quality)
            # Using CustomVoice variant for voice selection support
            self.tts_model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                device_map=self.device,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            )

            # Default voice settings
            self.tts_voice = 'Vivian'  # Default voice
            self.tts_language = 'English'
            self.tts_speed = 1.0

            self.use_qwen_tts = True
            self.tts_sample_rate = 24000  # Qwen3-TTS outputs 24kHz
            logger.info(f"Successfully loaded Qwen3-TTS (voice: {self.tts_voice})")

        except ImportError:
            logger.warning("qwen-tts not installed. Install with: pip install qwen-tts")
            logger.warning("Falling back to SpeechT5")
            self._init_fallback_tts()
        except Exception as e:
            logger.warning(f"Could not load Qwen3-TTS: {e}")
            self._init_fallback_tts()

    def _init_fallback_tts(self):
        """Fallback to SpeechT5 if Qwen3-TTS unavailable."""
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.fallback_tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)

        # Load speaker embedding
        from datasets import load_dataset
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(self.device)

        self.use_qwen_tts = False
        self.tts_sample_rate = 16000
        logger.info("Using fallback: SpeechT5 TTS")

    def set_voice(self, voice: str):
        """
        Set the TTS voice (only for Qwen3-TTS).

        Available voices vary by model. Common options include:
        - Vivian, Emily, James, etc.
        """
        if self.use_qwen_tts:
            self.tts_voice = voice
            logger.info(f"Voice set to: {voice}")

    def set_speed(self, speed: float):
        """Set TTS speed (0.5 to 2.0, default 1.0)."""
        self.tts_speed = max(0.5, min(2.0, speed))
        logger.info(f"Speed set to: {self.tts_speed}")

    def listen_and_transcribe(self, silence_timeout=10.0, max_duration=120.0, status_callback=None):
        """
        Records audio using voice activity detection (VAD).
        Stops recording after silence_timeout seconds of silence.

        Args:
            silence_timeout: Seconds of silence before stopping (default 10)
            max_duration: Maximum recording duration in seconds (default 120)
            status_callback: Optional callback for status updates (receives string)

        Returns:
            Transcribed text string
        """
        fs = 16000  # 16kHz for Whisper
        chunk_duration = 0.1  # 100ms chunks for VAD
        chunk_samples = int(fs * chunk_duration)

        # VAD parameters
        energy_threshold = 0.01  # Minimum RMS energy to consider as speech
        speech_detected = False
        silence_start = None
        recording_start = time.time()

        audio_chunks = []
        self.is_listening = True

        logger.info("Listening with VAD... (speak now)")
        if status_callback:
            status_callback("Listening... (speak now)")

        try:
            with sd.InputStream(samplerate=fs, channels=1, dtype='float32') as stream:
                while self.is_listening:
                    # Check max duration
                    elapsed = time.time() - recording_start
                    if elapsed > max_duration:
                        logger.info("Max recording duration reached")
                        break

                    # Read audio chunk
                    chunk, overflowed = stream.read(chunk_samples)
                    audio_chunks.append(chunk.copy())

                    # Calculate RMS energy
                    rms = np.sqrt(np.mean(chunk ** 2))

                    if rms > energy_threshold:
                        # Speech detected
                        if not speech_detected:
                            logger.info("Speech started")
                            if status_callback:
                                status_callback("Hearing you...")
                        speech_detected = True
                        silence_start = None
                    else:
                        # Silence
                        if speech_detected:
                            if silence_start is None:
                                silence_start = time.time()
                            else:
                                silence_duration = time.time() - silence_start
                                remaining = silence_timeout - silence_duration
                                if remaining <= 3 and remaining > 0:
                                    if status_callback:
                                        status_callback(f"Silence detected... ({int(remaining)}s)")
                                if silence_duration >= silence_timeout:
                                    logger.info(f"Silence timeout ({silence_timeout}s) reached")
                                    break

            self.is_listening = False

            if not audio_chunks:
                return ""

            # Concatenate all audio
            audio_data = np.concatenate(audio_chunks).squeeze()

            # Check if we actually got any speech
            if not speech_detected:
                logger.info("No speech detected")
                return ""

            logger.info("Processing audio...")
            if status_callback:
                status_callback("Processing speech...")

            # Transcribe
            if self.use_faster_whisper:
                segments, info = self.stt_model.transcribe(
                    audio_data,
                    beam_size=5,
                    language="en",
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                text = "".join([segment.text for segment in segments]).strip()
            else:
                result = self.stt_pipe(audio_data)
                text = result["text"].strip()

            return text

        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return ""
        finally:
            self.is_listening = False

    def stop_listening(self):
        """Stop the current listening session."""
        self.is_listening = False

    def speak(self, text, avatar_callback=None):
        """
        Converts text to speech using Qwen3-TTS and plays it with phoneme-based lip sync.

        Args:
            text: Text to speak
            avatar_callback: function(is_talking: bool) - called when speech starts/stops
        """
        if not text:
            return

        self.is_speaking = True
        logger.info(f"Speaking: {text}")

        # Get lip sync engine
        lip_sync = get_lip_sync_engine()

        try:
            if self.use_qwen_tts:
                self._speak_qwen(text, lip_sync, avatar_callback)
            else:
                self._speak_fallback(text, lip_sync, avatar_callback)

        except Exception as e:
            logger.error(f"TTS Error: {e}")
        finally:
            lip_sync.stop_playback()
            if avatar_callback:
                avatar_callback(False)
            self.is_speaking = False

    def _speak_qwen(self, text, lip_sync, avatar_callback):
        """Generate and play speech using Qwen3-TTS."""
        import soundfile as sf

        # Generate audio with Qwen3-TTS
        wavs, sr = self.tts_model.generate_custom_voice(
            text=text,
            language=self.tts_language,
            speaker=self.tts_voice,
        )

        if wavs is None or len(wavs) == 0:
            logger.warning("No audio generated")
            return

        # Get the first waveform
        speech_np = wavs[0] if isinstance(wavs, list) else wavs

        # Ensure it's a numpy array
        if hasattr(speech_np, 'cpu'):
            speech_np = speech_np.cpu().numpy()

        # Update sample rate from model output
        self.tts_sample_rate = sr

        # Calculate duration
        audio_duration = len(speech_np) / self.tts_sample_rate

        # Start lip sync
        lip_sync.start_playback(text, audio_duration)
        logger.info(f"Lip sync started for {audio_duration:.2f}s")

        # Notify avatar
        if avatar_callback:
            avatar_callback(True)

        # Play audio
        sd.play(speech_np, samplerate=self.tts_sample_rate)
        sd.wait()

    def _speak_fallback(self, text, lip_sync, avatar_callback):
        """Generate and play speech using SpeechT5 (fallback)."""
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        speech = self.fallback_tts_model.generate_speech(
            inputs["input_ids"],
            self.speaker_embeddings,
            vocoder=self.vocoder
        )

        speech_np = speech.cpu().numpy()
        audio_duration = len(speech_np) / self.tts_sample_rate

        # Start lip sync
        lip_sync.start_playback(text, audio_duration)
        logger.info(f"Lip sync started for {audio_duration:.2f}s")

        # Notify avatar
        if avatar_callback:
            avatar_callback(True)

        # Play audio
        sd.play(speech_np, samplerate=self.tts_sample_rate)
        sd.wait()

    def speak_async(self, text, avatar_callback=None):
        """
        Non-blocking version of speak(). Runs TTS in a separate thread.
        """
        thread = threading.Thread(
            target=self.speak,
            args=(text, avatar_callback),
            daemon=True
        )
        thread.start()
        return thread


if __name__ == "__main__":
    # Test the audio engine
    print("Initializing Audio Engine...")
    engine = AudioEngine()

    print("\n--- Testing TTS ---")
    engine.speak("Hello! I am CoFoc, your AI companion. How can I help you today?")

    print("\n--- Testing STT ---")
    print("Say something (5 seconds)...")
    text = engine.listen_and_transcribe(duration=5)
    print(f"Heard: {text}")

    if text:
        engine.speak(f"I heard you say: {text}")
