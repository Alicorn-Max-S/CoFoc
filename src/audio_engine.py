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
    - Kokoro for TTS (82M params, fast, high quality)

    Optimized for RTX 3090 with CUDA acceleration.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Audio Engine running on: {self.device}")

        # --- STT Setup (faster-whisper) ---
        self._init_stt()

        # --- TTS Setup (Kokoro) ---
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
        """Initialize Kokoro for text-to-speech."""
        logger.info("Loading Kokoro TTS...")

        try:
            from kokoro import KPipeline

            # Initialize Kokoro with American English
            # 'a' = American English, 'b' = British English
            self.tts_pipeline = KPipeline(lang_code='a')

            # Available voices: af_heart, af_bella, af_nicole, af_sarah, af_sky
            # am_adam, am_michael (male voices)
            self.tts_voice = 'af_heart'  # Default female voice, natural sounding
            self.tts_speed = 1.0

            self.use_kokoro = True
            self.tts_sample_rate = 24000  # Kokoro outputs 24kHz
            logger.info(f"Successfully loaded Kokoro TTS (voice: {self.tts_voice})")

        except ImportError:
            logger.warning("Kokoro not installed, falling back to SpeechT5")
            self._init_fallback_tts()
        except Exception as e:
            logger.warning(f"Could not load Kokoro: {e}")
            self._init_fallback_tts()

    def _init_fallback_tts(self):
        """Fallback to SpeechT5 if Kokoro unavailable."""
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)

        # Load speaker embedding
        from datasets import load_dataset
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(self.device)

        self.use_kokoro = False
        self.tts_sample_rate = 16000
        logger.info("Using fallback: SpeechT5 TTS")

    def set_voice(self, voice: str):
        """
        Set the TTS voice (only for Kokoro).

        Available voices:
        - Female: af_heart, af_bella, af_nicole, af_sarah, af_sky
        - Male: am_adam, am_michael
        """
        if self.use_kokoro:
            self.tts_voice = voice
            logger.info(f"Voice set to: {voice}")

    def set_speed(self, speed: float):
        """Set TTS speed (0.5 to 2.0, default 1.0)."""
        self.tts_speed = max(0.5, min(2.0, speed))
        logger.info(f"Speed set to: {self.tts_speed}")

    def listen_and_transcribe(self, duration=5):
        """
        Records audio for a fixed duration and transcribes it.
        Uses faster-whisper for 4x faster transcription.
        """
        fs = 16000  # 16kHz for Whisper
        logger.info("Listening...")

        try:
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
            sd.wait()
            logger.info("Processing audio...")

            audio_data = recording.squeeze()

            if self.use_faster_whisper:
                # faster-whisper expects numpy array or file path
                segments, info = self.stt_model.transcribe(
                    audio_data,
                    beam_size=5,
                    language="en",
                    vad_filter=True,  # Filter out non-speech
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                text = "".join([segment.text for segment in segments]).strip()
            else:
                # Fallback to transformers pipeline
                result = self.stt_pipe(audio_data)
                text = result["text"].strip()

            return text

        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return ""

    def speak(self, text, avatar_callback=None):
        """
        Converts text to speech using Kokoro and plays it with phoneme-based lip sync.

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
            if self.use_kokoro:
                self._speak_kokoro(text, lip_sync, avatar_callback)
            else:
                self._speak_fallback(text, lip_sync, avatar_callback)

        except Exception as e:
            logger.error(f"TTS Error: {e}")
        finally:
            lip_sync.stop_playback()
            if avatar_callback:
                avatar_callback(False)
            self.is_speaking = False

    def _speak_kokoro(self, text, lip_sync, avatar_callback):
        """Generate and play speech using Kokoro TTS."""
        # Generate audio with Kokoro
        # Kokoro returns a generator that yields (samples, sample_rate, phonemes)
        audio_chunks = []
        phoneme_data = []

        generator = self.tts_pipeline(
            text,
            voice=self.tts_voice,
            speed=self.tts_speed
        )

        for samples, sample_rate, phonemes in generator:
            audio_chunks.append(samples)
            if phonemes:
                phoneme_data.append(phonemes)

        if not audio_chunks:
            logger.warning("No audio generated")
            return

        # Concatenate all audio chunks
        speech_np = np.concatenate(audio_chunks)

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
        speech = self.tts_model.generate_speech(
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
