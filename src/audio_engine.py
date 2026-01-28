import torch
from transformers import AutoProcessor, AutoModelForTextToSpeech, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from transformers import pipeline
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import queue
import logging
import time
import io
import librosa

from lipsync import get_lip_sync_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AudioEngine")

class AudioEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Audio Engine running on: {self.device}")

        # --- STT Setup (Whisper) ---
        logger.info("Loading Whisper STT...")
        self.stt_pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base.en", # Using base.en for speed/accuracy balance
            device=self.device
        )

        # --- TTS Setup ---
        logger.info("Loading TTS Model...")
        self.tts_model_id = "Qwen/Qwen3-TTS-1.7B"  # Attempting requested model
        
        try:
            # Hypothetical loading for Qwen3 TTS if it adheres to standard AutoModel
            self.processor = AutoProcessor.from_pretrained(self.tts_model_id)
            self.tts_model = AutoModelForTextToSpeech.from_pretrained(self.tts_model_id).to(self.device)
            self.use_fallback_tts = False
            logger.info(f"Successfully loaded {self.tts_model_id}")
        except Exception as e:
            logger.warning(f"Could not load specific model {self.tts_model_id}: {e}")
            logger.info("Falling back to Microsoft SpeechT5...")
            
            # Fallback: SpeechT5 (High quality, fast)
            self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            self.tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(self.device)
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
            
            # Load xvector speaker embedding for SpeechT5
            from datasets import load_dataset
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            self.speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(self.device)
            
            self.use_fallback_tts = True

        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.is_speaking = False

    def listen_and_transcribe(self, duration=5):
        """
        Records audio for a fixed duration and transcribes it.
        In a real app, you'd want VAD (Voice Activity Detection).
        For this prototype, we record fixed chunks.
        """
        fs = 16000  # Whisper expects 16kHz
        logger.info("Listening...")
        
        try:
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
            sd.wait()
            logger.info("Processing audio...")
            
            # Squeeze to 1D array
            audio_data = recording.squeeze()
            
            # Transcribe
            result = self.stt_pipe(audio_data)
            text = result["text"].strip()
            
            return text
        except Exception as e:
            logger.error(f"Error in hearing: {e}")
            return ""

    def speak(self, text, avatar_callback=None):
        """
        Converts text to speech and plays it with phoneme-based lip sync.
        avatar_callback: function(is_talking: bool)
        """
        if not text:
            return

        self.is_speaking = True
        logger.info(f"Speaking: {text}")

        # Get lip sync engine
        lip_sync = get_lip_sync_engine()

        try:
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)

            if self.use_fallback_tts:
                speech = self.tts_model.generate_speech(inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder)
            else:
                # Generic generation for AutoModel
                speech = self.tts_model.generate(**inputs)

            # Move to CPU and numpy
            speech_np = speech.cpu().numpy()

            # Calculate audio duration
            sample_rate = 16000
            audio_duration = len(speech_np) / sample_rate

            # Start phoneme-based lip sync BEFORE playing audio
            lip_sync.start_playback(text, audio_duration)
            logger.info(f"Lip sync started for {audio_duration:.2f}s")

            # Notify avatar that speaking has started
            if avatar_callback:
                avatar_callback(True)

            # Play audio
            sd.play(speech_np, samplerate=sample_rate)
            sd.wait()

        except Exception as e:
            logger.error(f"TTS Error: {e}")
        finally:
            # Stop lip sync
            lip_sync.stop_playback()

            if avatar_callback:
                avatar_callback(False)
            self.is_speaking = False

if __name__ == "__main__":
    # Test
    engine = AudioEngine()
    print("Say something (5s)...")
    text = engine.listen_and_transcribe()
    print(f"Heard: {text}")
    engine.speak(f"I heard you say: {text}")
