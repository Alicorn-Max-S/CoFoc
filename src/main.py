import sys
import threading
import time
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QThread, pyqtSignal, QObject

from avatar import AvatarWidget, configure_opengl_format
from brain import Brain
from audio_engine import AudioEngine

class AssistantSignals(QObject):
    update_status = pyqtSignal(str)
    set_speaking = pyqtSignal(bool)

class AssistantWorker(QThread):
    def __init__(self, signals):
        super().__init__()
        self.signals = signals
        self.running = True
        self.brain = None
        self.audio = None

    def run(self):
        # Initialize heavy models in the thread to avoid freezing GUI on startup
        self.signals.update_status.emit("Initializing Brain (Ollama)...")
        self.brain = Brain()
        
        self.signals.update_status.emit("Initializing Audio Engine (Whisper + TTS)...")
        self.audio = AudioEngine()
        
        self.signals.update_status.emit("Ready! Speak now.")
        
        while self.running:
            try:
                # 1. Listen
                self.signals.update_status.emit("Listening...")
                user_text = self.audio.listen_and_transcribe(duration=5)
                
                if not user_text:
                    self.signals.update_status.emit("Heard silence.")
                    continue
                    
                self.signals.update_status.emit(f"Heard: {user_text}")
                
                # 2. Think
                self.signals.update_status.emit("Thinking...")
                response = self.brain.think(user_text)
                
                # 3. Speak
                self.signals.update_status.emit("Speaking...")
                
                # We need a callback to sync the avatar mouth
                # Since we are in a thread, we use signals
                def avatar_callback(is_speaking):
                    self.signals.set_speaking.emit(is_speaking)
                
                self.audio.speak(response, avatar_callback)
                
            except Exception as e:
                print(f"Error in loop: {e}")
                time.sleep(1)

    def stop(self):
        self.running = False
        self.wait()

def main():
    # Configure OpenGL format before creating QApplication
    configure_opengl_format()

    app = QApplication(sys.argv)

    # Create 3D Avatar Window
    avatar = AvatarWidget()
    avatar.show()
    
    # Create Signals
    signals = AssistantSignals()
    signals.set_speaking.connect(avatar.set_speaking)
    signals.update_status.connect(lambda s: print(f"[System]: {s}"))

    # Start Worker Thread
    worker = AssistantWorker(signals)
    worker.start()
    
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        worker.stop()

if __name__ == "__main__":
    main()
