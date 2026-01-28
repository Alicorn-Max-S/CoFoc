import sys
import argparse
import threading
import time
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QThread, pyqtSignal, QObject

from avatar import AvatarWidget, configure_opengl_format
from brain import Brain
from audio_engine import AudioEngine

# Timeout for resetting conversation context (in seconds)
INACTIVITY_TIMEOUT = 30.0


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

        # Push-to-talk state
        self.talk_requested = threading.Event()
        self.last_interaction_time = time.time()

    def request_talk(self):
        """Called when user presses the talk key."""
        self.talk_requested.set()

    def run(self):
        # Initialize heavy models in the thread to avoid freezing GUI on startup
        self.signals.update_status.emit("Initializing Brain (Ollama)...")
        self.brain = Brain()

        self.signals.update_status.emit("Initializing Audio Engine (Whisper + TTS)...")
        self.audio = AudioEngine()

        self.signals.update_status.emit("Ready! Press T to talk.")

        while self.running:
            try:
                # Wait for user to press talk key (with timeout to check for inactivity)
                if not self.talk_requested.wait(timeout=1.0):
                    # Check for inactivity timeout
                    if time.time() - self.last_interaction_time > INACTIVITY_TIMEOUT:
                        if self.brain.history:
                            self.brain.reset_context()
                            self.signals.update_status.emit("Context reset due to inactivity. Press T to talk.")
                            self.last_interaction_time = time.time()
                    continue

                # Clear the event for next time
                self.talk_requested.clear()

                # 1. Listen
                self.signals.update_status.emit("Listening... (speak now)")
                user_text = self.audio.listen_and_transcribe(duration=5)

                if not user_text:
                    self.signals.update_status.emit("Heard silence. Press T to talk.")
                    continue

                # Update last interaction time
                self.last_interaction_time = time.time()

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

                # Update last interaction time after speaking
                self.last_interaction_time = time.time()
                self.signals.update_status.emit("Ready! Press T to talk.")

            except Exception as e:
                print(f"Error in loop: {e}")
                self.signals.update_status.emit("Error occurred. Press T to try again.")
                time.sleep(1)

    def stop(self):
        self.running = False
        self.talk_requested.set()  # Unblock the wait
        self.wait()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CoFoc - 3D AI Assistant")
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to a GLB/glTF/VRM avatar model')
    parser.add_argument('--no-ai', action='store_true',
                        help='Run avatar only without AI/audio (for testing)')
    args = parser.parse_args()

    # Configure OpenGL format before creating QApplication
    configure_opengl_format()

    app = QApplication(sys.argv)

    # Create 3D Avatar Window with optional custom model
    avatar = AvatarWidget(model_path=args.model)
    avatar.show()

    # Disable click-through so user can interact with keyboard
    avatar.set_click_through(False)

    if args.no_ai:
        # Run in test mode without AI
        print("[System]: Running in test mode (no AI)")
        print("[System]: Press Space to toggle speaking animation")
        print("[System]: Press R to toggle camera rotation")
        print("[System]: Press C to toggle click-through mode")
        print("[System]: Press Esc to quit")

        sys.exit(app.exec())
    else:
        # Create Signals
        signals = AssistantSignals()
        signals.set_speaking.connect(avatar.set_speaking)
        signals.update_status.connect(lambda s: print(f"[System]: {s}"))

        # Start Worker Thread
        worker = AssistantWorker(signals)

        # Connect the avatar's talk signal to the worker
        avatar.talk_requested.connect(worker.request_talk)

        worker.start()

        print("[System]: Controls:")
        print("[System]:   T     - Press to talk (push-to-talk)")
        print("[System]:   R     - Toggle camera rotation")
        print("[System]:   C     - Toggle click-through mode")
        print("[System]:   Esc   - Quit")
        print("[System]: Context resets after 30 seconds of inactivity.")

        try:
            sys.exit(app.exec())
        except KeyboardInterrupt:
            worker.stop()


if __name__ == "__main__":
    main()
