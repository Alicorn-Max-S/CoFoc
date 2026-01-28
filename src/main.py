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

        # Conversation state
        self.talk_requested = threading.Event()
        self.last_interaction_time = time.time()
        self.in_conversation = False  # True when actively conversing (auto-listen mode)

    def request_talk(self):
        """Called when user presses the talk key."""
        self.talk_requested.set()

    def run(self):
        # Initialize heavy models in the thread to avoid freezing GUI on startup
        self.signals.update_status.emit("Initializing Brain (Ollama)...")
        self.brain = Brain()

        self.signals.update_status.emit("Initializing Audio Engine (Whisper + Qwen3-TTS)...")
        self.audio = AudioEngine()

        self.signals.update_status.emit("Ready! Press T to start talking.")

        while self.running:
            try:
                if not self.in_conversation:
                    # Waiting for user to press T to start conversation
                    if not self.talk_requested.wait(timeout=1.0):
                        continue
                    self.talk_requested.clear()
                    self.in_conversation = True
                    self.last_interaction_time = time.time()

                # Check for inactivity timeout (30 seconds of silence = reset context)
                if time.time() - self.last_interaction_time > INACTIVITY_TIMEOUT:
                    if self.brain.history:
                        self.brain.reset_context()
                        self.signals.update_status.emit("Context reset due to inactivity. Press T to start a new conversation.")
                    else:
                        self.signals.update_status.emit("No activity. Press T to start talking.")
                    self.in_conversation = False
                    self.last_interaction_time = time.time()
                    continue

                # 1. Listen
                self.signals.update_status.emit("Listening... (speak now)")
                user_text = self.audio.listen_and_transcribe(duration=5)

                if not user_text:
                    # Silence detected - check if we should keep listening or timeout
                    elapsed = time.time() - self.last_interaction_time
                    remaining = INACTIVITY_TIMEOUT - elapsed
                    if remaining > 0:
                        self.signals.update_status.emit(f"Heard silence. Listening... ({int(remaining)}s until timeout)")
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
                def avatar_callback(is_speaking):
                    self.signals.set_speaking.emit(is_speaking)

                self.audio.speak(response, avatar_callback)

                # Update last interaction time after speaking
                self.last_interaction_time = time.time()
                self.signals.update_status.emit("Listening... (speak now)")

            except Exception as e:
                print(f"Error in loop: {e}")
                self.signals.update_status.emit("Error occurred. Press T to try again.")
                self.in_conversation = False
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
        print("[System]:   T     - Press to start conversation")
        print("[System]:   R     - Toggle camera rotation")
        print("[System]:   C     - Toggle click-through mode")
        print("[System]:   Esc   - Quit")
        print("[System]: After speaking, I'll keep listening automatically.")
        print("[System]: Context resets after 30 seconds of silence.")

        try:
            sys.exit(app.exec())
        except KeyboardInterrupt:
            worker.stop()


if __name__ == "__main__":
    main()
