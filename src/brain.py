import ollama
import logging


class Brain:
    def __init__(self, model_name="MeanAI:Latest"):
        self.model_name = model_name
        self.history = []
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("Brain")

    def reset_context(self):
        """Reset the conversation history/context."""
        self.history = []
        self.logger.info("Conversation context has been reset")

    def think(self, user_input):
        """Sends user input to Ollama and gets a response."""
        self.logger.info(f"Thinking about: {user_input}")

        try:
            # Simple chat history management
            self.history.append({"role": "user", "content": user_input})

            response = ollama.chat(model=self.model_name, messages=self.history)

            bot_response = response["message"]["content"]
            self.history.append({"role": "assistant", "content": bot_response})

            # Prune history if too long
            if len(self.history) > 10:
                self.history = self.history[-10:]

            return bot_response

        except Exception as e:
            self.logger.error(f"Error communicating with Ollama: {e}")
            return "I'm having trouble thinking right now. Is Ollama running?"
