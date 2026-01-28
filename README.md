# AI Avatar Assistant

This project implements a fully interactive AI avatar that listens to you, thinks using a local LLM (Ollama), and speaks back using the Qwen-3 TTS model, all while displaying an articulating avatar on your screen.

## Prerequisites

1.  **Ollama**: You must have Ollama installed and running.
    *   Download from [ollama.com](https://ollama.com).
    *   Run `ollama run llama3` (or your preferred model) in a separate terminal to ensure the model is downloaded.
    *   Keep the Ollama server running (usually runs in the background).

2.  **Python 3.10+**: Ensure you have a capable Python environment.

3.  **Hardware**: NVIDIA GPU recommended for the TTS and STT models (CUDA).

## Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Run the application:
    ```bash
    python src/main.py
    ```

## Controls

*   **Esc**: Quit the application.
*   **Interaction**: The avatar continuously listens. Speak clearly.