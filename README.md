# CoFoc

CoFoc is a fully interactive 3D AI assistant that listens to you, thinks using a local LLM (Ollama), and speaks back using the Qwen-3 TTS model, all while displaying a real-time animated 3D avatar on your screen.

## Features

- **3D Animated Avatar**: A fully rigged 3D humanoid model with skeletal animation, realistic lighting, and smooth movements
- **Voice Interaction**: Speak naturally and get spoken responses
- **Local AI**: Uses Ollama for privacy-focused local LLM inference
- **Real-time Animation**: Idle breathing, lip-sync during speech, blinking, and gesture animations

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

*   **Esc**: Quit the application
*   **R**: Toggle camera auto-rotation
*   **Space**: Toggle speaking animation (for testing)
*   **Interaction**: The avatar continuously listens. Speak clearly.

## 3D Avatar System

The avatar is built with a custom OpenGL rendering engine featuring:

### Architecture
- **Math3D**: Vector, matrix, and quaternion utilities for 3D transformations
- **Shaders**: GLSL shaders with Phong lighting, rim effects, and skeletal animation support
- **Geometry**: Procedural mesh generation (spheres, cylinders, capsules, boxes)
- **Animation**: Skeletal animation system with bone hierarchy, keyframe interpolation, and animation blending
- **Avatar Model**: Fully rigged humanoid with 17 bones and multiple body parts

### Visual Features
- Real-time skeletal animation
- Idle breathing animation
- Speaking gestures and head movement
- Automatic blinking
- Phong lighting with rim highlights
- Anti-aliased rendering with multisampling

### Bone Structure
```
root
└── hips
    └── spine
        └── chest
            ├── neck
            │   └── head
            │       ├── jaw
            │       ├── eye_l
            │       └── eye_r
            ├── shoulder_l
            │   └── upper_arm_l
            │       └── forearm_l
            │           └── hand_l
            └── shoulder_r
                └── upper_arm_r
                    └── forearm_r
                        └── hand_r
```

## Project Structure

```
CoFoc/
├── src/
│   ├── main.py           # Application entry point
│   ├── avatar.py         # 3D avatar widget (QOpenGLWidget)
│   ├── avatar_model.py   # Humanoid model construction
│   ├── animation.py      # Skeletal animation system
│   ├── geometry.py       # 3D mesh primitives
│   ├── shaders.py        # GLSL shader programs
│   ├── math3d.py         # 3D math utilities
│   ├── brain.py          # LLM interface (Ollama)
│   └── audio_engine.py   # Speech recognition and synthesis
├── requirements.txt
└── README.md
```