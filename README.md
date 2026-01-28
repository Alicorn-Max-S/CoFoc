# CoFoc

CoFoc is a fully interactive 3D AI assistant that listens to you, thinks using a local LLM (Ollama), and speaks back using the Qwen-3 TTS model, all while displaying a real-time animated 3D avatar floating on your screen with a transparent background.

## Features

- **Transparent Overlay**: Avatar floats on your screen with no background, always visible in the corner
- **Custom 3D Avatars**: Load high-quality GLB/glTF/VRM models from Ready Player Me, VRoid Hub, Mixamo, and more
- **Real-time Animation**: Idle breathing, lip-sync during speech, blinking, and gesture animations
- **Voice Interaction**: Speak naturally and get spoken responses
- **Local AI**: Uses Ollama for privacy-focused local LLM inference

## Prerequisites

1.  **Ollama**: You must have Ollama installed and running.
    *   Download from [ollama.com](https://ollama.com).
    *   Run `ollama run llama3` (or your preferred model) in a separate terminal.

2.  **Python 3.10+**: Ensure you have a capable Python environment.

3.  **Hardware**: NVIDIA GPU recommended for the TTS and STT models (CUDA).

## Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  (Optional) Download a custom avatar:
    ```bash
    # See avatar sources below
    python src/download_avatar.py --info
    ```

3.  Run the application:
    ```bash
    python src/main.py
    ```

    Or with a custom model:
    ```bash
    python src/main.py --model models/avatar.glb
    ```

    Or in test mode (no AI, just avatar):
    ```bash
    python src/main.py --no-ai
    ```

## Controls

*   **Esc**: Quit the application
*   **R**: Toggle camera auto-rotation
*   **Space**: Toggle speaking animation (for testing)
*   **C**: Toggle click-through mode (allow/block mouse clicks)
*   **Drag**: Move the avatar window (when click-through is disabled)

## Custom Avatars

CoFoc supports loading high-quality 3D avatar models. Place your model in the `models/` folder as `avatar.glb` or specify a path with `--model`.

### Supported Formats
- **GLB** (recommended) - Binary glTF 2.0
- **glTF** - JSON + separate binary files
- **VRM** - VTuber standard format (based on glTF)

### Avatar Sources

#### 1. Ready Player Me (Easiest)
Create a free customizable avatar at [readyplayer.me](https://readyplayer.me/avatar)

```bash
# Download your avatar (get ID from the avatar URL)
python src/download_avatar.py --source readyplayer --id YOUR_AVATAR_ID
```

#### 2. VRoid Hub (Best for Anime/VTuber Style)
Browse thousands of free VRM models at [hub.vroid.com](https://hub.vroid.com)

- Download VRM files directly
- Check license terms for each model
- Place in `models/` folder as `avatar.vrm`

#### 3. VRoid Studio (Create Your Own)
Free avatar creation software from [vroid.com](https://vroid.com/en/studio)

- Design your own anime-style character
- Export as VRM format
- Full customization of appearance

#### 4. Mixamo (Realistic Characters)
Free rigged characters at [mixamo.com](https://www.mixamo.com)

- Download as FBX
- Convert to glTF using Blender
- Includes animations

#### 5. Sketchfab
Search for free models at [sketchfab.com](https://sketchfab.com/tags/humanoid)

- Look for "humanoid rigged" models
- Download in glTF/GLB format
- Check license before use

### Download Utility

```bash
# Show all avatar source options
python src/download_avatar.py --info

# Download from Ready Player Me
python src/download_avatar.py --source readyplayer --id 6185a4acfb622cf1cdc49348

# Download from any URL
python src/download_avatar.py --url https://example.com/model.glb
```

## 3D Rendering System

The avatar is rendered using a custom OpenGL engine with:

### Features
- **Transparent Background**: Avatar floats over your desktop
- **Skeletal Animation**: Full bone hierarchy with smooth interpolation
- **Phong Lighting**: Realistic shading with rim highlights
- **GPU Skinning**: Efficient vertex skinning on the GPU
- **Anti-aliasing**: Multisampled rendering for smooth edges

### Architecture
- **Math3D**: Vector, matrix, and quaternion utilities
- **Shaders**: GLSL 3.3 with skeletal animation support
- **Geometry**: Procedural and loaded mesh support
- **Animation**: Keyframe and procedural animation blending
- **glTF Loader**: Full glTF 2.0/GLB/VRM support

## Project Structure

```
CoFoc/
├── models/              # Place avatar models here
│   └── avatar.glb       # Default avatar (auto-loaded)
├── src/
│   ├── main.py          # Application entry point
│   ├── avatar.py        # 3D avatar widget (QOpenGLWidget)
│   ├── avatar_model.py  # Procedural humanoid model
│   ├── gltf_loader.py   # glTF/GLB/VRM model loader
│   ├── download_avatar.py # Avatar download utility
│   ├── animation.py     # Skeletal animation system
│   ├── geometry.py      # 3D mesh primitives
│   ├── shaders.py       # GLSL shader programs
│   ├── math3d.py        # 3D math utilities
│   ├── brain.py         # LLM interface (Ollama)
│   └── audio_engine.py  # Speech recognition and synthesis
├── requirements.txt
└── README.md
```

## Troubleshooting

### Avatar not showing
- Ensure PyOpenGL is installed: `pip install PyOpenGL PyOpenGL-accelerate`
- Check that your GPU supports OpenGL 3.3+

### Model not loading
- Verify the file is a valid GLB/glTF/VRM file
- Try a different model from Ready Player Me
- Check console for error messages

### Transparent background not working
- This feature may not work on all Linux window managers
- Try running with a compositing window manager (e.g., picom)

## Credits

- [Ready Player Me](https://readyplayer.me) - Avatar creation platform
- [VRoid Hub](https://hub.vroid.com) - VRM model community
- [Mixamo](https://mixamo.com) - Character animations
- [Ollama](https://ollama.com) - Local LLM runtime
