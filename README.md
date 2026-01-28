# CoFoc

CoFoc is a fully interactive 3D AI assistant that listens to you, thinks using a local LLM (Ollama), and speaks back using Qwen3-TTS, all while displaying a real-time animated 3D avatar floating on your screen with a transparent background.

## Features

- **Transparent Overlay**: Full-body avatar floats on your screen with no background
- **Custom 3D Avatars**: Load high-quality GLB/glTF/VRM models from Ready Player Me, VRoid Hub, Mixamo, and more
- **Natural Voice Detection**: Speak naturally - VAD detects when you're talking and when you stop
- **Rich Animations**: Idle breathing, phoneme-based lip-sync, blinking, and expressive hand gestures during speech
- **Continuous Conversation**: Press T to start, then just keep talking - no need to press again
- **Local AI**: Uses Ollama for privacy-focused local LLM inference
- **Context Management**: Automatic context reset after 30 seconds of silence
- **High-Quality TTS**: Qwen3-TTS provides natural, expressive speech synthesis

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

*   **T**: Start conversation - press once to begin, then just keep talking
*   **Esc**: Quit the application
*   **R**: Toggle camera auto-rotation
*   **Space**: Toggle speaking animation (for testing)
*   **C**: Toggle click-through mode (allow/block mouse clicks)
*   **Drag**: Move the avatar window (when click-through is disabled)

## How It Works

1. **Model Rendering**: The full-body 3D avatar is rendered with a transparent background, floating on your screen
2. **Start Conversation**: Press **T** once to begin the conversation
3. **Natural Speech Detection**: Speak naturally - the system detects when you're talking using voice activity detection (VAD)
4. **Auto-Detect End of Speech**: When you stop talking for 10 seconds, your message is automatically sent
5. **LLM Response**: The transcribed text is sent to Ollama for a response
6. **Animated Response**: The response is spoken using Qwen3-TTS with phoneme-based lip-sync and hand gestures
7. **Auto-Listen**: After speaking, the assistant automatically listens for your next message
8. **Context Reset**: If there's no speech for 30 seconds total, the conversation context resets and waits for T

## Custom Avatars

CoFoc supports loading high-quality 3D avatar models. Place your model in the `models/` folder as `avatar.glb` or specify a path with `--model`.

### Supported Formats
- **GLB** (recommended) - Binary glTF 2.0
- **glTF** - JSON + separate binary files
- **VRM** - VTuber standard format (based on glTF)

### Model Requirements for Lip-Sync

For the best lip-sync animation experience, your model should have:

1. **Skeletal Rig**: A proper bone hierarchy with at minimum:
   - `Head` or `head` bone (for head movement during speech)
   - `Spine` or `spine` bone (for breathing animation)
   - `Jaw` or `jaw` bone (for mouth movement - **most important for lip-sync**)

2. **Recommended Bone Structure**:
   ```
   Root
   └── Hips
       └── Spine
           └── Chest
               └── Neck
                   └── Head
                       └── Jaw (for lip-sync)
   ```

3. **File Size**: Models under 50MB load fastest. Very detailed models may cause initial lag.

### Recommended Models

Here are specific model recommendations that work well with CoFoc:

#### Best for Beginners: Ready Player Me
- **Why**: Automatically includes proper rig with jaw bone
- **Style**: Semi-realistic, customizable
- **Cost**: Free
- **Direct Link**: [readyplayer.me/avatar](https://readyplayer.me/avatar)

**Step-by-step**:
1. Visit readyplayer.me/avatar
2. Create and customize your avatar
3. Click "Next" and then copy the avatar URL (e.g., `https://models.readyplayer.me/6185a4acfb622cf1cdc49348.glb`)
4. Download using the utility:
   ```bash
   python src/download_avatar.py --source readyplayer --id 6185a4acfb622cf1cdc49348
   ```
5. Or download the GLB directly and save as `models/avatar.glb`

#### Best for Anime Style: VRoid Hub
- **Why**: VRM format designed for VTubing with proper lip-sync support
- **Style**: Anime/VTuber
- **Cost**: Free (check individual licenses)
- **Direct Link**: [hub.vroid.com](https://hub.vroid.com)

**Recommended models from VRoid Hub**:
- Search for "free download" + "full body"
- Look for models with "commercial use OK" or "modification OK" licenses
- Popular creators: Pikamee, Ina, or search for "original character"

**Step-by-step**:
1. Browse hub.vroid.com and find a model you like
2. Check the usage terms (許諾範囲)
3. Click download to get the VRM file
4. Move to `models/avatar.vrm`
5. Run: `python src/main.py --model models/avatar.vrm`

#### Best for Realistic Style: Mixamo + Blender
- **Why**: High-quality rigged characters
- **Style**: Realistic, game-ready
- **Cost**: Free
- **Direct Link**: [mixamo.com](https://www.mixamo.com)

**Step-by-step**:
1. Create a free Adobe account and visit mixamo.com
2. Go to "Characters" tab
3. Select a character (e.g., "Kachujin", "X Bot", "Y Bot")
4. Click Download → Format: FBX → Download
5. Open Blender and import the FBX: File → Import → FBX
6. Export as glTF: File → Export → glTF 2.0 → Check "Export Deformation Bones Only"
7. Save as `models/avatar.glb`

#### Best for VTubing: VRoid Studio (Create Your Own)
- **Why**: Full customization, export directly to VRM
- **Style**: Anime
- **Cost**: Free
- **Direct Link**: [vroid.com/en/studio](https://vroid.com/en/studio)

**Step-by-step**:
1. Download and install VRoid Studio
2. Create your character (face, hair, body, clothes)
3. Go to "Export" in the top menu
4. Choose "Export as VRM"
5. Save to `models/avatar.vrm`

### Quick Import Guide

```bash
# Method 1: Use the download utility
python src/download_avatar.py --info  # See all options

# Method 2: Ready Player Me (easiest)
python src/download_avatar.py --source readyplayer --id YOUR_AVATAR_ID

# Method 3: Direct URL download
python src/download_avatar.py --url https://example.com/model.glb

# Method 4: Manual placement
# Simply copy your GLB/VRM file to: models/avatar.glb

# Run with custom model
python src/main.py --model path/to/your/model.glb
```

### Troubleshooting Models

| Issue | Solution |
|-------|----------|
| Model not showing | Verify the file is valid GLB/glTF/VRM with a 3D viewer like [gltf-viewer.donmccurdy.com](https://gltf-viewer.donmccurdy.com/) |
| Lip-sync not working | Model needs a "Jaw" or "jaw" bone in the skeleton |
| Model appears too small/large | Model scale is auto-detected; if issues persist, rescale in Blender |
| Model appears dark | Check that the model has proper materials/textures |
| T-pose stuck | Model needs a valid skeleton; try a different source |

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
