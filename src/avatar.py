"""
3D Avatar Widget for CoFoc

OpenGL-based 3D avatar with skeletal animation and transparent background overlay.
Supports loading external glTF/GLB/VRM models.
"""

import sys
import os
import math
import time
from pathlib import Path
from typing import Optional, List, Tuple

from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QSurfaceFormat

try:
    from PyQt6.QtOpenGLWidgets import QOpenGLWidget
    from OpenGL.GL import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("Warning: PyOpenGL not available. Install with: pip install PyOpenGL PyOpenGL-accelerate")

from math3d import Vec3, Mat4, Quaternion, radians
from shaders import Shader, create_avatar_shader
from geometry import Mesh
from lipsync import get_lip_sync_engine, VisemeShape


# Check if we have an external model to load
MODELS_DIR = Path(__file__).parent.parent / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "avatar.glb"


class ExternalAvatarRenderer:
    """Renderer for external glTF/GLB/VRM models."""

    def __init__(self):
        self.meshes: List[Tuple[Mesh, dict]] = []  # (mesh, material_props)
        self.skeleton = None
        self.animations = []
        self.current_animation = None
        self.animation_time = 0.0
        self.model_loaded = False

        # Animation state
        self.is_speaking = False
        self.mouth_blend = 0.0
        self.blink_amount = 0.0
        self.blink_timer = 0.0

        # Model transform (for positioning)
        self.position = Vec3(0, 0, 0)
        self.rotation = Quaternion.identity()
        self.scale = 1.0

    def load_model(self, path: str) -> bool:
        """Load a glTF/GLB/VRM model."""
        try:
            from gltf_loader import GLTFLoader, build_meshes_from_gltf, build_skeleton_from_gltf

            loader = GLTFLoader()
            model = loader.load(path)

            # Build GPU meshes
            mesh_data = build_meshes_from_gltf(model)
            for mesh, material in mesh_data:
                material_props = {
                    'diffuse': material.base_color[:3],
                    'specular': (0.3, 0.3, 0.3),
                    'shininess': 32.0 * (1.0 - material.roughness),
                    'emission': sum(material.emissive) / 3.0,
                }
                self.meshes.append((mesh, material_props))

            # Build skeleton
            if model.skins:
                self.skeleton = build_skeleton_from_gltf(model)
                self.skeleton.reset_pose()
                self.skeleton.set_bind_pose()

            # Store animations
            self.animations = model.animations

            self.model_loaded = True
            print(f"Loaded model with {len(self.meshes)} meshes")
            return True

        except Exception as e:
            print(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def set_speaking(self, speaking: bool) -> None:
        """Set whether the avatar is speaking."""
        self.is_speaking = speaking

    def update(self, delta_time: float) -> None:
        """Update animations with phoneme-based lip sync."""
        if not self.model_loaded:
            return

        # Get lip sync engine for phoneme-based mouth animation
        lip_sync = get_lip_sync_engine()

        # Update mouth animation from lip sync engine
        if lip_sync.is_playing:
            # Get current viseme shape from phoneme timeline
            viseme_shape, _ = lip_sync.get_current_viseme()
            # Use the viseme's mouth openness
            self.mouth_blend = viseme_shape.to_simple()
            # Store full viseme for detailed animation
            self._current_viseme = viseme_shape
        elif self.is_speaking:
            # Fallback to simple animation if lip sync not available
            import random
            target = random.uniform(0.3, 1.0)
            self.mouth_blend += (target - self.mouth_blend) * 10 * delta_time
            self._current_viseme = None
        else:
            self.mouth_blend += (0.0 - self.mouth_blend) * 5 * delta_time
            self._current_viseme = None

        # Update blinking
        self.blink_timer += delta_time
        if self.blink_timer > 3.0:  # Blink every ~3 seconds
            import random
            if random.random() < 0.3:
                self.blink_amount = 1.0
                self.blink_timer = 0.0

        if self.blink_amount > 0:
            self.blink_amount = max(0, self.blink_amount - delta_time * 8)

        # Update skeleton animation
        if self.skeleton:
            self.animation_time += delta_time
            # Add subtle idle animation
            self._apply_procedural_animation(delta_time)
            self.skeleton.update()

    def _apply_procedural_animation(self, delta_time: float) -> None:
        """Apply procedural idle and speaking animations with phoneme-based lip sync."""
        if not self.skeleton:
            return

        t = self.animation_time

        # Breathing
        breath = math.sin(t * 1.5) * 0.5 + 0.5

        # Head movement - more dynamic when speaking
        head = self.skeleton.get_bone("Head") or self.skeleton.get_bone("head")
        if head:
            nod = math.sin(t * 0.8) * 0.02
            tilt = math.sin(t * 0.5) * 0.015
            if self.is_speaking:
                # More expressive head movement when speaking
                nod += math.sin(t * 3) * 0.04
                tilt += math.sin(t * 2.5) * 0.02
            head.local_transform.rotation = Quaternion.from_euler(nod, tilt, 0)

        # Spine breathing
        spine = self.skeleton.get_bone("Spine") or self.skeleton.get_bone("spine")
        if spine:
            spine.local_transform.scale_factor = Vec3(1.0, 1.0 + breath * 0.01, 1.0 + breath * 0.015)

        # Jaw/mouth animation using viseme data
        jaw = self.skeleton.get_bone("Jaw") or self.skeleton.get_bone("jaw")
        if jaw:
            viseme = getattr(self, '_current_viseme', None)
            if viseme and isinstance(viseme, VisemeShape):
                # Use detailed viseme shape for jaw animation
                # jaw_open controls vertical opening
                jaw_rotation = viseme.jaw_open * 0.25  # Max ~15 degrees
                jaw.local_transform.rotation = Quaternion.from_euler(jaw_rotation, 0, 0)
            else:
                # Fallback to simple mouth blend
                jaw.local_transform.rotation = Quaternion.from_euler(self.mouth_blend * 0.2, 0, 0)

        # Lip pucker for rounded vowels (if model supports it)
        # This would require blend shapes which glTF models may have
        # For now, we can at least adjust the jaw differently for different sounds

    def get_bone_matrices(self) -> Optional[List[Mat4]]:
        """Get bone matrices for GPU skinning."""
        if self.skeleton:
            return self.skeleton.get_bone_matrices()
        return None

    def delete(self) -> None:
        """Clean up GPU resources."""
        for mesh, _ in self.meshes:
            mesh.delete()
        self.meshes.clear()


class Avatar3DWidget(QOpenGLWidget):
    """OpenGL widget for rendering the 3D avatar with transparency."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Request alpha channel for transparency
        fmt = self.format()
        fmt.setAlphaBufferSize(8)
        self.setFormat(fmt)

        # OpenGL resources
        self.shader: Optional[Shader] = None
        self.external_renderer: Optional[ExternalAvatarRenderer] = None
        self.fallback_avatar = None  # Will use procedural avatar if no model

        # Camera settings
        self.camera_distance = 1.8
        self.camera_height = 1.0
        self.camera_target = Vec3(0, 0.9, 0)
        self.camera_angle = 0.0

        # Lighting
        self.light_position = Vec3(2, 3, 3)
        self.light_color = Vec3(1.0, 0.98, 0.95)
        self.ambient_color = Vec3(0.2, 0.22, 0.25)

        # Animation state
        self.is_speaking = False
        self.last_time = time.time()

        # Auto-rotation
        self.auto_rotate = False
        self.rotation_speed = 0.3

        # Model path
        self.model_path: Optional[str] = None

    def set_model_path(self, path: str) -> None:
        """Set the model to load."""
        self.model_path = path

    def initializeGL(self) -> None:
        """Initialize OpenGL resources."""
        if not OPENGL_AVAILABLE:
            return

        # Transparent background - alpha = 0 for full transparency
        glClearColor(0.0, 0.0, 0.0, 0.0)

        # Enable depth testing
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)

        # Enable back-face culling
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)

        # Enable alpha blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Enable multisampling
        glEnable(GL_MULTISAMPLE)

        # Create shader
        try:
            self.shader = create_avatar_shader()
        except Exception as e:
            print(f"Shader compilation error: {e}")
            return

        # Try to load external model
        self.external_renderer = ExternalAvatarRenderer()

        model_loaded = False
        if self.model_path and os.path.exists(self.model_path):
            model_loaded = self.external_renderer.load_model(self.model_path)
        elif DEFAULT_MODEL_PATH.exists():
            model_loaded = self.external_renderer.load_model(str(DEFAULT_MODEL_PATH))

        # If no external model, use procedural fallback
        if not model_loaded:
            print("No external model found, using procedural avatar")
            try:
                from avatar_model import HumanoidAvatar
                self.fallback_avatar = HumanoidAvatar()
            except Exception as e:
                print(f"Failed to create fallback avatar: {e}")

    def resizeGL(self, width: int, height: int) -> None:
        """Handle widget resize."""
        if not OPENGL_AVAILABLE:
            return
        glViewport(0, 0, width, height)

    def paintGL(self) -> None:
        """Render the scene with transparent background."""
        if not OPENGL_AVAILABLE or not self.shader:
            return

        # Calculate delta time
        current_time = time.time()
        delta_time = min(current_time - self.last_time, 0.1)  # Cap at 100ms
        self.last_time = current_time

        # Update animations
        if self.external_renderer and self.external_renderer.model_loaded:
            self.external_renderer.update(delta_time)
        elif self.fallback_avatar:
            self.fallback_avatar.update(delta_time)

        # Auto rotation
        if self.auto_rotate:
            self.camera_angle += self.rotation_speed * delta_time

        # Clear with transparency
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Calculate matrices
        aspect = self.width() / max(1, self.height())
        projection = Mat4.perspective(radians(35), aspect, 0.1, 100.0)

        # Camera position
        cam_x = math.sin(self.camera_angle) * self.camera_distance
        cam_z = math.cos(self.camera_angle) * self.camera_distance
        camera_pos = Vec3(cam_x, self.camera_height, cam_z)

        view = Mat4.look_at(camera_pos, self.camera_target, Vec3.up())
        model = Mat4.identity()

        # Use shader
        self.shader.use()

        # Set common uniforms
        self.shader.set_uniform_mat4("uProjection", projection)
        self.shader.set_uniform_mat4("uView", view)
        self.shader.set_uniform_mat4("uModel", model)

        # Set lighting uniforms
        self.shader.set_uniform_vec3("uCameraPos", camera_pos)
        self.shader.set_uniform_vec3("uLightPos", self.light_position)
        self.shader.set_uniform_vec3("uLightColor", self.light_color)
        self.shader.set_uniform_vec3("uAmbientColor", self.ambient_color)

        # Render based on what's available
        if self.external_renderer and self.external_renderer.model_loaded:
            self._render_external_model()
        elif self.fallback_avatar:
            self._render_fallback_avatar()

    def _render_external_model(self) -> None:
        """Render the external glTF model."""
        bone_matrices = self.external_renderer.get_bone_matrices()
        if bone_matrices:
            self.shader.set_uniform_int("uUseSkinning", 1)
            self.shader.set_uniform_mat4_array("uBoneMatrices", bone_matrices)
        else:
            self.shader.set_uniform_int("uUseSkinning", 0)

        for mesh, material_props in self.external_renderer.meshes:
            self.shader.set_uniform_vec3_values("uDiffuseColor", *material_props['diffuse'])
            self.shader.set_uniform_vec3_values("uSpecularColor", *material_props['specular'])
            self.shader.set_uniform_float("uShininess", material_props['shininess'])
            self.shader.set_uniform_float("uEmission", material_props['emission'])
            mesh.draw()

    def _render_fallback_avatar(self) -> None:
        """Render the procedural avatar."""
        bone_matrices = self.fallback_avatar.get_bone_matrices()
        if bone_matrices:
            self.shader.set_uniform_int("uUseSkinning", 1)
            self.shader.set_uniform_mat4_array("uBoneMatrices", bone_matrices)
        else:
            self.shader.set_uniform_int("uUseSkinning", 0)

        for part in self.fallback_avatar.body_parts:
            if not part.visible:
                continue
            self.shader.set_uniform_vec3_values("uDiffuseColor", *part.diffuse_color)
            self.shader.set_uniform_vec3_values("uSpecularColor", *part.specular_color)
            self.shader.set_uniform_float("uShininess", part.shininess)
            self.shader.set_uniform_float("uEmission", part.emission)
            part.mesh.draw()

    def set_speaking(self, speaking: bool) -> None:
        """Set whether the avatar is speaking."""
        self.is_speaking = speaking
        if self.external_renderer:
            self.external_renderer.set_speaking(speaking)
        if self.fallback_avatar:
            self.fallback_avatar.set_speaking(speaking)

    def cleanup(self) -> None:
        """Clean up OpenGL resources."""
        if self.external_renderer:
            self.external_renderer.delete()
        if self.fallback_avatar:
            self.fallback_avatar.delete()
        if self.shader:
            self.shader.delete()


class AvatarWidget(QWidget):
    """
    Main avatar window widget with transparent background overlay.

    The avatar appears in the corner of your screen with no background,
    floating over other windows.
    """

    # Signal emitted when user presses the talk key (T)
    talk_requested = pyqtSignal()

    def __init__(self, model_path: Optional[str] = None):
        super().__init__()

        # Window setup for transparent overlay
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowTransparentForInput  # Click-through when not focused
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

        # Geometry - larger window for better avatar visibility
        self.resize(350, 450)

        # Position in bottom right corner
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.width() - 370, screen.height() - 470)

        # Create OpenGL widget
        self.gl_widget = Avatar3DWidget(self)
        if model_path:
            self.gl_widget.set_model_path(model_path)

        # Layout with no margins
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.gl_widget)

        # State
        self.is_speaking = False
        self._click_through = True

        # Animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update)
        self.timer.start(16)  # ~60 FPS

    def _update(self) -> None:
        """Update animation and trigger repaint."""
        self.gl_widget.update()

    def set_speaking(self, speaking: bool) -> None:
        """Set whether the avatar is speaking."""
        self.is_speaking = speaking
        self.gl_widget.set_speaking(speaking)

    def set_click_through(self, enabled: bool) -> None:
        """Enable/disable click-through mode."""
        self._click_through = enabled
        flags = self.windowFlags()
        if enabled:
            flags |= Qt.WindowType.WindowTransparentForInput
        else:
            flags &= ~Qt.WindowType.WindowTransparentForInput
        self.setWindowFlags(flags)
        self.show()

    def closeEvent(self, event) -> None:
        """Handle window close."""
        self.timer.stop()
        self.gl_widget.cleanup()
        super().closeEvent(event)

    def keyPressEvent(self, event) -> None:
        """Handle key presses."""
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        elif event.key() == Qt.Key.Key_R:
            self.gl_widget.auto_rotate = not self.gl_widget.auto_rotate
        elif event.key() == Qt.Key.Key_Space:
            self.set_speaking(not self.is_speaking)
        elif event.key() == Qt.Key.Key_C:
            # Toggle click-through
            self.set_click_through(not self._click_through)
        elif event.key() == Qt.Key.Key_T:
            # Push-to-talk: emit signal to start listening
            self.talk_requested.emit()
        super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:
        """Handle mouse press for dragging."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        """Handle mouse move for dragging."""
        if event.buttons() & Qt.MouseButton.LeftButton and hasattr(self, '_drag_pos'):
            self.move(event.globalPosition().toPoint() - self._drag_pos)
        super().mouseMoveEvent(event)


def configure_opengl_format() -> None:
    """Configure OpenGL format for the application with transparency support."""
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setAlphaBufferSize(8)  # Important for transparency
    fmt.setSamples(4)  # Multisampling for anti-aliasing
    fmt.setSwapInterval(1)  # VSync
    QSurfaceFormat.setDefaultFormat(fmt)


if __name__ == "__main__":
    # Configure OpenGL before creating QApplication
    configure_opengl_format()

    app = QApplication(sys.argv)

    # Check for command line model path
    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    window = AvatarWidget(model_path)
    window.show()

    # Disable click-through for standalone testing
    window.set_click_through(False)

    # For testing, toggle speaking periodically
    def test_speaking():
        window.set_speaking(not window.is_speaking)

    test_timer = QTimer()
    test_timer.timeout.connect(test_speaking)
    test_timer.start(3000)

    sys.exit(app.exec())
