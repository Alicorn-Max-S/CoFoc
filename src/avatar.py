"""
3D Avatar Widget for CoFoc

OpenGL-based 3D avatar with skeletal animation, replacing the 2D version.
"""

import sys
import math
import time
from typing import Optional

from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("Warning: PyOpenGL not available. Install with: pip install PyOpenGL PyOpenGL-accelerate")

from math3d import Vec3, Mat4, radians
from shaders import Shader, create_avatar_shader
from avatar_model import HumanoidAvatar, AvatarBodyPart


class Avatar3DWidget(QOpenGLWidget):
    """OpenGL widget for rendering the 3D avatar."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # OpenGL resources
        self.shader: Optional[Shader] = None
        self.avatar: Optional[HumanoidAvatar] = None

        # Camera settings
        self.camera_distance = 2.5
        self.camera_height = 1.5
        self.camera_target = Vec3(0, 1.4, 0)
        self.camera_angle = 0.0

        # Lighting
        self.light_position = Vec3(2, 3, 3)
        self.light_color = Vec3(1.0, 0.98, 0.95)
        self.ambient_color = Vec3(0.15, 0.18, 0.22)

        # Animation state
        self.is_speaking = False
        self.last_time = time.time()

        # Auto-rotation
        self.auto_rotate = False
        self.rotation_speed = 0.3

    def initializeGL(self) -> None:
        """Initialize OpenGL resources."""
        if not OPENGL_AVAILABLE:
            return

        # Set clear color (dark blue-gray background)
        glClearColor(0.1, 0.12, 0.15, 0.0)

        # Enable depth testing
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)

        # Enable back-face culling
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)

        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Enable multisampling if available
        glEnable(GL_MULTISAMPLE)

        # Create shader
        try:
            self.shader = create_avatar_shader()
        except Exception as e:
            print(f"Shader compilation error: {e}")
            return

        # Create avatar
        try:
            self.avatar = HumanoidAvatar()
        except Exception as e:
            print(f"Avatar creation error: {e}")
            return

    def resizeGL(self, width: int, height: int) -> None:
        """Handle widget resize."""
        if not OPENGL_AVAILABLE:
            return

        glViewport(0, 0, width, height)

    def paintGL(self) -> None:
        """Render the scene."""
        if not OPENGL_AVAILABLE or not self.shader or not self.avatar:
            return

        # Calculate delta time
        current_time = time.time()
        delta_time = current_time - self.last_time
        self.last_time = current_time

        # Update animations
        self.avatar.update(delta_time)

        # Auto rotation
        if self.auto_rotate:
            self.camera_angle += self.rotation_speed * delta_time

        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Calculate matrices
        aspect = self.width() / max(1, self.height())
        projection = Mat4.perspective(radians(45), aspect, 0.1, 100.0)

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

        # Upload bone matrices
        bone_matrices = self.avatar.get_bone_matrices()
        if bone_matrices:
            self.shader.set_uniform_int("uUseSkinning", 1)
            self.shader.set_uniform_mat4_array("uBoneMatrices", bone_matrices)
        else:
            self.shader.set_uniform_int("uUseSkinning", 0)

        # Render all body parts
        for part in self.avatar.body_parts:
            if not part.visible:
                continue

            # Set material uniforms
            self.shader.set_uniform_vec3_values("uDiffuseColor", *part.diffuse_color)
            self.shader.set_uniform_vec3_values("uSpecularColor", *part.specular_color)
            self.shader.set_uniform_float("uShininess", part.shininess)
            self.shader.set_uniform_float("uEmission", part.emission)

            # Draw mesh
            part.mesh.draw()

    def set_speaking(self, speaking: bool) -> None:
        """Set whether the avatar is speaking."""
        self.is_speaking = speaking
        if self.avatar:
            self.avatar.set_speaking(speaking)

    def cleanup(self) -> None:
        """Clean up OpenGL resources."""
        if self.avatar:
            self.avatar.delete()
        if self.shader:
            self.shader.delete()


class AvatarWidget(QWidget):
    """
    Main avatar window widget.

    Provides the same interface as the old 2D AvatarWidget for compatibility.
    """

    def __init__(self):
        super().__init__()

        # Window setup for transparency and overlay
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Geometry
        self.resize(400, 500)

        # Position in bottom right corner
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.width() - 420, screen.height() - 520)

        # Create OpenGL widget
        self.gl_widget = Avatar3DWidget(self)

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.gl_widget)

        # State (for compatibility with old interface)
        self.is_speaking = False

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
            # Toggle auto-rotation
            self.gl_widget.auto_rotate = not self.gl_widget.auto_rotate
        elif event.key() == Qt.Key.Key_Space:
            # Toggle speaking for testing
            self.set_speaking(not self.is_speaking)
        super().keyPressEvent(event)


def configure_opengl_format() -> None:
    """Configure OpenGL format for the application."""
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)  # Multisampling for anti-aliasing
    fmt.setSwapInterval(1)  # VSync
    QSurfaceFormat.setDefaultFormat(fmt)


if __name__ == "__main__":
    # Configure OpenGL before creating QApplication
    configure_opengl_format()

    app = QApplication(sys.argv)

    window = AvatarWidget()
    window.show()

    # For testing, toggle speaking periodically
    def test_speaking():
        window.set_speaking(not window.is_speaking)

    test_timer = QTimer()
    test_timer.timeout.connect(test_speaking)
    test_timer.start(3000)

    sys.exit(app.exec())
