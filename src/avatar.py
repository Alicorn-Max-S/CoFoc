import sys
import math
import random
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPointF, QRectF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QRadialGradient

class AvatarWidget(QWidget):
    def __init__(self):
        super().__init__()
        
        # Window setup for transparency and overlay
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Geometry
        self.resize(400, 500)
        # Position in bottom right corner
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.width() - 420, screen.height() - 520)

        # State
        self.is_speaking = False
        self.mouth_openness = 0.0
        self.blink_state = 0.0 # 0 = open, 1 = closed
        self.arm_angle = 0.0
        self.time = 0.0
        
        # Animation Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(16) # ~60 FPS

    def set_speaking(self, speaking):
        self.is_speaking = speaking

    def update_animation(self):
        self.time += 0.1
        
        # Idle Animation (Bobbing)
        self.bob_offset = math.sin(self.time * 0.5) * 5
        
        # Arm Waving (Articulating Limbs)
        self.arm_angle = math.sin(self.time * 2) * 15
        if self.is_speaking:
            self.arm_angle += math.sin(self.time * 5) * 10 # Excited arms when talking

        # Mouth Animation
        if self.is_speaking:
            # Random jaw movement smoothed
            target = random.random()
            self.mouth_openness += (target - self.mouth_openness) * 0.3
        else:
            self.mouth_openness += (0 - self.mouth_openness) * 0.2

        # Blinking
        if random.random() < 0.01:
            self.blink_state = 1.0
        if self.blink_state > 0:
            self.blink_state -= 0.1
        
        self.update() # Trigger repaint

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Colors
        skin_color = QColor("#FFDAB9")
        robot_color = QColor("#E0E0E0")
        dark_gray = QColor("#404040")
        
        # Center of drawing
        cx = self.width() / 2
        cy = self.height() / 2 + self.bob_offset

        # --- BODY ---
        painter.setBrush(robot_color)
        painter.setPen(QPen(dark_gray, 3))
        body_rect = QRectF(cx - 50, cy, 100, 120)
        painter.drawRoundedRect(body_rect, 15, 15)

        # --- ARMS (Articulating) ---
        self.draw_arm(painter, cx - 50, cy + 20, -45 - self.arm_angle, True) # Left
        self.draw_arm(painter, cx + 50, cy + 20, 45 + self.arm_angle, False) # Right

        # --- HEAD ---
        head_rect = QRectF(cx - 60, cy - 100, 120, 90)
        painter.setBrush(robot_color)
        painter.drawRoundedRect(head_rect, 20, 20)

        # --- EYES ---
        eye_y = cy - 70
        eye_h = 20 * (1.0 - max(0, self.blink_state)) # Blink squish
        
        painter.setBrush(QColor("black"))
        # Left Eye
        painter.drawEllipse(QPointF(cx - 25, eye_y), 10, eye_h/2)
        # Right Eye
        painter.drawEllipse(QPointF(cx + 25, eye_y), 10, eye_h/2)

        # --- MOUTH ---
        mouth_w = 40
        mouth_h = 5 + (20 * self.mouth_openness)
        mouth_rect = QRectF(cx - mouth_w/2, cy - 40, mouth_w, mouth_h)
        painter.setBrush(dark_gray)
        painter.drawRoundedRect(mouth_rect, 5, 5)

    def draw_arm(self, painter, x, y, angle, is_left):
        painter.save()
        painter.translate(x, y)
        painter.rotate(angle)
        
        # Arm segment
        painter.setBrush(QColor("#C0C0C0"))
        painter.drawRoundedRect(QRectF(-10, 0, 20, 70), 8, 8)
        
        # Hand
        painter.setBrush(QColor("#A0A0A0"))
        painter.drawEllipse(QPointF(0, 75), 12, 12)
        
        painter.restore()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AvatarWidget()
    window.show()
    sys.exit(app.exec())
