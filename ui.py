import sys
import time

import cv2
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow, QProgressBar, QPushButton, QVBoxLayout, QWidget

from main import DetectionEngine


class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drowsiness Monitor")
        self.resize(1320, 820)
        self.setMinimumSize(1050, 700)

        self.engine = None
        self.running = False
        self.session_start = None

        self.setStyleSheet("""
            QMainWindow, QWidget { background: #0b1017; color: #e8edf3; font-family: Segoe UI; }
            QFrame#panel { background: #121923; border: 1px solid #202b38; border-radius: 16px; }
            QLabel#title { font-size: 24px; font-weight: 700; }
            QLabel#subtitle { color: #8492a3; font-size: 12px; }
            QLabel#status { font-size: 32px; font-weight: 800; }
            QLabel#metric { font-size: 22px; font-weight: 700; }
            QLabel#caption { color: #8492a3; font-size: 11px; }
            QPushButton { background: #1b2634; border: 1px solid #2b3a4b; border-radius: 10px; padding: 10px 18px; font-weight: 600; }
            QPushButton:hover { background: #253346; }
            QPushButton#primary { background: #2b7cff; border: 0; }
            QPushButton#danger { background: #a62d3c; border: 0; }
            QProgressBar { background: #202a35; border: 0; border-radius: 7px; height: 12px; text-align: center; }
            QProgressBar::chunk { background: #2b7cff; border-radius: 7px; }
        """)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(18)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("DROWSINESS MONITOR")
        title.setObjectName("title")
        subtitle = QLabel("Real-time driver attention monitoring")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()
        self.system = QLabel("● SYSTEM READY")
        self.system.setStyleSheet("color:#38d996; font-weight:700;")
        header.addWidget(self.system)
        layout.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(18)
        layout.addLayout(body, 1)

        camera_panel = QFrame()
        camera_panel.setObjectName("panel")
        camera_layout = QVBoxLayout(camera_panel)
        camera_layout.setContentsMargins(12, 12, 12, 12)
        self.camera = QLabel("Camera stopped\n\nPress START MONITORING")
        self.camera.setAlignment(Qt.AlignCenter)
        self.camera.setStyleSheet("background:#070b10; border-radius:12px; color:#657384; font-size:18px;")
        camera_layout.addWidget(self.camera, 1)
        body.addWidget(camera_panel, 3)

        side = QFrame()
        side.setObjectName("panel")
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(24, 24, 24, 24)
        side_layout.setSpacing(18)
        body.addWidget(side, 1)

        lbl = QLabel("DRIVER STATUS")
        lbl.setObjectName("caption")
        side_layout.addWidget(lbl)
        self.status = QLabel("READY")
        self.status.setObjectName("status")
        side_layout.addWidget(self.status)

        self.risk_label = QLabel("Drowsiness risk 0%")
        side_layout.addWidget(self.risk_label)
        self.risk = QProgressBar()
        self.risk.setRange(0, 100)
        self.risk.setValue(0)
        self.risk.setTextVisible(False)
        side_layout.addWidget(self.risk)

        side_layout.addSpacing(8)
        self.eye = self.metric("EYE ASPECT RATIO", "--")
        self.mouth = self.metric("MOUTH ASPECT RATIO", "--")
        self.fps = self.metric("CAMERA", "READY")
        for w in (self.eye, self.mouth, self.fps):
            side_layout.addWidget(w)
        side_layout.addStretch()

        controls = QHBoxLayout()
        self.start_btn = QPushButton("START MONITORING")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.clicked.connect(self.stop)
        self.reset_btn = QPushButton("RESET")
        self.reset_btn.clicked.connect(self.reset)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)
        controls.addWidget(self.reset_btn)
        layout.addLayout(controls)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.frames = 0
        self.last_fps_time = time.monotonic()
        self.current_fps = 0

    def metric(self, caption, value):
        box = QFrame()
        box.setStyleSheet("background:#0e151e; border-radius:10px; padding:8px;")
        lay = QVBoxLayout(box)
        c = QLabel(caption)
        c.setObjectName("caption")
        v = QLabel(value)
        v.setObjectName("metric")
        lay.addWidget(c)
        lay.addWidget(v)
        box.value_label = v
        return box

    def start(self):
        if self.running:
            return
        try:
            self.engine = DetectionEngine()
            self.running = True
            self.session_start = time.monotonic()
            self.system.setText("● MONITORING")
            self.system.setStyleSheet("color:#38d996; font-weight:700;")
            self.start_btn.setEnabled(False)
            self.timer.start(30)
        except Exception as exc:
            self.system.setText(f"● ERROR: {exc}")
            self.system.setStyleSheet("color:#ff6374; font-weight:700;")

    def stop(self):
        self.timer.stop()
        if self.engine:
            self.engine.close()
            self.engine = None
        self.running = False
        self.start_btn.setEnabled(True)
        self.system.setText("● SYSTEM READY")
        self.system.setStyleSheet("color:#38d996; font-weight:700;")
        self.status.setText("READY")
        self.camera.setText("Camera stopped\n\nPress START MONITORING")

    def reset(self):
        if self.engine:
            self.engine.reset()
        self.status.setText("READY")
        self.risk.setValue(0)
        self.risk_label.setText("Drowsiness risk 0%")

    def update_frame(self):
        if not self.engine:
            return
        try:
            result = self.engine.process()
            if result is None:
                self.stop()
                return
            frame, status, ear, mar, score = result

            self.frames += 1
            elapsed = time.monotonic() - self.last_fps_time
            if elapsed >= 1:
                self.current_fps = self.frames / elapsed
                self.frames = 0
                self.last_fps_time = time.monotonic()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image).scaled(
                self.camera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.camera.setPixmap(pixmap)

            self.status.setText(status)
            self.risk.setValue(score)
            self.risk_label.setText(f"Drowsiness risk {score}%")
            self.eye.value_label.setText(f"{ear:.3f}")
            self.mouth.value_label.setText(f"{mar:.3f}")
            self.fps.value_label.setText(f"{self.current_fps:.1f} FPS")

            if status == "DROWSY":
                self.status.setStyleSheet("color:#ff5366; font-size:32px; font-weight:800;")
                self.risk.setStyleSheet("QProgressBar::chunk { background:#ff5366; border-radius:7px; }")
            elif status in ("WARNING", "YAWNING", "HEAD OFF-CENTER"):
                self.status.setStyleSheet("color:#ffb547; font-size:32px; font-weight:800;")
                self.risk.setStyleSheet("QProgressBar::chunk { background:#ffb547; border-radius:7px; }")
            else:
                self.status.setStyleSheet("color:#38d996; font-size:32px; font-weight:800;")
                self.risk.setStyleSheet("QProgressBar::chunk { background:#2b7cff; border-radius:7px; }")
        except Exception as exc:
            self.system.setText(f"● ERROR: {exc}")
            self.system.setStyleSheet("color:#ff6374; font-weight:700;")
            self.stop()

    def closeEvent(self, event):
        self.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Dashboard()
    window.show()
    sys.exit(app.exec())
