import sys
import time

import cv2
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QFileDialog,
    QMessageBox,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from main import DetectionEngine, analyze_photo


class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drowsiness Monitor")
        self.resize(1500, 900)
        self.setMinimumSize(1180, 760)

        self.engine = None
        self.running = False
        self.session_start = None

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #0b1017;
                color: #e8edf3;
                font-family: "Segoe UI";
            }

            QFrame#panel {
                background: #121923;
                border: 1px solid #243140;
                border-radius: 16px;
            }

            QFrame#metricCard {
                background: #0d141d;
                border: 1px solid #1d2936;
                border-radius: 12px;
            }

            QLabel#title {
                font-size: 26px;
                font-weight: 700;
                color: #f1f5f9;
            }

            QLabel#subtitle {
                color: #8492a3;
                font-size: 13px;
            }

            QLabel#section {
                color: #8fa0b3;
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 1px;
            }

            QLabel#status {
                font-size: 34px;
                font-weight: 800;
            }

            QLabel#metric {
                color: #f1f5f9;
                font-size: 21px;
                font-weight: 700;
            }

            QLabel#caption {
                color: #7f90a3;
                font-size: 10px;
                font-weight: 600;
                letter-spacing: 0.6px;
            }

            QLabel#session {
                color: #9aa9ba;
                font-size: 11px;
            }

            QLabel#cameraPlaceholder {
                background: #070b10;
                border-radius: 11px;
                color: #657384;
                font-size: 17px;
            }

            QPushButton {
                background: #1a2633;
                color: #e8edf3;
                border: 1px solid #2b3a4b;
                border-radius: 10px;
                padding: 12px 18px;
                min-height: 20px;
                font-size: 13px;
                font-weight: 600;
            }

            QPushButton:hover {
                background: #253346;
            }

            QPushButton:disabled {
                background: #18212c;
                color: #566474;
                border-color: #202b37;
            }

            QPushButton#primary {
                background: #2b7cff;
                border: 1px solid #2b7cff;
            }

            QPushButton#primary:hover {
                background: #3b88ff;
            }

            QPushButton#danger {
                background: #a62d3c;
                border: 1px solid #a62d3c;
            }

            QPushButton#danger:hover {
                background: #bd3446;
            }

            QProgressBar {
                background: #202b37;
                border: 0;
                border-radius: 6px;
                height: 12px;
            }

            QProgressBar::chunk {
                background: #2b7cff;
                border-radius: 6px;
            }
        """)

        root = QWidget()
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(30, 26, 30, 28)
        layout.setSpacing(18)

        # Header
        header = QHBoxLayout()
        header.setSpacing(16)

        title_box = QVBoxLayout()
        title_box.setSpacing(4)

        title = QLabel("DROWSINESS MONITOR")
        title.setObjectName("title")

        subtitle = QLabel("Real-time driver attention monitoring")
        subtitle.setObjectName("subtitle")

        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()

        self.system = QLabel("●  SYSTEM READY")
        self.system.setStyleSheet("color:#38d996; font-size:12px; font-weight:700;")
        self.system.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header.addWidget(self.system)

        layout.addLayout(header)

        # Main content
        body = QHBoxLayout()
        body.setSpacing(18)
        layout.addLayout(body, 1)

        # Camera panel
        camera_panel = QFrame()
        camera_panel.setObjectName("panel")
        camera_panel.setMinimumWidth(650)

        camera_layout = QVBoxLayout(camera_panel)
        camera_layout.setContentsMargins(12, 12, 12, 12)
        camera_layout.setSpacing(0)

        self.camera = QLabel("Camera stopped\n\nPress START MONITORING")
        self.camera.setObjectName("cameraPlaceholder")
        self.camera.setAlignment(Qt.AlignCenter)
        self.camera.setMinimumSize(560, 360)
        self.camera.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        camera_layout.addWidget(self.camera)

        body.addWidget(camera_panel, 1)

        # Right information panel
        side = QFrame()
        side.setObjectName("panel")
        side.setMinimumWidth(360)
        side.setMaximumWidth(440)

        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(22, 20, 22, 20)
        side_layout.setSpacing(12)

        section = QLabel("DRIVER STATUS")
        section.setObjectName("section")
        side_layout.addWidget(section)

        self.status = QLabel("READY")
        self.status.setObjectName("status")
        self.status.setMinimumHeight(58)
        self.status.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        side_layout.addWidget(self.status)

        self.risk_label = QLabel("Drowsiness risk 0%")
        self.risk_label.setObjectName("session")
        side_layout.addWidget(self.risk_label)

        self.risk = QProgressBar()
        self.risk.setRange(0, 100)
        self.risk.setValue(0)
        self.risk.setTextVisible(False)
        self.risk.setMinimumHeight(12)
        side_layout.addWidget(self.risk)

        session_row = QHBoxLayout()
        session_title = QLabel("SESSION")
        session_title.setObjectName("caption")
        self.session_time = QLabel("00:00")
        self.session_time.setObjectName("session")
        self.session_time.setAlignment(Qt.AlignRight)
        session_row.addWidget(session_title)
        session_row.addStretch()
        session_row.addWidget(self.session_time)
        side_layout.addLayout(session_row)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet("color:#202c39;")
        side_layout.addWidget(divider)

        # Metric cards
        self.eye = self.make_metric("EYE ASPECT RATIO", "--")
        self.mouth = self.make_metric("MOUTH ASPECT RATIO", "--")
        self.fps = self.make_metric("CAMERA FPS", "--")

        for widget in (self.eye, self.mouth, self.fps):
            side_layout.addWidget(widget)

        side_layout.addStretch(1)
        body.addWidget(side, 0)

        # Bottom controls
        controls = QHBoxLayout()
        controls.setSpacing(12)

        self.photo_btn = QPushButton("CHECK PHOTO")
        self.photo_btn.clicked.connect(self.check_photo)

        self.start_btn = QPushButton("START MONITORING")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self.start)

        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.clicked.connect(self.stop)

        self.reset_btn = QPushButton("RESET")
        self.reset_btn.clicked.connect(self.reset)

        for button in (self.start_btn, self.stop_btn, self.reset_btn, self.photo_btn):
            button.setMinimumHeight(46)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            controls.addWidget(button, 1)

        layout.addLayout(controls)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.frames = 0
        self.last_fps_time = time.monotonic()
        self.current_fps = 0.0

    def make_metric(self, caption, value):
        box = QFrame()
        box.setObjectName("metricCard")
        box.setMinimumHeight(82)
        box.setMaximumHeight(96)
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        lay = QVBoxLayout(box)
        lay.setContentsMargins(14, 11, 14, 11)
        lay.setSpacing(5)

        c = QLabel(caption)
        c.setObjectName("caption")
        c.setMinimumHeight(16)
        c.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        v = QLabel(value)
        v.setObjectName("metric")
        v.setMinimumHeight(28)
        v.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        v.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        lay.addWidget(c)
        lay.addWidget(v)

        box.value_label = v
        return box

    def check_photo(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Photo",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)",
        )
        if not path:
            return

        try:
            frame, status, ear, mar, score = analyze_photo(path)

            self.status.setText(status)
            self.risk.setValue(int(score))
            self.risk_label.setText(f"Photo risk {int(score)}%")
            self.eye.value_label.setText(f"{ear:.3f}" if ear else "--")
            self.mouth.value_label.setText(f"{mar:.3f}" if mar else "--")
            self.fps.value_label.setText("PHOTO")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            image = QImage(
                rgb.data, w, h, ch * w, QImage.Format_RGB888
            ).copy()
            target = self.camera.size()
            pixmap = QPixmap.fromImage(image).scaled(
                max(1, target.width() - 2),
                max(1, target.height() - 2),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.camera.setPixmap(pixmap)

            if status == "DROWSY":
                color = "#ff5366"
                message = "DROWSINESS DETECTED"
            elif status == "EYES CLOSED":
                color = "#ffb547"
                message = "EYES CLOSED — PHOTO CANNOT CONFIRM DROWSINESS"
            elif status == "YAWNING":
                color = "#ffb547"
                message = "YAWNING / FATIGUE SIGNAL"
            elif status == "NO FACE":
                color = "#9aa9ba"
                message = "NO FACE DETECTED"
            else:
                color = "#38d996"
                message = "PERSON APPEARS AWAKE"

            self.status.setStyleSheet(
                f"color:{color}; font-size:34px; font-weight:800;"
            )
            QMessageBox.information(
                self,
                "Photo Check",
                f"{message}\n\nRisk score: {int(score)}%\nEAR: {ear:.3f}\nMAR: {mar:.3f}",
            )

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Photo Check Error",
                f"Could not analyze this photo.\n\n{exc}",
            )

    def start(self):
        if self.running:
            return

        try:
            self.engine = DetectionEngine()
            self.running = True
            self.session_start = time.monotonic()
            self.frames = 0
            self.last_fps_time = time.monotonic()
            self.current_fps = 0.0

            self.system.setText("●  MONITORING")
            self.system.setStyleSheet(
                "color:#38d996; font-size:12px; font-weight:700;"
            )

            self.start_btn.setEnabled(False)
            self.timer.start(30)

        except Exception as exc:
            self.system.setText("●  CAMERA ERROR")
            self.system.setStyleSheet(
                "color:#ff6374; font-size:12px; font-weight:700;"
            )
            self.camera.setText(f"Unable to start camera\n\n{exc}")

    def stop(self):
        self.timer.stop()

        if self.engine:
            self.engine.close()
            self.engine = None

        self.running = False
        self.session_start = None
        self.start_btn.setEnabled(True)

        self.system.setText("●  SYSTEM READY")
        self.system.setStyleSheet(
            "color:#38d996; font-size:12px; font-weight:700;"
        )

        self.status.setText("READY")
        self.status.setStyleSheet(
            "color:#38d996; font-size:34px; font-weight:800;"
        )

        self.risk.setValue(0)
        self.risk_label.setText("Drowsiness risk 0%")
        self.session_time.setText("00:00")
        self.eye.value_label.setText("--")
        self.mouth.value_label.setText("--")
        self.fps.value_label.setText("--")
        self.camera.clear()
        self.camera.setText("Camera stopped\n\nPress START MONITORING")

    def reset(self):
        if self.engine:
            self.engine.reset()

        self.status.setText("READY")
        self.status.setStyleSheet(
            "color:#38d996; font-size:34px; font-weight:800;"
        )
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

            # FPS calculation
            self.frames += 1
            elapsed = time.monotonic() - self.last_fps_time

            if elapsed >= 1.0:
                self.current_fps = self.frames / elapsed
                self.frames = 0
                self.last_fps_time = time.monotonic()

            # Session timer
            if self.session_start is not None:
                seconds = int(time.monotonic() - self.session_start)
                minutes, secs = divmod(seconds, 60)
                hours, minutes = divmod(minutes, 60)
                self.session_time.setText(
                    f"{hours:02d}:{minutes:02d}:{secs:02d}"
                    if hours
                    else f"{minutes:02d}:{secs:02d}"
                )

            # Camera image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape

            image = QImage(
                rgb.data,
                w,
                h,
                ch * w,
                QImage.Format_RGB888,
            ).copy()

            target = self.camera.size()
            pixmap = QPixmap.fromImage(image).scaled(
                max(1, target.width() - 2),
                max(1, target.height() - 2),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.camera.setPixmap(pixmap)

            # Dashboard values
            self.status.setText(status)
            self.risk.setValue(int(score))
            self.risk_label.setText(f"Drowsiness risk {int(score)}%")
            self.eye.value_label.setText(f"{ear:.3f}")
            self.mouth.value_label.setText(f"{mar:.3f}")
            self.fps.value_label.setText(f"{self.current_fps:.1f} FPS")

            if status == "DROWSY":
                self.status.setStyleSheet(
                    "color:#ff5366; font-size:34px; font-weight:800;"
                )
                self.risk.setStyleSheet(
                    "QProgressBar::chunk { background:#ff5366; border-radius:6px; }"
                )

            elif status in ("WARNING", "YAWNING", "HEAD OFF-CENTER"):
                self.status.setStyleSheet(
                    "color:#ffb547; font-size:34px; font-weight:800;"
                )
                self.risk.setStyleSheet(
                    "QProgressBar::chunk { background:#ffb547; border-radius:6px; }"
                )

            elif status == "NO FACE":
                self.status.setStyleSheet(
                    "color:#9aa9ba; font-size:34px; font-weight:800;"
                )
                self.risk.setStyleSheet(
                    "QProgressBar::chunk { background:#607084; border-radius:6px; }"
                )

            else:
                self.status.setStyleSheet(
                    "color:#38d996; font-size:34px; font-weight:800;"
                )
                self.risk.setStyleSheet(
                    "QProgressBar::chunk { background:#2b7cff; border-radius:6px; }"
                )

        except Exception as exc:
            self.system.setText("●  DETECTOR ERROR")
            self.system.setStyleSheet(
                "color:#ff6374; font-size:12px; font-weight:700;"
            )
            self.stop()
            self.camera.setText(f"Detector error\n\n{exc}")

    def closeEvent(self, event):
        self.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Keep text rendering consistent on Windows.
    app.setFont(QFont("Segoe UI", 10))

    window = Dashboard()
    window.show()
    sys.exit(app.exec())
