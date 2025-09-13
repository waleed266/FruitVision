import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout,
    QHBoxLayout, QFrame, QFileDialog
)
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt
import subprocess

# Paths to scripts
WEBCAM_SCRIPT = "webcam_yolo.py"
SPLIT_SCRIPT = "realsense_split_yolo.py"
IMAGE_DETECTION_SCRIPT = "image_detection.py"  # Path to the new image detection script
LOGO_PATH = "logo.png"  # your logo file

class FruitVisionGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("FruitVision")
        self.setGeometry(200, 200, 900, 700)
        self.setStyleSheet("background-color: #1E1E1E;")

        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # -----------------------------
        # Top: logo + project name
        # -----------------------------
        top_layout = QHBoxLayout()
        logo_label = QLabel()
        pixmap = QPixmap(LOGO_PATH).scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label.setPixmap(pixmap)
        logo_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        top_layout.addWidget(logo_label, alignment=Qt.AlignLeft)

        top_layout.addStretch()
        title_label = QLabel("FruitVision")
        title_label.setFont(QFont("Arial", 36, QFont.Bold))
        title_label.setStyleSheet("color: #32CD32;")
        title_label.setAlignment(Qt.AlignCenter)
        top_layout.addWidget(title_label, alignment=Qt.AlignCenter)
        top_layout.addStretch()
        top_layout.addSpacing(80)
        main_layout.addLayout(top_layout)

        # -----------------------------
        # Center: buttons (3 now)
        # -----------------------------
        button_layout = QHBoxLayout()
        button_layout.setSpacing(30)

        # Webcam button
        btn_webcam = QPushButton("Webcam Only")
        btn_webcam.setFont(QFont("Arial", 16))
        btn_webcam.setCursor(Qt.PointingHandCursor)
        btn_webcam.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 15px;
                padding: 20px 30px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        btn_webcam.clicked.connect(self.run_webcam)
        button_layout.addWidget(btn_webcam)

        # Camera + Sensor button
        btn_split = QPushButton("Camera + Sensor")
        btn_split.setFont(QFont("Arial", 16))
        btn_split.setCursor(Qt.PointingHandCursor)
        btn_split.setStyleSheet("""
            QPushButton {
                background-color: #FF5722;
                color: white;
                border-radius: 15px;
                padding: 20px 30px;
            }
            QPushButton:hover {
                background-color: #E64A19;
            }
        """)
        btn_split.clicked.connect(self.run_split)
        button_layout.addWidget(btn_split)

        # Image Detection button
        btn_image = QPushButton("Image Detection")
        btn_image.setFont(QFont("Arial", 16))
        btn_image.setCursor(Qt.PointingHandCursor)
        btn_image.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 15px;
                padding: 20px 30px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        btn_image.clicked.connect(self.run_image_detection)
        button_layout.addWidget(btn_image)

        main_layout.addStretch()
        main_layout.addLayout(button_layout)
        main_layout.addStretch()

        # -----------------------------
        # Separator line
        # -----------------------------
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("color: #555555;")
        main_layout.addWidget(line)

        # -----------------------------
        # Bottom: developers and project info
        # -----------------------------
        info_layout = QVBoxLayout()
        info_layout.setSpacing(10)
        dev_heading = QLabel("Developers")
        dev_heading.setFont(QFont("Arial", 16, QFont.Bold))
        dev_heading.setStyleSheet("color: #32CD32;")
        dev_heading.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(dev_heading)

        dev_text = QLabel("FruitVision Group 43")
        dev_text.setFont(QFont("Arial", 14))
        dev_text.setStyleSheet("color: #CCCCCC;")
        dev_text.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(dev_text)

        proj_heading = QLabel("About Project")
        proj_heading.setFont(QFont("Arial", 16, QFont.Bold))
        proj_heading.setStyleSheet("color: #32CD32;")
        proj_heading.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(proj_heading)

        proj_text = QLabel(
            "This project integrates YOLOv8-based 2D fruit detection with Intel RealSense depth "
            "sensing to create a real-time 3D detection system. It allows simultaneous detection "
            "of multiple fruit types using computer vision while measuring precise distances using "
            "the depth sensor. The module supports webcam mode, RGB+Depth mode, and image detection mode "
            "providing an interactive tool for research, robotics, and smart agricultural applications."
        )
        proj_text.setFont(QFont("Arial", 12))
        proj_text.setStyleSheet("color: #CCCCCC;")
        proj_text.setAlignment(Qt.AlignCenter)
        proj_text.setWordWrap(True)
        info_layout.addWidget(proj_text)

        main_layout.addLayout(info_layout)
        self.setLayout(main_layout)

    # -----------------------------
    # Button functions
    # -----------------------------
    def run_webcam(self):
        subprocess.Popen([sys.executable, WEBCAM_SCRIPT])

    def run_split(self):
        subprocess.Popen([sys.executable, SPLIT_SCRIPT])

    def run_image_detection(self):
        # Open file dialog to select image for detection
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Image for Detection", "", "Image Files (*.png *.jpg *.jpeg)", options=options
        )
        if filename:
            subprocess.Popen([sys.executable, IMAGE_DETECTION_SCRIPT, filename])

# -----------------------------
# Run GUI
# -----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FruitVisionGUI()
    window.show()
    sys.exit(app.exec_())
