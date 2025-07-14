from rapidvision.detection import shared_data
from rapidvision.ui.camera_profile_dialog import CameraProfileDialog

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QLabel, QSpinBox

# Constants
SETTINGS_WINDOW_WIDTH = 300
SETTINGS_WINDOW_HEIGHT = 200
MAX_FPS = 120
MIN_FPS = 1


class SettingsMenu(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Settings")
        self.setGeometry(100, 100, SETTINGS_WINDOW_WIDTH, SETTINGS_WINDOW_HEIGHT)

        layout = QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(QLabel('Settings Menu'))

        self.create_toggle_buttons(layout, "Show Detect Nums", "show_amount_of_detections")
        self.create_toggle_buttons(layout, 'Show Detect Nums per Class', 'show_amount_of_detections_per_class')

        self.cam_calibration_button = QPushButton('Calibrate Camera')
        self.cam_calibration_button.clicked.connect(self.calibrate_camera)
        layout.addWidget(self.cam_calibration_button)

        self.camera_profile_button = QPushButton('Manage Camera Profiles')
        self.camera_profile_button.clicked.connect(self.open_camera_profiles)
        layout.addWidget(self.camera_profile_button)

        self.fps_controller_label = QLabel(f'FPS: {shared_data.Settings.fps_controller}')
        layout.addWidget(self.fps_controller_label)

        self.fps_controller_spinbox = QSpinBox()
        self.fps_controller_spinbox.setMinimum(MIN_FPS)
        self.fps_controller_spinbox.setMaximum(MAX_FPS)
        self.fps_controller_spinbox.setValue(shared_data.Settings.fps_controller)
        self.fps_controller_spinbox.valueChanged.connect(self.update_fps)
        layout.addWidget(self.fps_controller_spinbox)

        self.close_button = QPushButton('Close Settings')
        self.close_button.clicked.connect(self.close)
        layout.addWidget(self.close_button)

    def create_toggle_buttons(self, layout, text, setting_name):
        button = QPushButton(f"{text}: {'ON' if getattr(shared_data.Settings, setting_name) else 'OFF'}")
        button.setCheckable(True)
        button.clicked.connect(lambda: self.toggle_setting(button, text, setting_name))
        layout.addWidget(button)

    def toggle_setting(self, button, text, setting_name):
        setattr(shared_data.Settings, setting_name, not getattr(shared_data.Settings, setting_name))
        button.setText(f"{text}: {'ON' if getattr(shared_data.Settings, setting_name) else 'OFF'}")

    def calibrate_camera(self):
        shared_data.Settings.calibrate_camera = not shared_data.Settings.calibrate_camera
        self.close()

    def open_camera_profiles(self):
        """Open the camera profile management dialog."""
        self.close()
        dialog = CameraProfileDialog()
        dialog.exec_()
        self.open()  # Reopen settings after closing the dialog

    def update_fps(self, new_fps):
        self.fps_controller_label.setText(f'FPS: {new_fps}')
        shared_data.Settings.fps_controller = new_fps