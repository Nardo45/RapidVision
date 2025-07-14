from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QSpinBox, QLineEdit, QMessageBox
)

# Custom imports
from rapidvision.utils.general import absolute_path, extract_json_2_dict, save_2_json
from rapidvision.detection import shared_data

# Constants
CAMERA_PROFILES_FILE: str = absolute_path('RapidVision', 'camera_profiles.json', 'config')
DEFUALT_PROFILES: dict = extract_json_2_dict(CAMERA_PROFILES_FILE).get("default_camera_profile")

class CameraProfileDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Profile Manager")
        self.setGeometry(150, 150, 500, 300)

        self.load_profiles()
        self.current_selected_profile = None

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Left: Profile List
        self.profile_list = QListWidget()
        self.profile_list.itemClicked.connect(self.on_profile_selected)
        main_layout.addWidget(self.profile_list)

        # Right: Form and Buttons
        form_layout = QVBoxLayout()

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Profile Name")
        form_layout.addWidget(QLabel("Name"))
        form_layout.addWidget(self.name_input)

        self.width_input = QSpinBox()
        self.width_input.setMaximum(9999)
        form_layout.addWidget(QLabel("Width"))
        form_layout.addWidget(self.width_input)

        self.height_input = QSpinBox()
        self.height_input.setMaximum(9999)
        form_layout.addWidget(QLabel("Height"))
        form_layout.addWidget(self.height_input)

        self.fps_input = QSpinBox()
        self.fps_input.setRange(1, 120)
        form_layout.addWidget(QLabel("FPS"))
        form_layout.addWidget(self.fps_input)

        self.apply_button = QPushButton("Apply Profile")
        self.apply_button.clicked.connect(self.apply_profile)
        form_layout.addWidget(self.apply_button)

        self.save_button = QPushButton("Save Custom Profile")
        self.save_button.clicked.connect(self.save_custom_profile)
        form_layout.addWidget(self.save_button)

        self.delete_button = QPushButton("Delete Custom Profile")
        self.delete_button.clicked.connect(self.delete_custom_profile)
        form_layout.addWidget(self.delete_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        form_layout.addWidget(self.close_button)

        main_layout.addLayout(form_layout)

        self.refresh_profile_list()

    def load_profiles(self):
        """Load custom camera profiles from the JSON file."""
        self.custom_profiles = extract_json_2_dict(CAMERA_PROFILES_FILE).get("custom_camera_profiles", [])

    def save_profiles(self):
        """Save the current custom profiles to the JSON file."""
        save_2_json(CAMERA_PROFILES_FILE, {"custom_camera_profiles": self.custom_profiles})

    def refresh_profile_list(self):
        """Refresh the profile list widget with current profiles."""
        self.profile_list.clear()
        for profile in DEFUALT_PROFILES:
            item = QListWidgetItem(f"[Default] {profile.get('name', 'Default')}")
            item.setData(1000, profile)  # Custom data for the profile
            self.profile_list.addItem(item)
        for profile in self.custom_profiles:
            item = QListWidgetItem(profile.get('name', 'Custom Profile'))
            item.setData(1000, profile)
            self.profile_list.addItem(item)

    def on_profile_selected(self, item: QListWidgetItem):
        """Handle profile selection from the list."""
        profile = item.data(1000)
        self.current_selected_profile = profile
        self.name_input.setText(profile.get('name', ''))
        self.width_input.setValue(profile.get('width', 640))
        self.height_input.setValue(profile.get('height', 480))
        self.fps_input.setValue(profile.get('fps', 30))

    def apply_profile(self):
        """Apply the selected profile to the camera."""
        if not self.current_selected_profile:
            QMessageBox.warning(self, "No Profile", "Select a profile to apply.")
            return
        
        # Update shared data settings
        shared_data.shared_variables.set_current_cam_profile(self.current_selected_profile)

        # Reset camera with the new profile
        shared_data.shared_variables.set_reset_camera(True)
        QMessageBox.information(self, "Profile Applied", f"Applied profile: {self.current_selected_profile}")
        self.close()

    def save_custom_profile(self):
        """Save a new custom camera profile."""
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Invalid Name", "Profile name cannot be empty.")
            return
        for profile in DEFUALT_PROFILES + self.custom_profiles:
            if profile.get('name') == name:
                QMessageBox.warning(self, "Profile Exists", "A profile with this name already exists.")
                return
        
        new_profile = {
            'name': name,
            'width': self.width_input.value(),
            'height': self.height_input.value(),
            'fps': self.fps_input.value()
        }
        self.custom_profiles.append(new_profile)
        self.save_profiles()
        self.refresh_profile_list()
        QMessageBox.information(self, "Profile Saved", f"Profile: {name}")

    def delete_custom_profile(self):
        """Delete the currently selected custom profile."""
        if not self.current_selected_profile:
            QMessageBox.warning(self, "No Profile", "Select a profile to delete.")
            return
        profile = self.current_selected_profile
        if profile in DEFUALT_PROFILES:
            QMessageBox.warning(self, "Cannot Delete", "Cannot delete default profiles.")
            return
        self.custom_profiles = [p for p in self.custom_profiles if p != profile]
        self.save_profiles()
        self.refresh_profile_list()
        QMessageBox.information(self, "Profile Deleted", f"Deleted profile: {profile.get('name', 'Unknown')}")
        self.current_selected_profile = None