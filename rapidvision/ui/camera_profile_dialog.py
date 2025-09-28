from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QSpinBox, QLineEdit, QMessageBox
    )
from PyQt5.QtCore import Qt, QTimer

# Custom imports
from rapidvision.utils.general import (
    absolute_path, extract_json_2_dict, save_2_json
    )
from rapidvision.detection import shared_data

# Constants
CAMERA_PROFILES_FILE: str = absolute_path('RapidVision', 'camera_profiles.json', 'config')

# Normalize whatever is in the JSON into a list of profiles
_raw_defaults = extract_json_2_dict(CAMERA_PROFILES_FILE) or {}
_DEF = _raw_defaults.get("default_camera_profile", [])
if isinstance(_DEF, dict):
    # If it's a single profile dict
    if 'name' in _DEF:
        DEFUALT_PROFILES = [_DEF]
    else:
        # if mapping name->profile, convert to list of values
        DEFUALT_PROFILES = list(_DEF.values())
elif _DEF is None:
    DEFUALT_PROFILES = []
else:
    DEFUALT_PROFILES = list(_DEF)


class CameraProfileDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Profile Manager")
        self.setGeometry(150, 150, 500, 300)

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

        # Load and display
        self.load_profiles()
        self.refresh_profile_list()

    def load_profiles(self):
        """
        Load custom camera profiles from the JSON file and ensureit's a list.
        """
        data = extract_json_2_dict(CAMERA_PROFILES_FILE) or {}
        custom = data.get("custom_camera_profiles", [])
        if custom is None:
            custom = []
        # ensure list
        if isinstance(custom, dict):
            # if it's a single dict profile, wrap it
            if 'name' in custom:
                self.custom_profiles = [custom]
            else:
                self.custom_profiles = list(custom.values())
        else:
            self.custom_profiles = list(custom)

    def save_profiles(self):
        """Save the current custom profiles to the JSON file."""
        # ensure serializable list
        try:
            save_2_json(CAMERA_PROFILES_FILE, {"custom_camera_profiles": self.custom_profiles})
        except Exception as e:
            self._show_message("Save Error", f"Failed to save profiles: {e}", kind='warning')

    def refresh_profile_list(self):
        """Refresh the profile list widget with current profiles."""
        self.profile_list.clear()
        # defaults
        for profile in DEFUALT_PROFILES or []:
            name = profile.get('name') if isinstance(profile, dict) else str(profile)
            item = QListWidgetItem(f"[Default] {name}")
            item.setData(1000, profile)  # Custom data for the profile
            self.profile_list.addItem(item)
        # customs
        for profile in self.custom_profiles or []:
            name = profile.get('name') if isinstance(profile, dict) else str(profile)
            item = QListWidgetItem(name)
            item.setData(1000, profile)
            self.profile_list.addItem(item)

    def on_profile_selected(self, item: QListWidgetItem):
        """Handle profile selection from the list."""
        profile = item.data(1000)
        if not isinstance(profile, dict):
            # defensive: ignore non-dict entries
            self._show_message("Invalid profile", "Selected profile has invalid data.", kind='warning')
            self.current_selected_profile = None
            return
        self.current_selected_profile = profile
        self.name_input.setText(profile.get('name', ''))
        self.width_input.setValue(int(profile.get('width', 640) or 640))
        self.height_input.setValue(int(profile.get('height', 480) or 480))
        self.fps_input.setValue(int(profile.get('fps', 30) or 30))

    def apply_profile(self):
        """Apply the selected profile to the camera."""
        if not self.current_selected_profile:
            self._show_message("No Profile", "Select a profile to apply.", kind='warning')
            return

        # Update shared data settings (ensure methods exist)
        try:
            shared_data.shared_variables.set_current_cam_profile(self.current_selected_profile)
            shared_data.shared_variables.set_reset_camera(True)
        except Exception as e:
            self._show_message("Apply Error", f"Failed to apply profile: {e}", kind='warning')
            return

        # Inform the user non-modally, so we avoid deadlocks with other threads
        self._show_message("Profile Applied", f"Applied profile: {self.current_selected_profile.get('name', '')}", kind='info')
        self.close()

    def save_custom_profile(self):
        """Save a new custom camera profile."""
        name = self.name_input.text().strip()
        if not name:
            self._show_message("Invalid Name", "Profile name cannot be empty.", kind='warning')
            return
        # check duplicates
        combined = []
        combined.extend([p for p in (DEFUALT_PROFILES or []) if isinstance(p, dict)])
        combined.extend([p for p in (self.custom_profiles or []) if isinstance(p, dict)])
        for profile in combined:
            if profile.get('name') == name:
                self._show_message("Profile Exists", "A profile with this name already exists.", kind='warning')
                return

        new_profile = {
            'name': name,
            'width': int(self.width_input.value()),
            'height': int(self.height_input.value()),
            'fps': int(self.fps_input.value())
        }
        self.custom_profiles.append(new_profile)
        self.save_profiles()
        self.refresh_profile_list()
        self._show_message("Profile Saved", f"Profile: {name}", kind='info')

    def delete_custom_profile(self):
        """Delete the currently selected custom profile."""
        if not self.current_selected_profile:
            self._show_message("No Profile", "Select a profile to delete.", kind='warning')
            return
        profile = self.current_selected_profile
        if isinstance(profile, dict) and profile in DEFUALT_PROFILES:
            self._show_message("Cannot Delete", "Cannot delete default profiles.", kind='warning')
            return
        # only remove from custom list
        self.custom_profiles = [p for p in (self.custom_profiles or []) if p != profile]
        self.save_profiles()
        self.refresh_profile_list()
        self._show_message("Profile Deleted", f"Deleted profile: {profile.get('name', 'Unknown')}", kind='info')
        self.current_selected_profile = None

    def _show_message(self, title: str, text: str, kind: str = 'info', timeout_ms: int = 3000):
        """
        Non-blocking helper to show a short message to the user.
        kind: 'info' or 'warning'
        timeout_ms: auto close in milliseconds
        """
        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(text)
        if kind == 'warning':
            msg.setIcon(QMessageBox.Warning)
        else:
            msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok)
        # make it non-modal to avoid blocking the main event loop if other threads are interacting with shared state
        msg.setWindowModality(Qt.NonModal)
        msg.show()
        # auto-close after timeout_ms to ensure no lingering modal
        QTimer.singleShot(timeout_ms, msg.accept)
