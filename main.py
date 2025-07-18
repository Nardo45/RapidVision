# Import necessary libraries
import sys, cv2, torch

import third_party.ppyoloe
sys.modules['ppyoloe'] = third_party.ppyoloe

from rapidvision.detection import ppyoloe_wrapper as detections

# Import custom modules
from rapidvision.utils.general import absolute_path, extract_json_2_dict
from rapidvision.detection import shared_data
from rapidvision.camera import vision_io
from rapidvision.ui.settings import SettingsMenu

# Import required classes from PyQt5
from threading import Thread
from PyQt5.QtGui import QImage, QPainter
from PyQt5.QtCore import QTimer, QPoint, Qt
from PyQt5.QtWidgets import QApplication, QWidget

# Define global variables
allow_right = True
allow_left = True
last_direction = 2
old_fps = shared_data.Settings.fps_controller

# Constants
FONT_SIZE_DIVISOR = 50


class VideoWidget(QWidget):
    """A class for displaying video from OpenCV and detection boxes"""
    def __init__(self):
        QWidget.__init__(self)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.update_fps()

        self.image = None
        self.num_of_detections: int = 0
        self.num_of_detections_per_class: int = 0
        self.y_point = 0
        self.x_point = 0

    def convert_cv2qimage(self, cv_img):
        """Convert OpenCV image to QImage"""
        if shared_data.shared_variables.get_latest_detections() is not None and not shared_data.Settings.calibrate_camera:
            self.num_of_detections, self.num_of_detections_per_class, new_frame = detections.display_detections(cv_img)
        else:
            self.num_of_detections, self.num_of_detections_per_class, new_frame = 0, 0, cv_img

        rgb_image = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
        height, width, ch = rgb_image.shape
        return QImage(rgb_image.data, width, height, ch * width, QImage.Format_RGB888)
    
    def paintEvent(self, event):
        global last_direction, allow_left, allow_right
        painter = QPainter(self)

        font_size = min(self.width(), self.height()) // FONT_SIZE_DIVISOR
        font = painter.font()
        font.setPointSize(font_size)
        painter.setFont(font)

        if self.image is not None:
            self.draw_image(painter)
            self.draw_detections(painter)
            if allow_left != True:
                allow_left = True
            if allow_right != True:
                allow_right = True
        else:
            self.draw_no_feed(painter)

    def draw_image(self, painter: QPainter):
        scaled_image = self.image.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x_offset = (self.width() - scaled_image.width()) // 2
        y_offset = (self.height() - scaled_image.height()) // 2
        self.y_point = y_offset
        self.x_point = x_offset
        painter.drawImage(QPoint(x_offset, y_offset), scaled_image)

    def draw_detections(self, painter: QPainter):
        x_offset = self.x_point + 5
        y_offset = self.y_point + 5
        painter.setPen(Qt.white)

        if shared_data.Settings.show_amount_of_detections:
            text = f"Detections: {self.num_of_detections}"
            y_offset = self.draw_text(painter, text, x_offset, y_offset)

        if shared_data.Settings.show_amount_of_detections_per_class and self.num_of_detections_per_class:
            for text in self.num_of_detections_per_class:
                y_offset = self.draw_text(painter, text, x_offset, y_offset)

    def draw_text(self, painter: QPainter, text, x_offset, y_offset):
        text_rect = painter.fontMetrics().boundingRect(text)
        painter.drawText(QPoint(x_offset, y_offset + text_rect.height()), text)
        return y_offset + text_rect.height()
    
    def draw_no_feed(self, painter: QPainter):
        global last_direction, allow_left, allow_right
        painter.fillRect(self.rect(), Qt.black)
        painter.setPen(Qt.red)
        text = "No camera feed detected."
        text_rect = painter.fontMetrics().boundingRect(text)
        x_offset = (self.width() - text_rect.width()) // 2
        y_offset = (self.height() - text_rect.height()) // 2
        painter.drawText(QPoint(x_offset, y_offset), text)

        if last_direction == 0 and allow_left:
            allow_left = False
        elif last_direction == 1 and allow_right:
            allow_right = False

    def update_frame(self):
        """Update the frame displayed in the widget and fps if changed."""
        global old_fps

        new_img = shared_data.shared_variables.get_latest_frame()
        if new_img is not None:
            self.image = self.convert_cv2qimage(new_img)
        else:
            self.image = new_img
        
        if shared_data.Settings.fps_controller != old_fps:
            self.update_fps()
            old_fps = shared_data.Settings.fps_controller
        
        self.update()

    def update_fps(self):
        fps = shared_data.Settings.fps_controller
        if fps > 0:
            self.new_interval = float((1 / fps) * 1000)
            self.timer.stop()
            self.timer.start(int(self.new_interval))

    def display_settings(self):
        self.settings_menu = SettingsMenu()
        self.settings_menu.exec_()

    def keyPressEvent(self, event):
        global allow_left, allow_right, last_direction

        if event.key() == Qt.Key_K:
            self.display_settings()

        elif event.key() == Qt.Key_Left and allow_left:
            shared_data.shared_variables.set_cam_idx(shared_data.shared_variables.get_cam_idx() - 1)
            allow_right = True
            last_direction = 0
            shared_data.shared_variables.set_reset_camera(True)

        elif event.key() == Qt.Key_Right and allow_right:
            shared_data.shared_variables.set_cam_idx(shared_data.shared_variables.get_cam_idx() + 1)
            allow_left = True
            last_direction = 1
            shared_data.shared_variables.set_reset_camera(True)

        else:
            super().keyPressEvent(event)  # Call the base class method for other key presses


# Main entry point of the application
if __name__ == "__main__":
    print('Starting RapidVision...')
    # Load average camera focal length data from JSON file
    shared_data.shared_variables.set_cam_focal_len(extract_json_2_dict(absolute_path('RapidVision', 'cam_cali_data.json', 'data')))

    # Set Device to CPU
    device = torch.device('cpu')

    # Load model to CPU
    model = torch.load(absolute_path('RapidVision', 'ppyoloe_crn_l_300e_coco.pt', 'model'), map_location='cpu', weights_only=False)
    model.eval()

    # Start seperate threads as daemon processes
    detection_thread = Thread(target=detections.read_objects, args=(model, device,), daemon=True)
    detection_thread.start()

    camera_thread = Thread(target=vision_io.camera_capture, daemon=True)
    camera_thread.start()

    # Create and show the main window
    app = QApplication(sys.argv)
    widget = VideoWidget()
    widget.setWindowTitle('RapidVision')
    #widget.setWindowIcon(QIcon(absolute_path('RapidVision', 'RapidVision_Icon.png', 'Icon'))) TODO: Create an icon and use this line to display it
    widget.show()

    # Ensure the detection thread is stopped when the main window is closed
    app.aboutToQuit.connect(lambda: shared_data.shared_variables.stop_thread.set())

    # Execute the application
    sys.exit(app.exec_())