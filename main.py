# Import necessary libraries
import sys
import cv2
import torch

from rapidvision.detection import ppyoloe_wrapper as detections

# Import custom modules
from rapidvision.utils.general import absolute_path, extract_json_2_dict
from rapidvision.detection import shared_data
from rapidvision.camera import vision_io
from rapidvision.ui.settings import SettingsMenu

# Import required classes from PyQt5
from threading import Thread
from PyQt5.QtGui import QImage, QPainter, QColor
from PyQt5.QtCore import QTimer, QPoint, Qt, QRect
from PyQt5.QtWidgets import QApplication, QWidget

# Fixes error when third party libs import ppyoloe
import third_party.ppyoloe
sys.modules['ppyoloe'] = third_party.ppyoloe

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
            if shared_data.Settings.measure_inf_time:
                self.draw_inf_time(painter)
            if allow_left is not True:
                allow_left = True
            if allow_right is not True:
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

    def draw_inf_time(self, painter: QPainter):
        """
        Draw the rolling average inference time in the top-right corner using draw_text.
        Expects shared_data.shared_variables.get_avg_inference_time() to return seconds (float) or None.
        """
        inf = shared_data.shared_variables.get_avg_inference_time()
        if inf is None:
            return

        # Format text
        ms = inf * 1000.0
        fps_text = ""
        try:
            if inf > 0:
                fps_text = f" ({1.0/inf:.1f} FPS)"
        except Exception:
            pass

        text = f"Inference: {ms:.1f} ms{fps_text}"

        # Metrics and box size (use painter font metrics)
        fm = painter.fontMetrics()
        padding_x = 8
        text_w = fm.boundingRect(text).width()

        box_w = text_w + padding_x * 2

        # compute left edge of the scaled image area then subtract box width + margin
        scaled_image = self.image.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (self.width() + scaled_image.width()) // 2 - box_w - 5
        y = 10  # 10px margin from top

        # Use draw_text to render rectangle+text
        # draw_text returns the next y position; we ignore it here.
        self.draw_text(painter, text, x, y)

    def draw_text(self, painter: QPainter, text, x_offset, y_offset):
        """
        Draw a single line of text with a semi-transparent black background box.
        Returns the new y_offset (start position for the next line).
        """
        fm = painter.fontMetrics()
        padding_x = 8
        padding_y = 6

        text_w = fm.boundingRect(text).width()
        text_h = fm.height()

        box_w = text_w + padding_x * 2
        box_h = text_h + padding_y * 2

        painter.save()
        # background
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 160))  # semi-transparent black
        painter.drawRect(QRect(x_offset, y_offset, box_w, box_h))

        # text
        painter.setPen(Qt.white)
        text_x = x_offset + padding_x
        painter.drawText(QPoint(text_x, y_offset + padding_y + fm.ascent()), text)
        painter.restore()

        return y_offset + box_h

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
        # If already open, raise it
        if hasattr(self, 'settings_menu') and getattr(self, 'settings_menu') is not None:
            if self.settings_menu.isVisible():
                self.settings_menu.raise_()
                self.settings_menu.activateWindow()
                return

        # Create settings dialog with parent so stacking/focus is correct
        self.settings_menu = SettingsMenu(parent=self)
        # Show modelessly (no nested modal loop)
        self.settings_menu.setWindowModality(Qt.WindowModal)
        self.settings_menu.show()
        self.settings_menu.raise_()
        self.settings_menu.activateWindow()

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
            # Call the base class method for other key presses
            super().keyPressEvent(event)


# Main entry point of the application
if __name__ == "__main__":
    print('Starting RapidVision...')

    if hasattr(torch.backends, "nnpack"):
        try:
            # Try creating a dummy tensor to check if NNPACK works
            x = torch.randn(1, 3, 64, 64)
            torch.backends.nnpack.convolution(x, x, padding=1)
        except Exception:
            # Disable only if unsupported
            torch.backends.nnpack.set_flags(False)

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
    # widget.setWindowIcon(QIcon(absolute_path('RapidVision', 'RapidVision_Icon.png', 'Icon'))) TODO: Create an icon and use this line to display it
    widget.show()

    # Ensure the detection thread is stopped when the main window is closed
    app.aboutToQuit.connect(lambda: shared_data.shared_variables.stop_thread.set())

    # Execute the application
    sys.exit(app.exec_())
