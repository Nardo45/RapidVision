# Import necessary libraries
import sys, cv2, detections, torch

# Import custom modules
import useful_funcs as uf
from shared_data import shared_variables as sv, Settings

# Import required classes from PyQt5
from threading import Thread
from PyQt5.QtGui import QImage, QPainter
from PyQt5.QtCore import QTimer, QPoint, Qt
from PyQt5.QtWidgets import QApplication, QPushButton, QSpinBox, QWidget, QDialog, QVBoxLayout, QLabel

# Define global variables
allow_right = True
allow_left = True
last_direction = None


class SettingsMenu(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Settings")
        self.setGeometry(100, 100, 300, 200)

        layout = QVBoxLayout()

        label = QLabel("Settings Menu")
        layout.addWidget(label)

        # Toggle display detection nums
        self.show_detect_nums_button = QPushButton(f"Show Detect Nums: {'ON' if Settings.show_amount_of_detections else 'OFF'}")
        self.show_detect_nums_button.setCheckable(True)
        self.show_detect_nums_button.clicked.connect(self.toggle_detect_nums)
        layout.addWidget(self.show_detect_nums_button)

        # Toggle display detection nums per class
        self.show_detect_nums_per_class_button = QPushButton(f"Show Detect Nums per Class: {'ON' if Settings.show_amount_of_detections_per_class else 'OFF'}")
        self.show_detect_nums_per_class_button.setCheckable(True)
        self.show_detect_nums_per_class_button.clicked.connect(self.toggle_detect_nums_per_class)
        layout.addWidget(self.show_detect_nums_per_class_button)

        # Toggle for cam calibration
        self.cam_calibration_button = QPushButton("Calibrate Camera")
        self.cam_calibration_button.clicked.connect(self.calibrate_camera)
        layout.addWidget(self.cam_calibration_button)

        # Control for the FPS
        self.fps_controller_label = QLabel("FPS: {}".format(Settings.fps_controller))
        layout.addWidget(self.fps_controller_label)

        # FPS controller spinbox
        self.fps_controller_spinbox = QSpinBox()
        self.fps_controller_spinbox.setRange(1, 120)
        self.fps_controller_spinbox.setValue(Settings.fps_controller)
        self.fps_controller_spinbox.valueChanged.connect(self.update_fps)
        layout.addWidget(self.fps_controller_spinbox)

        # Add a button to close the dialog
        self.close_button = QPushButton("Close Settings")
        self.close_button.clicked.connect(self.close)
        layout.addWidget(self.close_button)

        self.setLayout(layout)

    def toggle_detect_nums(self):
        """Toggle the 'Show Detect Nums' button state"""
        Settings.show_amount_of_detections = not Settings.show_amount_of_detections
        self.show_detect_nums_button.setText(f"Show Detect Nums: {'ON' if Settings.show_amount_of_detections else 'OFF'}")

    def toggle_detect_nums_per_class(self):
        """Toggle the 'Show Detect Nums per Class' button state"""
        Settings.show_amount_of_detections_per_class = not Settings.show_amount_of_detections_per_class
        self.show_detect_nums_per_class_button.setText(f"Show Detect Nums per Class: {'ON' if Settings.show_amount_of_detections_per_class else 'OFF'}")

    def calibrate_camera(self):
        '''Runs the calibration'''
        Settings.calibrate_camera = True
        self.close()

    def update_fps(self, value):
        """Updates the display FPS"""
        self.fps_controller_label.setText("FPS: {}".format(value))
        VideoWidget.update_fps(widget, value)



class VideoWidget(QWidget):
    """A class for displaying video from OpenCV and detection boxes"""

    def __init__(self):
        QWidget.__init__(self)
        self.live_feed = cv2.VideoCapture(sv.cam_index)

        # Set the camera to its maximum resolution
        self.live_feed.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
        self.live_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.update_fps(Settings.fps_controller)

    def convert_cv2qimage(self, cv2_img):
        """Convert OpenCV image to QImage"""
        if sv.latest_detections is not None and Settings.calibrate_camera is False:
            self.num_of_detections, self.num_of_detections_per_class, new_frame = detections.display_detections(cv2_img)
        else:
            self.num_of_detections, self.num_of_detections_per_class, new_frame = None, None, cv2_img

        # Convert the image to RGB format
        rgb_image = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
        height, width, ch = rgb_image.shape
        bytes_per_line = ch * width
        return QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    
    def paintEvent(self, event):
        global last_direction, allow_left, allow_right
        painter = QPainter(self)

        font_size = min(self.width(), self.height()) // 50

        font = painter.font()
        font.setPointSize(font_size)
        painter.setFont(font)

        if self.image:
            scaled_image = self.image.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x_offset = (self.width() - scaled_image.width()) // 2
            y_offset = (self.height() - scaled_image.height()) // 2
            painter.drawImage(QPoint(x_offset, y_offset), scaled_image)

            x_offset += 5

            if Settings.show_amount_of_detections and self.num_of_detections:
                painter.setPen(Qt.white)
                text = f"Detections: {self.num_of_detections}"
                text_rect = painter.fontMetrics().boundingRect(text)
                y_offset += text_rect.height()
                painter.drawText(QPoint(x_offset, y_offset), text)

            if Settings.show_amount_of_detections_per_class and self.num_of_detections_per_class:
                for texts in self.num_of_detections_per_class:
                    painter.setPen(Qt.white)
                    text_rect = painter.fontMetrics().boundingRect(texts)
                    y_offset += (text_rect.height())
                    painter.drawText(QPoint(x_offset, y_offset), texts)
        else:
            # Draw a black background and a text indicating no feed detected
            painter.fillRect(self.rect(), Qt.black)
            painter.setPen(Qt.red)
            text = 'No Feed Detected'
            text_rect = painter.fontMetrics().boundingRect(text)
            x_offset = (self.width() - text_rect.width()) // 2
            y_offset = (self.height() - text_rect.height()) // 2 + text_rect.height()
            painter.drawText(QPoint(x_offset, y_offset), text)

            if last_direction == 0:
                allow_left = False
            else:
                allow_right = False

    def update_frame(self):
        ret, frame = self.live_feed.read()
        sv.latest_frame = frame

        if ret:
            self.image = self.convert_cv2qimage(frame)
        else:
            self.image = None
        self.update()

    def update_fps(self, frames):
        if frames > 0:
            self.new_interval = float((1 / frames) * 1000)
            self.timer.stop()
            self.timer.start(int(self.new_interval))

    def reset_camera(self):
        self.live_feed.release()
        self.live_feed = cv2.VideoCapture(sv.cam_index)

        self.live_feed.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
        self.live_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)

    def display_settings(self):
        self.settings_menu = SettingsMenu()
        self.settings_menu.exec_()

    def keyPressEvent(self, event):
        global allow_left, allow_right, last_direction

        if event.key() == Qt.Key_K:
            self.display_settings()

        elif event.key() == Qt.Key_Left and allow_left:
            sv.cam_index -= 1
            allow_right = True
            last_direction = 0
            self.reset_camera()

        elif event.key() == Qt.Key.Key_Right and allow_right:
            sv.cam_index += 1
            allow_left = True
            last_direction = 1
            self.reset_camera()

        else:
            super().keyPressEvent(event)  # Call the base class method for other key presses



# Main entry point of the application
if __name__ == "__main__":
    print('Starting RapidVision...')

    # Load average camera focal length data from JSON file
    sv.avg_cam_focal_length = uf.extract_json_2_dict(uf.absolute_path('RapidVision', 'cam_cali_data.json', True, 'data'))

    # Load the YOLO model and set it to use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(uf.absolute_path('RapidVision', 'yolo_nas_l.pt', True, 'model'), map_location=device)

    # Start the detection thread as a daemon process
    detection_thread = Thread(target=detections.read_objects, args=(model, device,), daemon=True)
    detection_thread.start()

    # Create and show the main window
    app = QApplication(sys.argv)
    widget = VideoWidget()
    widget.setWindowTitle('RapidVision')
    widget.show()

    # Ensure the detection thread is stopped when the main window is closed
    app.aboutToQuit.connect(lambda: sv.stop_thread.set())

    # Execute the application
    sys.exit(app.exec_())