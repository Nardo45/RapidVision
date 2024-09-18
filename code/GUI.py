import sys, cv2, detections, torch

import shared_data as sd
import useful_funcs as uf

from threading import Thread
from PyQt5.QtGui import QImage, QPainter
from PyQt5.QtCore import QTimer, QPoint, Qt
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog

# global variables
allow_right = True
allow_left = True
last_direction = None

class VideoWidget(QWidget):
    """A class for displaying video from OpenCV and detection boxes"""

    def __init__(self):
        QWidget.__init__(self)
        self.camera_index = 0
        self.live_feed = cv2.VideoCapture(self.camera_index)

        # Set the camera to its maximum resolution
        self.live_feed.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
        self.live_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.update_fps(60)

    def convert_cv2qimage(self, cv2_img):
        """Convert OpenCV image to QImage"""
        if sd.shared_variables.latest_detections is not None:
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

            if sd.Settings.show_amount_of_detections and self.num_of_detections:
                painter.setPen(Qt.white)
                text = f"Detections: {self.num_of_detections}"
                text_rect = painter.fontMetrics().boundingRect(text)
                x_offset += 5
                y_offset += text_rect.height()
                painter.drawText(QPoint(x_offset, y_offset), text)

            if sd.Settings.show_amount_of_detections_per_class and self.num_of_detections_per_class:
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
        sd.shared_variables.latest_frame = frame
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
        self.live_feed = cv2.VideoCapture(self.camera_index)

        self.live_feed.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
        self.live_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)
    
    def set_settings_from_usr(self):
        new_fps, ok = QInputDialog.getInt(self, 'Set FPS', 'Enter new FPS:', value=int(round(1000 / self.new_interval, 0)), min=1, max=120)
        if ok:
            self.update_fps(new_fps)

    def keyPressEvent(self, event):
        global allow_left, allow_right, last_direction

        if event.key() == Qt.Key_K:
            self.set_settings_from_usr()
        elif event.key() == Qt.Key.Key_Left and allow_left:
            self.camera_index -= 1
            allow_right = True
            last_direction = 0
            self.reset_camera()
        elif event.key() == Qt.Key.Key_Right and allow_right:
            self.camera_index += 1
            allow_left = True
            last_direction = 1
            self.reset_camera()
        else:
            super().keyPressEvent(event) # Call the base class method for other key presses

if __name__ == "__main__":
    print('Starting RapidVision...')
    # Load the model and set it to use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(uf.absolute_path('RapidVision', 'yolo_nas_l.pt', True, 'model'), map_location=device)

    # Start the detection thread
    detection_thread = Thread(target=detections.read_objects, args=(model,device,), daemon=True)
    detection_thread.start()

    # Create and show the main window
    app = QApplication(sys.argv)

    widget = VideoWidget()
    widget.setWindowTitle('RapidVision')
    widget.show()

    # Ensures the thread is stopped when the main window is closed
    app.aboutToQuit.connect(lambda: sd.shared_variables.stop_thread.set())

    sys.exit(app.exec_())