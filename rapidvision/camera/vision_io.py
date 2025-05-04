import cv2

from rapidvision.detection.shared_data import shared_variables as sv

def camera_capture():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    # Set the camera to its maximum resolution
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
    cap.set(cv2.CAP_PROP_FPS, 30)
    while not sv.stop_thread.is_set():
        ret, frame = cap.read()
        if not ret:
            sv.set_latest_frame(None)
        else:
            sv.set_latest_frame(frame)