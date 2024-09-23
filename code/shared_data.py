# Import the Event class from the threading module
from threading import Event

# Define a class to hold various settings for the application
class Settings:
    # Flag to indicate whether to show the amount of detections
    show_amount_of_detections = True
    # Flag to indicate whether to show the amount of detections per class
    show_amount_of_detections_per_class = True
    # Frames per second (FPS) controller setting
    fps_controller = 60
    # Flag to indicate whether to calibrate the camera
    calibrate_camera = False
    # Flag to indicate whether to estimate distances
    distance_estimation = True

# Define a class to hold shared variables between threads
class shared_variables:
    # Holds the latest detections
    latest_detections = None
    # Holds the latest frame
    latest_frame = None
    # Holds the previous frame
    prev_frame = None
    # Holds the average camera focal length for each camera index
    avg_cam_focal_length = {}
    # Holds the current camera index
    cam_index = 0
    # Event to stop the thread
    stop_thread = Event()