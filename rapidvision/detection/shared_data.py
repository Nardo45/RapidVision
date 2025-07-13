import threading

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
    # Current camera profile
    current_camera_profile = None

# Define a class to hold shared variables between threads
class shared_variables:
    # Static variables shared across all class instances
    latest_detections = None
    latest_frame = None
    avg_cam_focal_length = {}
    cam_index = 0
    current_cam_profile = None
    reset_camera = False
    stop_thread = threading.Event()

    # Class-level lock for thread-safety
    lock = threading.Lock()

    @classmethod
    def set_latest_detections(cls, detections):
        with cls.lock: # Ensure thread-safety when modifying class variables
            cls.latest_detections = detections
    
    @classmethod
    def set_latest_frame(cls, frame):
        with cls.lock:
            cls.latest_frame = frame

    @classmethod
    def set_cam_focal_len(cls, focal_len):
        with cls.lock:
            cls.avg_cam_focal_length = focal_len
    
    @classmethod
    def set_cam_idx(cls, idx):
        with cls.lock:
            cls.cam_index = idx

    @classmethod
    def set_current_cam_profile(cls, profile):
        with cls.lock:
            cls.current_cam_profile = profile

    @classmethod
    def set_reset_camera(cls, reset):
        with cls.lock:
            cls.reset_camera = reset
    
    @classmethod
    def get_latest_detections(cls):
        with cls.lock:
            return cls.latest_detections
    
    @classmethod
    def get_latest_frame(cls):
        with cls.lock:
            return cls.latest_frame
    
    @classmethod
    def get_cam_focal_len(cls):
        with cls.lock:
            return cls.avg_cam_focal_length
    
    @classmethod
    def get_cam_idx(cls):
        with cls.lock:
            return cls.cam_index
        
    @classmethod
    def get_current_cam_profile(cls):
        with cls.lock:
            return cls.current_cam_profile
        
    @classmethod
    def get_reset_camera(cls):
        with cls.lock:
            return cls.reset_camera