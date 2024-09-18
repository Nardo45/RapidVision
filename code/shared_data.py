from threading import Event 

class Settings:
    show_amount_of_detections = True
    show_amount_of_detections_per_class = True

class shared_variables:
    latest_detections = None
    latest_frame = None
    prev_conf_score = 0 # Used for a smooth transition between neon-like colors
    stop_thread = Event()