from collections import Counter
import supervision as svision
from threading import Lock
from shared_data import shared_variables as sv
from os import environ
from time import sleep

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

frame_lock = Lock()

bounding_box_annotator = svision.BoxAnnotator()  # Updated to use BoxAnnotator
label_annotator = svision.LabelAnnotator()

def draw_object_bounds(frame, detections):
    """Draw bounding boxes around detected objects and get number of detected objects."""

    #detections = sv.latest_detections

    annotated_image = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)  # Fixed label annotator

    # Count total detections with a score above 0.5
    detection_num = len(detections.data['class_name'])

    # Create a dictionary to count detections per class
    class_names = detections.data['class_name']
    class_counts = Counter(class_names)

    # Create the formatted string
    formatted_output = ', '.join([f"{str(class_name).capitalize()} {count}" for class_name, count in class_counts.items()])

    class_counts = formatted_output.split(', ')

    return detection_num, class_counts

def read_objects(model):
    """Read objects from the live feed and update latest detections."""
    
    while not sv.stop_thread.is_set():
        with frame_lock:
            latest_frame = sv.latest_frame
            if latest_frame is not None:
                frame = latest_frame.copy()
                #frame = cvtColor(frame, COLOR_BGR2RGB) # Convert frame to RGB as TensorFlow expects RGB format

                results = model(frame)[0] # Perform object detection on the frame
                detections = svision.Detections.from_ultralytics(results) # Convert Ultralytics results to supervision Detections object

                # Update the latest detections in thread-safe manner
                sv.latest_detections = detections