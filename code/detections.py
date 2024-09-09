from object_detection.utils import visualization_utils as vis_util
from threading import Lock
from tensorflow import convert_to_tensor, newaxis
from shared_data import shared_variables as sv
from os import environ
from numpy import int64
from time import sleep
from cv2 import cvtColor, COLOR_BGR2RGB

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

frame_lock = Lock()

def draw_object_bounds(frame, detections, category_index):
    vis_util.visualize_boxes_and_labels_on_image_array(
        image=frame,
        boxes=detections['detection_boxes'],
        classes=detections['detection_classes'],
        scores=detections['detection_scores'],
        category_index=category_index,
        use_normalized_coordinates=True,
        min_score_thresh=0.5,
        line_thickness=4,
        max_boxes_to_draw=50
    )

    # Count total detections with a score above 0.5
    detection_num = sum(score > 0.5 for score in detections['detection_scores'])

    # Create a dictionary to count detections per class
    detection_num_per_class = {}
    for i, class_id in enumerate(detections['detection_classes']):
        if detections['detection_scores'][i] > 0.5: # Only count detections with a score above 0.5
            class_name = category_index[class_id]['name']
            if class_name in detection_num_per_class:
                detection_num_per_class[class_name] += 1
            else:
                detection_num_per_class[class_name] = 1

    return detection_num, detection_num_per_class

def read_objects(model):
    """Read objects from the live feed and update latest detections."""
    
    while not sv.stop_thread.is_set():
        with frame_lock:
            latest_frame = sv.latest_frame
            if latest_frame is not None:
                frame = latest_frame.copy()
                frame = cvtColor(frame, COLOR_BGR2RGB) # Convert frame to RGB as TensorFlow expects RGB format

                input_tensor = convert_to_tensor(frame)
                input_tensor = input_tensor[newaxis,...]

                detections = model(input_tensor)
                
                # Get detected objects
                num_detections = int(detections.pop('num_detections'))
                detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
                detections['num_detections'] = num_detections

                # Detection_classes should be ints.
                detections['detection_classes'] = detections['detection_classes'].astype(int64)

                # Update the latest detections in thread-safe manner
                sv.latest_detections = detections

        sleep(0.08)