import cv2

from threading   import Lock
from collections import Counter
from torch       import from_numpy
from shared_data import shared_variables as sv, Settings
from cam_cali    import cam_cali
from utils       import extract_json_2_dict, absolute_path

# Lock to ensure thread-safe access to the frame
frame_lock = Lock()

# Dictionary to store the smoothed confidence scores for each detected object
stored_conf_score = {}

def draw_bounding_boxes(frame, detections):
    """Draw bounding boxes around detected objects."""
    coords = detections.prediction.bboxes_xyxy
    conf_scores = detections.prediction.confidence
    labels = detections.prediction.labels
    class_names = detections.class_names
    label_names = [class_names[label].capitalize() for label in labels]

    # Alpha controls the smoothing factor
    alpha = 0.05
    
    for i, (bbox, conf_score, label_name) in enumerate(zip(coords, conf_scores, label_names)):
        x1, y1, x2, y2 = map(int, bbox)

        prev_conf_score = stored_conf_score.get(str(i), 0)
        smooth_conf_score = (1 - alpha) * prev_conf_score + alpha * conf_score
        stored_conf_score[str(i)] = smooth_conf_score
        
        # Smooth transition logic for neon-like colors
        if conf_score < 0.55:
            colour = (int(255 * (0.5 + smooth_conf_score)), int(255 * (0.5 - smooth_conf_score)), 255) # BGR format
        elif conf_score < 0.7:
            factor = (smooth_conf_score - 0.55) / 0.15
            colour = (int(255 * factor), int(255 * (1 - factor)), int(255 * (1 - factor)))
        else:
            factor = (smooth_conf_score - 0.7) / 0.3
            colour = (int(255 * factor), int(255 * (1 - factor)), 255)

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # Prepare text
        label_text = f"{label_names[i]}: {conf_score*100:.1f}%"
        text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_x = x1 + 5
        text_y = max(y1 - 5, text_size[1] + 10)

        cv2.rectangle(frame, (x1, int(text_y - text_size[1] - 10)), (int(text_x + text_size[0] + 10), int(text_y + 5)), (0, 0, 0), -1)
        cv2.putText(frame, label_text, (int(text_x), int(text_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame



def calculate_est_distance(bbox, label_name, dimensions_dict):
    """Estimates the distance to the detected object based on its bounding box and label."""
    X1, Y1, X2, Y2 = bbox
    OBJ_HEIGHT = Y2 - Y1
    OBJ_WIDTH = X2 - X1
    OBJ_ASPECT_RATIO = OBJ_WIDTH / OBJ_HEIGHT

    KNOWN_HEIGHT = dimensions_dict[label_name]['height']
    KNOWN_WIDTH = dimensions_dict[label_name]['width']
    KNOWN_ASPECT_RATIO = KNOWN_WIDTH / KNOWN_HEIGHT

    # Threshold for aspect ratio deviation
    aspect_ratio_tolerance = 0.8

    try:
        FOCAL_LENGTH = sv.avg_cam_focal_length[sv.cam_index]
    except KeyError:
        FOCAL_LENGTH = 800

    if OBJ_ASPECT_RATIO < KNOWN_ASPECT_RATIO - aspect_ratio_tolerance:
        # Objects width is off screen
        distance = (KNOWN_HEIGHT * FOCAL_LENGTH) / OBJ_HEIGHT
    elif OBJ_ASPECT_RATIO > KNOWN_ASPECT_RATIO + aspect_ratio_tolerance:
        # Objects height is off screen
        distance = (KNOWN_WIDTH * FOCAL_LENGTH) / OBJ_WIDTH
    else:
        # Both object width and height are on screen
        width_based_distance = (KNOWN_WIDTH * FOCAL_LENGTH) / OBJ_WIDTH
        height_based_distance = (KNOWN_HEIGHT * FOCAL_LENGTH) / OBJ_HEIGHT

        # Combine into one distance with a weighted average
        weight_width = abs(KNOWN_ASPECT_RATIO - OBJ_ASPECT_RATIO)
        weight_height = 1 - weight_width

        distance = (weight_width * width_based_distance + weight_height * height_based_distance) / 2
    
    return distance



def draw_est_distance(frame, detections):
    """Estimate distances to detected objects."""
    bboxes = detections.prediction.bboxes_xyxy
    labels = detections.prediction.labels
    class_names = detections.class_names


    dimensions_dict = extract_json_2_dict(absolute_path('RapidVision', 'object_dimensions.json', 'data'))
    height, _, _ = frame.shape

    for bbox, label in zip(bboxes, labels):
        label_name = class_names[label].lower()
        est_distance = calculate_est_distance(bbox, label_name, dimensions_dict)
        
        x2, y2 = map(int, bbox[2:])
        label_text = f"Distance: {est_distance:.2f}m"

        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        text_x = x2 - text_width - 10
        text_y = min(y2 + text_height - 5, height - text_height - 10)

        # Draw filled rectangle behind the text
        cv2.rectangle(frame,
                      (text_x, text_y + text_height + 10),
                      (text_x + text_width + 10, text_y - text_height + 5),
                      (0, 0, 0),
                      -1)

        # Draw text
        cv2.putText(frame,
                    label_text,
                    (text_x + 5, text_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1)
    
    return frame



def display_detections(frame):
    """Draw bounding boxes around detected objects and get number of detected objects."""

    detections = sv.latest_detections
    frame = draw_bounding_boxes(frame, detections)
    detection_num, class_counts = count_detections(detections)

    # If distance estimation is enabled, draw estimated distances on the frame
    if Settings.distance_estimation:
        frame = draw_est_distance(frame, detections)

    # Return the number of detections, class counts, and the processed frame
    return detection_num, class_counts, frame



def count_detections(detections):
    """Count total detections with a score above the threshold."""
    detected_class_IDs = detections.prediction.labels
    class_names = detections.class_names

    detection_num = len(detected_class_IDs)

    class_counts = Counter(class_names[label] for label in detected_class_IDs)

    formatted_output = ', '.join(f'{class_name.capitalize()}: {count}' for class_name, count in class_counts.items())

    return detection_num, formatted_output.split(', ')


def read_objects(model, device):
    """Read objects from the live feed and update latest detections."""
    while not sv.stop_thread.is_set():
        # If the camera calibration flag is set, calibrate the camera
        if Settings.calibrate_camera:
            print('Calibrating camera...')
            cam_cali()
        
        # If a new frame is available, perform object detection and update the latest detections
        elif sv.latest_frame is not None:
            latest_frame = sv.latest_frame
            frame = latest_frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB as TensorFlow expects RGB format
            frame_tensor = from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(device)  # Convert frame to PyTorch tensor
            
            # Perform object detection on the frame
            results = model.predict(frame_tensor, conf=0.4, fuse_model=False)  # Perform object detection on the frame

            # Update the latest results and detections in a thread-safe manner
            sv.latest_detections = results