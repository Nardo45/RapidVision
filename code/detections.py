import numpy as np
import useful_funcs as uf

import cv2, json, os

from threading   import Lock
from collections import Counter
from torch       import from_numpy
from PIL         import Image, ImageDraw, ImageFont
from shared_data import shared_variables as sv, Settings
from cam_cali    import cam_cali

# Lock to ensure thread-safe access to the frame
frame_lock = Lock()

# Dictionary to store the smoothed confidence scores for each detected object
stored_conf_score = {}

def draw_bounding_boxes(frame, detections):
    """Draw bounding boxes around detected objects."""
    coords = detections.prediction.bboxes_xyxy
    conf_scores = detections.prediction.confidence

    # Get all detected labels
    labels = detections.prediction.labels
    class_names = detections.class_names
    label_names = [class_names[label].capitalize() for label in labels]

    # Convert numpy array to PIL image
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)

    # Create an invisible overlay
    overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    # Alpha controls the smoothing factor
    alpha = 0.05

    font = ImageFont.load_default(15)
    
    for i in range(len(conf_scores)):
        x1, y1, x2, y2 = coords[i]

        conf_score = conf_scores[i]

        # Checks if the list of previous conf scores for this object exists
        try:
            prev_conf_score = stored_conf_score[str(i)]
        except KeyError:
            prev_conf_score = 0

        # Apply exponential smoothing to the conf score
        smooth_conf_score = (1 - alpha) * prev_conf_score + alpha * conf_score
        stored_conf_score[str(i)] = smooth_conf_score
        
        # Smooth transition logic for neon-like colors
        if conf_score < 0.55:
            red = 255
            green = int(255 * (0.5 - smooth_conf_score))
            blue = int(255 * (0.5 + smooth_conf_score))  # Scale between 128-255
        elif 0.55 <= conf_score < 0.7:
            # Interpolate between previous pink (255, 0, 255) and neon green (0, 255, 0)
            factor = (smooth_conf_score - 0.55) / (0.7 - 0.55)  # Normalize to range 0-1
            red = int(255 * (1 - factor))  # Transition from 255 to 0
            green = int(255 * factor)      # Transition from 0 to 255
            blue = int(255 * (1 - factor))
        else:
            # Interpolate between neon green (0, 255, 0) and neon cyan (0, 255, 255)
            factor = (smooth_conf_score - 0.7) / (1 - 0.7)  # Normalize to range 0-1
            red = int(255 * (1 - factor)) 
            green = int(255 * (1 - factor))   # Transition from 255 to 128 as factor increases
            blue = int(255 * factor)         # Transition from 0 to 255

        # Set the opacity based on the confidence score
        min_opacity = 50
        max_opacity = 155
        opacity = int(min_opacity + (max_opacity - min_opacity) * conf_score)

        overlay_draw.rectangle([(x1, y1), (x2, y2)], outline=(blue, green, red, opacity), width=6)

        # Draw the overlay on the frame
        pil_img.paste(overlay, (0, 0), overlay)

        # Draw the filled rectangle with labels and confidence scores
        label_text = f'{label_names[i]}: {conf_scores[i]*100:.1f}%'
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x, text_y = x1, y1 - text_height - 10

        # Draw filled rectangle behind the label
        draw.rectangle([text_x, text_y , text_x + text_width + 10, y1], fill=(0, 0, 0))
        
        # Draw the label and confidence score
        draw.text((text_x + 5, text_y), label_text, font=font, fill=(255, 255, 255))

    # Convert the PIL image back to a numpy array
    return np.array(pil_img)



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
        # Both object width and height are off screen
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

    # Get all detected labels
    labels = detections.prediction.labels
    class_names = detections.class_names
    label_names = [class_names[label].capitalize() for label in labels]
    label_names = [string.lower() for string in label_names]

    # Convert numpy array to PIL image
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)

    font = ImageFont.load_default(15)
    dimensions_dict = uf.extract_json_2_dict(uf.absolute_path('RapidVision', 'object_dimensions.json', 'data'))

    for i in range(len(label_names)):
        est_distance = calculate_est_distance(bboxes[i], label_names[i], dimensions_dict)
        
        _, _, x2, y2 = bboxes[i]

        # Draw the distance on the frame
        label_text = f'Distance: {est_distance:.2f}m'
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x, text_y = x2 - text_width - 10, y2 + text_height - 10

        height, _, _ = frame.shape
        if text_y + text_height > height - 10:
            text_y = height - text_height - 10

        # Draw filled rectangle behind the label
        draw.rectangle([text_x, text_y, text_x + text_width + 10, y2 + text_height + 10], fill=(0, 0, 0))
        
        # Draw the label and distance
        draw.text((text_x + 5, text_y), label_text, font=font, fill=(255, 255, 255))
    
    # Convert the PIL image back to a numpy array
    return np.array(pil_img)



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

    # Count total detections with a score above 0.5
    detection_num = len(detected_class_IDs)

    # Create a dictionary to count detections per class
    class_names = detections.class_names
    per_class_detections = {}
    for index in detected_class_IDs:
        if class_names[index] in per_class_detections:
            per_class_detections[class_names[index]] += 1
        else:
            per_class_detections[class_names[index]] = 1

    class_counts = Counter(per_class_detections)

    # Create the formatted string
    formatted_output = ', '.join([f"{str(class_name).capitalize()}: {count}" for class_name, count in class_counts.items()])

    class_counts = formatted_output.split(', ')

    return detection_num, class_counts


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