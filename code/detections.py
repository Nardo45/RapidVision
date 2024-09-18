import numpy as np

from threading   import Lock
from collections import Counter
from torch       import from_numpy
from cv2         import cvtColor, COLOR_BGR2RGB
from PIL         import Image, ImageDraw, ImageFont
from shared_data import shared_variables as sv

frame_lock = Lock()

def draw_bounding_boxes(frame, detections):
    """Draw bounding boxes around detected objects."""
    coords = detections.prediction.bboxes_xyxy
    conf_scores = detections.prediction.confidence

    # Get all detected lables
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
    alpha = 0.1
    
    for i in range(len(conf_scores)):
        x1, y1, x2, y2 = coords[i]
        width, height = x2 - x1, y2 - y1

        font = ImageFont.load_default(min(width, height) * 0.07) # Font size is adjusted based on the bbox size

        # Calculate the color based on the confidence score
        conf_score = conf_scores[i]

        # Apply exponential smoothing to the conf score
        smooth_conf_score = (1 - alpha) * sv.prev_conf_score + alpha * conf_score
        sv.prev_conf_score = smooth_conf_score
        
        # Smooth transition logic for neon-like colors
        if conf_score < 0.55:
            # Neon pinks and purples (high red and blue)
            red = 255
            green = 0
            blue = int(255 * (0.5 + smooth_conf_score))  # Scale between 128-255
        elif 0.55 <= conf_score < 0.7:
            # Smooth transition between pink/purple to neon green
            # Interpolate between previous pink (255, 0, 255) and neon green (0, 255, 0)
            factor = (smooth_conf_score - 0.55) / (0.7 - 0.55)  # Normalize to range 0-1
            red = int(255 * (1 - factor))  # Transition from 255 to 0
            green = int(255 * factor)      # Transition from 0 to 255
            blue = int(255 * (1 - factor)) # Transition from 255 to 0
        else:
            # Smooth transition between neon green to neon blue/cyan
            # Interpolate between neon green (0, 255, 0) and neon cyan (0, 255, 255)
            factor = (smooth_conf_score - 0.7) / (1 - 0.7)  # Normalize to range 0-1
            red = 0
            green = int(255 * (1 - factor))  # Transition from 255 to 128
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
        text_x, text_y = x1, y1 - text_height - min(width, height) * 0.05

        # Draw filled rectangle behind the label
        draw.rectangle([text_x, text_y , text_x + text_width + 10, y1],
                       fill=(blue, green, red))
        
        # Draw the label and confidence score
        draw.text((text_x + 5, text_y), label_text, font=font, fill=(0, 0, 0))

    # Convert the PIL image back to a numpy array
    return np.array(pil_img)



def display_detections(frame):
    """Draw bounding boxes around detected objects and get number of detected objects."""

    detections = sv.latest_detections

    frame_with_detections = draw_bounding_boxes(frame, detections)

    detection_num, class_counts = count_detections(detections)

    return detection_num, class_counts, frame_with_detections



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
        with frame_lock:
            latest_frame = sv.latest_frame
            if latest_frame is not None:
                frame = latest_frame.copy()
                frame = cvtColor(frame, COLOR_BGR2RGB) # Convert frame to RGB as TensorFlow expects RGB format
                frame_tensor = from_numpy(frame).permute(2,0,1).unsqueeze(0).to(device) # Convert frame to PyTorch tensor
                
                # Perform object detection on the frame
                results = model.predict(frame_tensor, conf=0.4, fuse_model=False) # Perform object detection on the frame

                # Update the latest results and detections in thread-safe manner
                sv.latest_detections = results