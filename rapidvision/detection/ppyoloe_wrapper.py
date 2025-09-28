import cv2
import numpy as np
import torchvision
import msgpack
import time

from collections import Counter, deque
from torch import from_numpy, no_grad, max as torch_max, stack as torch_stack
from rapidvision.detection.shared_data import shared_variables as sv, Settings
from rapidvision.camera.calibration import cam_cali
from rapidvision.utils.general import extract_json_2_dict, absolute_path

# Dictionary to store the smoothed confidence scores for each detected object
stored_conf_score = {}


def draw_bounding_boxes(frame, detections):
    """Draw bounding boxes around detected objects."""
    coords = detections["predictions"]["bboxes_xyxy"]
    conf_scores = detections["predictions"]["confidence"]
    labels = detections["predictions"]["labels"]
    class_names = detections["class_names"]
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
        label_text = f"{label_name}: {conf_score*100:.1f}%"
        text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_x = x1 + 5
        text_y = max(y1 - 5, text_size[1] + 10)

        cv2.rectangle(frame, (x1, int(text_y - text_size[1] - 10)), (int(text_x + text_size[0] + 10), int(text_y + 5)), (0, 0, 0), -1)
        cv2.putText(frame, label_text, (int(text_x), int(text_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def calculate_est_distance(bbox, label_name, dimensions_dict):
    """
    Estimates the distance to the detected object based on
    its bounding box and label,
    """
    X1, Y1, X2, Y2 = bbox
    OBJ_HEIGHT = Y2 - Y1
    OBJ_WIDTH = X2 - X1

    # Protect against degenerate boxes
    if OBJ_HEIGHT <= 0 or OBJ_WIDTH <= 0:
        return None

    OBJ_ASPECT_RATIO = OBJ_WIDTH / OBJ_HEIGHT

    KNOWN_HEIGHT = dimensions_dict[label_name]['height']
    KNOWN_WIDTH = dimensions_dict[label_name]['width']
    KNOWN_ASPECT_RATIO = KNOWN_WIDTH / KNOWN_HEIGHT

    # Threshold for aspect ratio deviation
    aspect_ratio_tolerance = 0.8

    # Grab pixel height and width from current cam profile (may be None)
    try:
        CURRENT_PROFILE = sv.get_current_cam_profile()
    except Exception:
        CURRENT_PROFILE = None

    # Get base focal length (fallback to 800 if not found)
    try:
        FOCAL_LENGTH = sv.get_cam_focal_len()[sv.get_cam_idx()]
    except Exception:
        FOCAL_LENGTH = 800

    # scale the focal length to match current profile resolution.
    if CURRENT_PROFILE and isinstance(CURRENT_PROFILE, dict):
        try:
            reference_w = CURRENT_PROFILE.get('width', 1920)
            reference_h = CURRENT_PROFILE.get('height', 1080)

            prof_w = int(reference_w)
            prof_h = int(reference_h)

            # compute scale factor (average of width & height scale)
            scale_x = prof_w / reference_w if reference_w else 1.0
            scale_y = prof_h / reference_h if reference_h else 1.0
            scale = (scale_x + scale_y) / 2.0

            FOCAL_LENGTH = float(FOCAL_LENGTH) * scale
        except Exception:
            # if anything goes wrong, keep original FOCAL_LENGTH
            pass

    # distance calculations
    if OBJ_ASPECT_RATIO < KNOWN_ASPECT_RATIO - aspect_ratio_tolerance:
        # Object's width is likely off-screen: use height
        distance = (KNOWN_HEIGHT * FOCAL_LENGTH) / OBJ_HEIGHT
    elif OBJ_ASPECT_RATIO > KNOWN_ASPECT_RATIO + aspect_ratio_tolerance:
        # Object's height is likely off-screen: use width
        distance = (KNOWN_WIDTH * FOCAL_LENGTH) / OBJ_WIDTH
    else:
        # Both object width and height are on screen
        width_based_distance = (KNOWN_WIDTH * FOCAL_LENGTH) / OBJ_WIDTH
        height_based_distance = (KNOWN_HEIGHT * FOCAL_LENGTH) / OBJ_HEIGHT

        # Weight: normalize the aspect-ratio difference to [0,1] then combine
        diff = abs(KNOWN_ASPECT_RATIO - OBJ_ASPECT_RATIO)
        # clamp normalization denominator to avoid division issues
        denom = max(KNOWN_ASPECT_RATIO, OBJ_ASPECT_RATIO, 1e-6)
        weight_width = min(max(diff / denom, 0.0), 1.0)
        weight_height = 1.0 - weight_width

        distance = (weight_width * width_based_distance + weight_height * height_based_distance) / 2.0

    return float(distance)


def draw_est_distance(frame, detections):
    """Estimate distances to detected objects."""

    bboxes = detections["predictions"]["bboxes_xyxy"]
    labels = detections["predictions"]["labels"]
    class_names = detections["class_names"]

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
    """
    Draw bounding boxes around detected objects and
    get number of detected objects.
    """

    detections = sv.get_latest_detections()
    frame = draw_bounding_boxes(frame, detections)
    detection_num, class_counts = count_detections(detections)

    # If distance estimation is enabled, draw estimated distances on the frame
    if Settings.distance_estimation:
        frame = draw_est_distance(frame, detections)

    # Return the number of detections, class counts, and the processed frame
    return detection_num, class_counts, frame


def count_detections(detections):
    """Count total detections with a score above the threshold."""
    detected_class_IDs = detections["predictions"]["labels"]
    class_names = detections["class_names"]

    detection_num = len(detected_class_IDs)

    class_counts = Counter(class_names[label] for label in detected_class_IDs)

    formatted_output = ', '.join(f'{class_name.capitalize()}: {count}' for class_name, count in class_counts.items())

    return detection_num, formatted_output.split(', ')


def decode_ppyoloe_output(output_tensor, class_names, scale, paddings, conf_thresh=0.5, iou_thresh=0.5):
    """
    Decode raw PP-YOLOE output tensor to usable detections.

    Args:
        output_tensor (torch.Tensor): shape [1, N, M]
        class_names (List[str]): list of class names
        conf_thresh (float): minimum confidence threshold
        iou_thresh (float): IOU threshold for NMS

    Returns:
        A dict mimicking YOLO-NAS-style structure.
    """
    # Remove batch dimensions: [1, N, M] -> [N, M]
    predictions = output_tensor[0]

    # x1, y1, x2, y2
    boxes = predictions[:, :4]
    objectness = predictions[:, 4]
    class_probs = predictions[:, 5:]

    # Final score = objectness * class prob
    scores, labels = torch_max(class_probs, dim=1)
    final_scores = objectness * scores

    # Filter by confidence threshold
    keep = final_scores > conf_thresh  # Creates a boolean mask
    boxes = boxes[keep]
    final_scores = final_scores[keep]
    labels = labels[keep]

    # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes = torch_stack([x1, y1, x2, y2], dim=1)

    # Undo padding and scaling
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - paddings[0]) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - paddings[1]) / scale

    # Apply NMSsudo
    nms_indices = torchvision.ops.nms(boxes, final_scores, iou_thresh)
    boxes = boxes[nms_indices]
    final_scores = final_scores[nms_indices]
    labels = labels[nms_indices]

    return {
        "predictions": {
            "bboxes_xyxy": boxes.cpu().numpy(),
            "confidence": final_scores.cpu().numpy(),
            "labels": labels.cpu().numpy()
        },
        "class_names": class_names
    }


def letterbox_image(image, target_size=(640, 640), color=(114, 114, 114)):
    """
    Resize image with unchanged aspect ratio using padding.

    Args:
        image (np.ndarray): Input image.
        target_size (tuple): Desired output size (width, height).
        color (tuple): Padding color in BGR format.

    Returns:
        np.ndarray: Resized and padded image.
        float: Scale factor used for resizing.
        int: Left padding.
        int: Top padding.
    """
    orig_h, orig_w = image.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Padding
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2

    padded_image = cv2.copyMakeBorder(
        resized_image,
        pad_top, pad_h - pad_top,
        pad_left, pad_w - pad_left,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return padded_image, scale, pad_left, pad_top


def read_objects(model, device):
    """Read objects from the live feed and update latest detections."""
    _inference_times = deque(maxlen=100)
    start = 0
    coco_file = absolute_path("RapidVision", "coco_classes.msgpack", "config")
    with open(coco_file, "rb") as file:
        coco_classes = msgpack.unpack(file, strict_map_key=False)

    while not sv.stop_thread.is_set():
        if Settings.calibrate_camera:
            print('Calibrating camera...')
            cam_cali()

        # If a new frame is available, perform object detection and update the latest detections
        elif sv.get_latest_frame() is not None:
            if Settings.measure_inf_time:
                # Measure model inference time
                start = time.perf_counter()

            img = sv.get_latest_frame()

            # Preprocess the frame for PPYOLOE
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert frame to RGB as TensorFlow expects RGB format
            img_padded, scale, pad_left, pad_top = letterbox_image(img_rgb, (640, 640)) # Resize to expected input size

            img_norm = img_padded.astype(np.float32) / 255.0
            img_norm = (img_norm - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_tensor = from_numpy(np.transpose(img_norm, (2, 0, 1))).unsqueeze(0).to(device)  # Convert frame to PyTorch tensor

            with no_grad():
                outputs = model(img_tensor)

            detections = decode_ppyoloe_output(outputs, coco_classes, scale, (pad_left, pad_top))

            if Settings.measure_inf_time:
                end = time.perf_counter()

                inf_time = end - start  # seconds
                _inference_times.append(inf_time)
                avg_inf = sum(_inference_times) / len(_inference_times)

                sv.set_avg_inference_time(avg_inf)

            # Save the result
            sv.set_latest_detections(detections)
