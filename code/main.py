import cv2
import numpy as np
import tensorflow as tf
import asyncio
import time
import threading
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.list_physical_devices('GPU')

from useful_funcs import center_pos
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Constants
WINDOW_NAME = "Live Feed"
LEFT_ARROW_KEY = 2424832
RIGHT_ARROW_KEY = 2555904
MODEL_PATH = 'C:\Coding Projects\Projects\Python\RapidVision\code\ssd_resnet50_v1_fpn_640x640_coco17_tpu-8'
LABEL_MAP_PATH = 'C:\Coding Projects\Projects\Python\RapidVision\code\mscoco_label_map.pbtxt'

# Global variables
camera_index = 0
allow_right = True
allow_left = True
last_direction = None
quit_app = False

# Shared variables for the frame capturing and bounding boxes
latest_detections = None
latest_frame = None
frame_lock = threading.Lock()

# Initialize video capture
live_feed = cv2.VideoCapture(camera_index)

# Get text width and height for centering
text = "No Feed"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
thickness = 2
(text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

# Create a notice text
error_text_no_feed = cv2.putText(
    img=np.zeros((480, 640, 3), dtype=np.uint8),
    text=text,
    org=center_pos((text_height, text_width), (480//2, 640//2)),
    fontFace=font,
    fontScale=font_scale,
    color=(255, 255, 255),
    thickness=thickness,
    lineType=cv2.LINE_AA
)

def key_management():
    """Handle key presses."""
    global last_direction, allow_left, allow_right, live_feed, camera_index, quit_app
    key = cv2.waitKeyEx(1)
    reset_camera = False

    if key == ord('q'):
        quit_app = True
    
    elif key == RIGHT_ARROW_KEY and allow_right:
        print("Switching to right camera...")
        reset_camera = True
        last_direction = 1
        allow_left = True
        camera_index += 1
        
    elif key == LEFT_ARROW_KEY and allow_left:
        print("Switching to left camera...")
        reset_camera = True
        last_direction = 0
        allow_right = True
        camera_index -= 1

    if reset_camera:
        live_feed.release()
        live_feed = cv2.VideoCapture(camera_index)

        print(f"Camera index: {camera_index}")


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

def read_objects(model):
    """Read objects from the live feed and update latest detections."""
    global latest_frame, latest_detections
    print("Initializing object detection...")
    
    while True:
        #print("Processing frame...")
        try:
            with frame_lock:
                if latest_frame is not None:
                    frame = latest_frame.copy()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert frame to RGB as TensorFlow expects RGB format

            input_tensor = tf.convert_to_tensor(frame)
            input_tensor = input_tensor[tf.newaxis,...]

            detections = model(input_tensor)
            
            # Get detected objects
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # Detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            # Update the latest detections in thread-safe manner
            with frame_lock:
                latest_detections = detections

            if quit_app:
                break

            time.sleep(0.08)
        
        except:
            pass

async def read_frame():
    """Read the current frame from the live feed."""
    global allow_left, allow_right, latest_frame
    print("Initializing frame capturing...")

    while True:
        ret, frame = live_feed.read()
        if not ret:
            # Doesn't allow switching to more unavailable cameras
            if last_direction == 1:
                allow_right = False
            else:
                allow_left = False
            # Create a notice text
            frame = error_text_no_feed

        with frame_lock:
            latest_frame = frame.copy()
                   
        if quit_app:
            break

        await asyncio.sleep(0.02) # Async sleep to cooperate with the main event loop

    live_feed.release()

async def display_frames(category_index):
    global latest_frame, latest_detections
    print("Initializing frame display...")

    while True:
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()

                # If detections are available, draw bounding boxes
                if latest_detections is not None:
                    draw_object_bounds(frame, latest_detections, category_index)
                
                cv2.imshow(WINDOW_NAME, frame)

        key_management()
        if quit_app:
            break

        await asyncio.sleep(0.02) # Async sleep to cooperate with the main event loop
    
    cv2.destroyAllWindows()

async def main():
    global MODEL_PATH, LABEL_MAP_PATH, run_detection_thread
    print("Initializing model...")
    model = tf.saved_model.load(MODEL_PATH)
    print("Initializing category index...")
    category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH, use_display_name=True)

    # Start frame display in separate thread
    detection_thread = threading.Thread(target=read_objects, args=(model,), daemon=True)
    detection_thread.start()

    # Run the async frame capture and display concurrently
    await asyncio.gather(
        read_frame(),
        display_frames(category_index)
    )

    # Cleanup
    run_detection_thread = False
    detection_thread.join()

if __name__ == "__main__":
    asyncio.run(main())