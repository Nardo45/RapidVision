import cv2
import numpy as np

from useful_funcs import center_pos

# Constants
WINDOW_NAME = "Live Feed"

# Global variables
camera_index = 0
allow_right = True
allow_left = True
last_direction = None

# Captures the video from the default camera connection
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
    global last_direction, allow_left, allow_right, live_feed, camera_index
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        return 0
    
    elif key == ord('r') and allow_right:
        last_direction = 1
        allow_left = True
        camera_index += 1
        live_feed.release()
        live_feed = cv2.VideoCapture(camera_index)
        
    elif key == ord('l') and allow_left:
        last_direction = 0
        allow_right = True
        camera_index -= 1
        live_feed.release()
        live_feed = cv2.VideoCapture(camera_index)

def read_frame():
    """Read the current frame from the live feed."""
    global allow_left, allow_right
    ret, frame = live_feed.read()

    if not ret:
        # Doesn't allow switching to more unavailable cameras
        if last_direction == 1:
            allow_right = False
        else:
            allow_left = False
        # Create a notice text
        frame = error_text_no_feed
    return frame

def main():
    """Main loop."""
    while True:
        # Read the current frame from the live feed
        frame = read_frame()

        cv2.imshow(WINDOW_NAME, frame)

        # Call the key management function
        result = key_management()
        if result == 0:
            break

    # Release the video capture and close all windows
    live_feed.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()