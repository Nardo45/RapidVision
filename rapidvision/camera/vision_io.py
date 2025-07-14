import cv2

from rapidvision.detection.shared_data import shared_variables as sv
from rapidvision.utils.general import absolute_path, extract_json_2_dict

def try_set_camera_format(cap, width, height, fps):
    """Attempt to set the camera format to the specified width, height, and fps."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    return (actual_width == width and actual_height == height and actual_fps == fps)



def load_camera_profiles():
    """Load the JSON file of common camera profiles."""
    json_path = absolute_path('RapidVision', 'camera_profiles.json', 'config')
    profiles: dict = extract_json_2_dict(json_path)
    return profiles.get("common_camera_profiles", [])



def camera_capture():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    profiles = load_camera_profiles()

    selected = False
    for profile in profiles:
        width = profile['width']
        height = profile['height']
        fps = profile['fps']
        if try_set_camera_format(cap, width, height, fps):
            selected = True
            sv.set_current_cam_profile(profile)
            break

    if not selected:
        print("No suitable camera profile found. Using default settings.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

    while not sv.stop_thread.is_set():
        ret, frame = cap.read()
        if not ret:
            sv.set_latest_frame(None)
        else:
            sv.set_latest_frame(frame)

        if sv.get_reset_camera():
            cap.release()
            cap = reset_camera()
            sv.set_reset_camera(False)

def reset_camera():
    """Reset the camera to the current profile settings."""
    print("Resetting camera to current profile settings...")
    if sv.get_cam_idx() < 0: sv.set_cam_idx(0)
    cap = cv2.VideoCapture(sv.get_cam_idx())
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    profile = sv.get_current_cam_profile()

    if profile:
        width = profile['width']
        height = profile['height']
        fps = profile['fps']

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
    else:
        print("No camera profile set. Using default settings.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera. Setting latest frame to None.")
        sv.set_latest_frame(None)
        print(sv.get_latest_frame())
    else:
        sv.set_latest_frame(frame)

    return cap