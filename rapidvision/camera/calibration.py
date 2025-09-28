import time
import cv2
import numpy as np

from rapidvision.utils.general import absolute_path, save_2_json
from rapidvision.detection.shared_data import shared_variables as sv, Settings


def cam_cali():
    """Calibrate the camera using a checkerboard pattern."""
    CHECKERBOARD = (6, 6)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane

    NUM_SAMPLES = 20
    num_imgz = 0

    # Prepare object points
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    # Grab frames until we have enough samples or the user cancels
    last_gray = None
    while num_imgz < NUM_SAMPLES and sv.latest_frame is not None:
        if not Settings.calibrate_camera:
            print('User cancelled calibration')
            return

        frame = sv.latest_frame
        if frame is None:
            time.sleep(0.05)
            continue

        # Make sure we have a grayscale image for chessboard detection
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        last_gray = gray  # keep last valid gray for later use

        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret:
            objpoints.append(objp.copy())
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            num_imgz += 1
            print(f'frame {num_imgz}/{NUM_SAMPLES}')
            # short pause so user can move board — non-blocking
            cv2.waitKey(200)
        else:
            # small sleep so we don't spin tightly when frames don't have a board
            time.sleep(0.05)

        # If we've collected enough frames, stop calibration loop
        if num_imgz >= NUM_SAMPLES:
            Settings.calibrate_camera = False
            break

    # If we never detected any valid chessboards, abort gracefully
    if len(objpoints) == 0 or last_gray is None:
        print("Calibration aborted: no valid chessboard frames collected.")
        Settings.calibrate_camera = False
        return

    # Calculate the focal length of the selected camera
    # Use last_gray.shape[::-1] as image size
    try:
        ret, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, last_gray.shape[::-1], None, None
        )
    except Exception as e:
        print(f"Calibration failed: {e}")
        Settings.calibrate_camera = False
        return

    if camera_matrix is None or camera_matrix.shape[0] < 2:
        print("Calibration failed: invalid camera matrix.")
        Settings.calibrate_camera = False
        return

    # Extract the focal length of the selected camera
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    focal_length = (fx + fy) / 2.0  # Average the focal length

    # Store in shared variables
    sv.avg_cam_focal_length[sv.cam_index] = focal_length

    # New data to be saved (same shape as your JSON expects)
    new_data = {
        f'cam {sv.cam_index}': {
            'focal_length': float(focal_length)
        }
    }

    # get path (string) and write JSON — NOTE: path first, data second
    cali_data_path = absolute_path('RapidVision', 'cam_cali_data.json', 'data')

    try:
        # Correct order: save_2_json(path, data)
        save_2_json(cali_data_path, new_data)
        print(f"Calibration saved for cam {sv.cam_index}: focal_length={focal_length}")
    except Exception as e:
        print(f"Failed to save calibration data: {e}")

    # Ensure flag reset
    Settings.calibrate_camera = False
