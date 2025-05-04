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

    while num_imgz < NUM_SAMPLES and sv.latest_frame is not None:
        if not Settings.calibrate_camera:
            print('User cancelled calibration')
            return
        
        frame = sv.latest_frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 and frame.shape[2] == 3 else frame.copy()

        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            num_imgz += 1
            print(f'frame {num_imgz}/{NUM_SAMPLES}')
            cv2.waitKey(200)

        if num_imgz >= NUM_SAMPLES:
            Settings.calibrate_camera = False
            break

    # Calculate the focal length of the selected camera
    ret, camera_matrix, _, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Extract the focal length of the selected camera
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    focal_length = (fx + fy) / 2  # Average the focal length
    sv.avg_cam_focal_length[sv.cam_index] = focal_length

    # New data to be saved
    new_data = {
        f'cam {sv.cam_index}': {
            'focal_length': focal_length
        }
    }

    cali_data_path = absolute_path('RapidVision', 'cam_cali_data.json', 'data')

    # Save the updated camera calibration data to JSON file
    save_2_json(new_data, cali_data_path)