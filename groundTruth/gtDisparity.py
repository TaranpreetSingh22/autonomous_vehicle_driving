import cv2
import numpy as np

def get_ground_truth_disp(disp_path):
    ############################################################################
    # Load KITTI ground truth disparity map from the specified path.
    ############################################################################

    ground_truth_disp = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
    ground_truth_mask = ground_truth_disp > 0  # Valid disparity pixels

    return ground_truth_disp, ground_truth_mask