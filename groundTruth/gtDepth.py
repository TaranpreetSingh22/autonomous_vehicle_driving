import numpy as np
from parameters.intrinsics import focal_length, baseline

def get_depth_from_disparity(ground_truth_disp):
    """
    Convert disparity map to depth map.

    Parameters:
    disp (np.ndarray): Disparity map.
    focal_length (float): Focal length of the camera.
    baseline (float): Baseline distance between the stereo cameras.

    Returns:
    np.ndarray: Depth map.
    """
    # Depth maps
    gt_depth = np.zeros_like(ground_truth_disp, dtype=np.float32)
    gt_depth[ground_truth_disp > 0] = (focal_length * baseline) / ground_truth_disp[ground_truth_disp > 0]

    return gt_depth