import numpy as np
from parameters.intrinsics import focal_length, baseline

def get_depth_from_disparity(disp):
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
    depth = np.zeros_like(disp, dtype=np.float32)
    depth[disp > 0] = (focal_length * baseline) / disp[disp > 0]

    return depth