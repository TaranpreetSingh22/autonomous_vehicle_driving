import numpy as np

# KITTI calibration (used only for depth scaling here)
focal_length = 721.5377
baseline = 0.54

# ---- Intrinsics (use your KITTI focal; principal point ~ image center if not loaded from calib) ----
def get_K_and_Kinv(depth):
    """
    Get camera intrinsics matrix K and its inverse Kinv.

    Parameters:
    w (int): Image width.
    h (int): Image height.

    Returns:
    K (np.ndarray): Camera intrinsics matrix.
    Kinv (np.ndarray): Inverse of camera intrinsics matrix.
    """
    h, w = depth.shape
    fx = fy = float(focal_length) 
    cx = w / 2.0
    cy = h / 2.0
    K  = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0,  0,  1]], dtype=np.float32)
    Kinv = np.linalg.inv(K)

    return K, Kinv

