import cv2
import numpy as np

def read_image_grayscale(image_path):
    """
    Read an image from the specified path and convert it to grayscale.

    Args:
        image_path (str): Path to the image file.
    Returns:
        np.ndarray: Grayscale image.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return img

def get_disparity_map(left_image_path, right_image_path):
    """
    Compute the disparity map from a pair of stereo images using StereoSGBM.

    Args:
        left_image_path (str): Path to the left image.
        right_image_path (str): Path to the right image.
    Returns:
        np.ndarray: Disparity map.
    """

    # StereoSGBM parameters
    min_disp = 0
    num_disp = 64
    block_size = 5
    uniqueness_ratio = 5
    speckle_window_size = 100
    speckle_range = 2

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Compute disparity map
    left_img = read_image_grayscale(left_image_path)
    right_img = read_image_grayscale(right_image_path)
    disp = stereo.compute(left_img, right_img).astype(np.float32) / 16.0

    return disp