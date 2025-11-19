import numpy as np

def compute_disparity_metrics(disp, ground_truth_disp, ground_truth_mask):
    """
    Compute disparity metrics comparing estimated disparity with ground truth.

    Parameters:
    - disp: np.ndarray
        Estimated disparity map.
    - ground_truth_disp: np.ndarray
        Ground truth disparity map.
    - ground_truth_mask: np.ndarray
        Boolean mask indicating valid pixels in the ground truth.

    Returns:
    - None
        Prints out the computed metrics.
    """

    if disp.shape != ground_truth_disp.shape or disp.shape != ground_truth_mask.shape:
        raise ValueError("Input arrays must have the same shape.")

    # Compare with ground truth
    abs_error = np.abs(disp - ground_truth_disp)
    masked_error = abs_error[ground_truth_mask]
    mean_abs_error = np.mean(masked_error)
    bad_pixels = np.sum(masked_error > 3.0) / np.sum(ground_truth_mask) * 100

    rel_error = np.zeros_like(ground_truth_disp)
    rel_error[ground_truth_mask] = abs_error[ground_truth_mask] / ground_truth_disp[ground_truth_mask]
    mean_rel_error = np.mean(rel_error[ground_truth_mask])
    rel_error_percent = mean_rel_error * 100

    print("\n===== DISPARITY METRICS =====")
    print(f"Mean Relative Error: {mean_rel_error:.4f}")
    print(f"Relative Error (%): {rel_error_percent:.2f}%")
    print(f"Mean Absolute Error: {mean_abs_error:.2f} px")
    print(f"Bad Pixel Percentage (>3 px error): {bad_pixels:.2f}%")

    return rel_error, abs_error

def compute_depth_metrics(depth, gt_depth, ground_truth_mask):
    """
    Compute depth metrics comparing estimated depth with ground truth.

    Parameters:
    - depth: np.ndarray
        Estimated depth map.
    - ground_truth_depth: np.ndarray
        Ground truth depth map.
    - ground_truth_mask: np.ndarray
        Boolean mask indicating valid pixels in the ground truth.

    Returns:
    - None
        Prints out the computed metrics.
    """
    if depth.shape != gt_depth.shape or depth.shape != ground_truth_mask.shape:
        raise ValueError("Input arrays must have the same shape.")

    # Depth errors
    depth_abs_error = np.abs(depth - gt_depth)
    depth_rel_error = np.zeros_like(gt_depth, dtype=np.float32)
    depth_rel_error[ground_truth_mask] = depth_abs_error[ground_truth_mask] / gt_depth[ground_truth_mask]
    mean_depth_abs_error = np.mean(depth_abs_error[ground_truth_mask])
    mean_depth_rel_error = np.mean(depth_rel_error[ground_truth_mask])
    depth_rel_error_percent = mean_depth_rel_error * 100

    print("\n===== DEPTH METRICS =====")
    print(f"Mean Absolute Depth Error: {mean_depth_abs_error:.2f} m")
    print(f"Mean Relative Depth Error: {mean_depth_rel_error:.4f}")
    print(f"Relative Depth Error (%): {depth_rel_error_percent:.2f}%")

    return depth_abs_error, depth_rel_error