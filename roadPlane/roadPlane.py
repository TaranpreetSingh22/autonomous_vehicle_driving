import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor
from parameters.depthThresholds import min_depth_thresh, max_depth_thresh

def detect_road_plane(depth):
    """
    Detect the road plane in a depth map using RANSAC on inverse depth.

    Parameters:
    - depth: np.ndarray
        Input depth map.

    Returns:
    - road_mask: np.ndarray
        Binary mask indicating detected road plane regions.
    """

    if depth.ndim != 2:
        raise ValueError("Input depth map must be a 2D array.")

    # === ROAD PLANE DETECTION (≤ 10 m) — INTRINSICS-FREE via inverse depth plane ===
    print("\nDetecting road plane up to 10 m (intrinsics-free)...")

    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    valid = (depth > 0) & np.isfinite(depth) & (depth >= min_depth_thresh) & (depth <= max_depth_thresh)
    valid &= (v >= h * 0.5)  # focus on bottom half to reduce walls/cars

    # Build inverse depth target
    inv_depth = np.zeros_like(depth, dtype=np.float32)
    inv_depth[valid] = 1.0 / depth[valid]

    # Features in pixel space (u, v), target = 1/Z
    U = u[valid].reshape(-1)
    V = v[valid].reshape(-1)
    Y = inv_depth[valid].reshape(-1)
    X_feat = np.stack([U, V], axis=1).astype(np.float32)

    # Optional subsample for speed on large images
    if X_feat.shape[0] > 250_000:
        idx = np.random.choice(X_feat.shape[0], 250_000, replace=False)
        X_fit, Y_fit = X_feat[idx], Y[idx]
    else:
        X_fit, Y_fit = X_feat, Y

    # RANSAC: fit plane 1/Z ≈ a*u + b*v + c
    ransac = RANSACRegressor(residual_threshold=0.003, max_trials=1000, random_state=0)  # ~3e-3 (1/m)
    ransac.fit(X_fit, Y_fit)

    # Inlier check (on all valid points)
    Y_pred = ransac.predict(X_feat)
    inliers = np.abs(Y - Y_pred) < 0.005  # tighten/loosen as needed 0.003

    # Map back to image mask
    road_mask = np.zeros_like(depth, dtype=np.uint8)
    road_mask[valid] = inliers.astype(np.uint8)

    # Clean up mask
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    return road_mask