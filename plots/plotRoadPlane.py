import numpy as np
import os
import matplotlib.pyplot as plt

def plotRoadPlane(left_img, road_mask, max_depth_thresh=20.0, save_path=False):
    # === Separate Road Mask Visualization ===
    fig = plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title(f"Detected Road Plane (<= {max_depth_thresh} m)")
    plt.imshow(left_img, cmap='gray')
    plt.imshow(np.ma.masked_where(road_mask == 0, road_mask), cmap='autumn', alpha=0.5)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Road Mask Only")
    plt.imshow(road_mask, cmap='gray')
    plt.axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[INFO] Saved BEV frame to: {save_path}")
    else:
        plt.show()