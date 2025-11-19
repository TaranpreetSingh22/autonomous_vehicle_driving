import os
import matplotlib.pyplot as plt

def plotMetrics (disp, ground_truth_disp, ground_truth_mask, abs_error, rel_error,
                      depth, gt_depth, depth_abs_error, depth_rel_error, save_path=False):
    # === Visualization === 
    fig = plt.figure(figsize=(20, 10))

    # Row 1: Disparity metrics
    plt.subplot(2, 4, 1)
    plt.title("Disparity (SGBM)")
    plt.imshow(disp * ground_truth_mask, cmap='plasma', vmin=0, vmax=20)
    plt.colorbar(); plt.axis("off")

    plt.subplot(2, 4, 2)
    plt.title("Ground Truth Disparity")
    plt.imshow(ground_truth_disp * ground_truth_mask, cmap='plasma', vmin=0, vmax=20)
    plt.colorbar(); plt.axis("off")

    plt.subplot(2, 4, 3)
    plt.title("Absolute Error")
    plt.imshow(abs_error * ground_truth_mask, cmap='inferno', vmin=0, vmax=20)
    plt.colorbar(); plt.axis("off")

    plt.subplot(2, 4, 4)
    plt.title("Relative Error (%)")
    plt.imshow(rel_error * 100, cmap='viridis', vmin=0, vmax=100)
    plt.colorbar(); plt.axis("off")

    # Row 2: Depth metrics
    plt.subplot(2, 4, 5)
    plt.title("Depth (SGBM)")
    plt.imshow(depth, cmap='plasma', vmin=1.0, vmax=20.0)
    plt.colorbar(); plt.axis("off")

    plt.subplot(2, 4, 6)
    plt.title("Ground Truth Depth")
    plt.imshow(gt_depth, cmap='plasma', vmin=1.0, vmax=20.0)
    plt.colorbar(); plt.axis("off")

    plt.subplot(2, 4, 7)
    plt.title("Absolute Depth Error")
    plt.imshow(depth_abs_error, cmap='inferno', vmin=0, vmax=20)
    plt.colorbar(); plt.axis("off")

    plt.subplot(2, 4, 8)
    plt.title("Relative Depth Error (%)")
    plt.imshow(depth_rel_error * 100, cmap='viridis', vmin=0, vmax=100)
    plt.colorbar(); plt.axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[INFO] Saved BEV frame to: {save_path}")
    else:
        plt.show()