import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plotGlobalStitch(projector, save_path=False):
    """
    Plots the stitched global BEV map from the GlobalPathProjector instance.
    Args:
        projector (GlobalPathProjector): Instance of the GlobalPathProjector class.
    """
    # Define custom color map for occupancy + path
    cmap = mcolors.ListedColormap([
        (0.15, 0.15, 0.15),  # -1 = unknown
        (0.10, 0.70, 0.10),  #  0 = free
        (0.85, 0.10, 0.10),  #  1 = obstacle
        (0.10, 0.30, 0.85)   #  2 = path
    ])

    global_bev = projector.get_map()
    NZ_global, NX_global = global_bev.shape

    fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    axes = [axes]

    im_global = axes[0].imshow(global_bev, cmap=cmap, origin='lower', vmin=-1, vmax=2)
    axes[0].set_aspect('equal')

    # Labels
    axes[0].set_title("Stitched Global Map (BEV: Free, Obstacles, Path)")
    axes[0].set_xlabel("Lateral distance (m)  [left  ⟵  0  ⟶  right]")
    axes[0].set_ylabel("Forward distance (m)  (0 at ego)")

    # Ticks and meters
    resolution = projector.resolution
    xticks = np.arange(0, NX_global, int(1 / resolution))
    yticks = np.arange(0, NZ_global, int(1 / resolution))
    meters_x = xticks * resolution
    meters_z = yticks * resolution

    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels([f"{x - (NX_global * resolution) / 2:.0f}" for x in meters_x])
    axes[0].set_yticks(yticks)
    axes[0].set_yticklabels([f"{z:.0f}" for z in meters_z])

    # Grid styling
    axes[0].set_xticks(np.arange(0, NX_global, 1), minor=True)
    axes[0].set_yticks(np.arange(0, NZ_global, 1), minor=True)
    axes[0].grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.7)
    axes[0].grid(True, which='minor', linestyle='-', linewidth=0.4, alpha=0.4)

    # Colorbar
    cbar_global = fig.colorbar(im_global, ax=axes[0], ticks=[-1, 0, 1, 2])
    cbar_global.set_label("Occupancy Type")
    cbar_global.set_ticklabels(["Unknown", "Free", "Obstacle", "Path"])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[INFO] Saved BEV frame to: {save_path}")
    else:
        plt.show()