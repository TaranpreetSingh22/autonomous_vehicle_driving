import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from parameters.localBevParams import CELL_SIZE_M, GRID_WIDTH_M, GRID_DEPTH_M, NX, NZ

def plotBev(bev, road_mask, left_img, max_depth=20, save_path=False):
    """
    Visualize the BEV occupancy grid along with the road mask overlay on the left image.
    Parameters:
    - bev: 2D numpy array of shape (NZ, NX) with values in {-1, 0, 1}
    - road_mask: 2D numpy array of the same height and width as left_img, binary mask (1=road, 0=not-road)
    - left_img: 2D numpy array (grayscale) of the left camera image
    - max_depth_thresh: float, maximum depth threshold used for road detection
    - CELL_SIZE_M: float, size of each cell in meters
    - GRID_WIDTH_M: float, total width of the grid in meters
    - GRID_DEPTH_M: float, total depth of the grid in meters
    """
    # ---- Visualization with cell/grid lines & metric ticks ----
    print("Visualizing BEV (grid with meters)...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

    # ---------------- Left subplot: Road plane ----------------
    axes[0].set_title(f"Detected Road Plane (<= {max_depth} m)")
    axes[0].imshow(left_img, cmap='gray')
    axes[0].imshow(np.ma.masked_where(road_mask == 0, road_mask), cmap='autumn', alpha=0.5)
    axes[0].axis("off")

    # ---------------- Right subplot: BEV occupancy ----------------
    # Color map for {-1,0,1}
    cmap = mcolors.ListedColormap([
        (0.15, 0.15, 0.15),  # -1 = gray
        (0.10, 0.70, 0.10),  #  0 = green
        (0.85, 0.10, 0.10)   #  1 = red
    ])

    im = axes[1].imshow(bev, cmap=cmap, origin="lower", vmin=-1, vmax=1)
    im.axes.set_aspect('equal')

    # im = axes[1].imshow(bev, origin="lower", cmap=cmap, vmin=-1, vmax=1)
    cbar = fig.colorbar(im, ax=axes[1], ticks=[-1, 0, 1])
    cbar.set_label("Occupancy")

    # Major ticks every 1 m, minor every 0.5 m
    meters_x = np.arange(0, GRID_WIDTH_M + 1e-6, 1.0)
    meters_z = np.arange(0, GRID_DEPTH_M + 1e-6, 1.0)
    xticks = (meters_x / CELL_SIZE_M).astype(int)
    yticks = (meters_z / CELL_SIZE_M).astype(int)

    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels([f"{x - GRID_WIDTH_M/2:.0f}" for x in meters_x])  
    axes[1].set_yticks(yticks)
    axes[1].set_yticklabels([f"{z:.0f}" for z in meters_z])

    axes[1].set_xlabel("Lateral distance (m)  [left  ⟵  0  ⟶  right]")
    axes[1].set_ylabel("Forward distance (m)  (0 at ego)")

    axes[1].set_xticks(np.arange(0, NX, 1), minor=True)
    axes[1].set_yticks(np.arange(0, NZ, 1), minor=True)

    # major Grid lines
    axes[1].grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.7)

    # minor Grid lines
    axes[1].grid(True, which='minor', linestyle='-', linewidth=0.4, alpha=0.4)

    axes[1].set_title("BEV Occupancy (-1 unknown, 0 road/free, 1 obstacle on road)")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[INFO] Saved BEV frame to: {save_path}")
    else:
        plt.show()