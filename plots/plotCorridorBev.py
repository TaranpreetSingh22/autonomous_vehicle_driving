import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from parameters.localBevParams import CELL_SIZE_M, GRID_WIDTH_M, GRID_DEPTH_M, NX, NZ

def get_path(bev, path):
    """
    Overlay the A* path on the BEV grid.
    Args:
        bev (np.ndarray): 2D array representing the BEV occupancy grid.
                          Values: -1 (unknown), 0 (free), 1 (obstacle)
        a_path (list of tuples): Each tuple is (z, x) representing the path cell coordinates.
    Returns:
        np.ndarray: BEV grid with path overlayed (path cells marked as 2).
    """
    path_bev = np.copy(bev)
    if path:
        for (left, right, z) in path:
            path_bev[z, left:right+1] = 2
    
    return path_bev

def plotCorridorBev(bev, path=None, save_path=False):
    """
    Plots the BEV occupancy grid and overlays the drivable path if provided.
    Args:
        bev (np.ndarray): 2D array representing the BEV occupancy grid.
                          Values: -1 (unknown), 0 (free), 1 (obstacle)
        path (list of tuples): Each tuple is (left, right, z) representing
                               the drivable corridor at row z.
    """
    # ---- Colormap for {-1,0,1,2} ----
    # -1 = gray (unknown), 0 = green (free), 1 = red (obstacle), 2 = blue (path)
    cmap = mcolors.ListedColormap([
        (0.15, 0.15, 0.15),  # -1 = gray
        (0.10, 0.70, 0.10),  #  0 = green
        (0.85, 0.10, 0.10),  #  1 = red
        (0.10, 0.30, 0.85)   #  2 = blue (path)
    ])

    # ---- Create figure with two subplots ----
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    # ------------------- Plot 1: Original BEV -------------------
    im0 = axes[0].imshow(bev, cmap=cmap, origin="lower", vmin=-1, vmax=2)
    axes[0].set_aspect('equal') # IMPORTANT: Keeps cells square

    cbar0 = fig.colorbar(im0, ax=axes[0], ticks=[-1, 0, 1])
    cbar0.set_label("Occupancy")

    meters_x = np.arange(0, GRID_WIDTH_M + 1e-6, 1.0)
    meters_z = np.arange(0, GRID_DEPTH_M + 1e-6, 1.0)
    xticks = (meters_x / CELL_SIZE_M).astype(int)
    yticks = (meters_z / CELL_SIZE_M).astype(int)

    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels([f"{x - GRID_WIDTH_M/2:.0f}" for x in meters_x])
    axes[0].set_yticks(yticks)
    axes[0].set_yticklabels([f"{z:.0f}" for z in meters_z])

    axes[0].set_xlabel("Lateral distance (m)  [left  ⟵  0  ⟶  right]")
    axes[0].set_ylabel("Forward distance (m)  (0 at ego)")
    axes[0].set_title("Original BEV Occupancy")

    # Define ticks for the minor grid (every cell)
    axes[0].set_xticks(np.arange(0, NX, 1), minor=True)
    axes[0].set_yticks(np.arange(0, NZ, 1), minor=True)

    # Major grid (every 1m): Solid, thicker, and darker
    axes[0].grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.7)

    # Minor grid (every cell): Thinner, lighter, but still visible
    axes[0].grid(True, which='minor', linestyle='-', linewidth=0.4, alpha=0.4)


    # ------------------- Plot 2: Path Visualization -------------------
    path_bev = get_path(bev, path)

    im1 = axes[1].imshow(path_bev, cmap=cmap, origin="lower", vmin=-1, vmax=2)
    axes[1].set_aspect('equal') # IMPORTANT: Keeps cells square

    cbar1 = fig.colorbar(im1, ax=axes[1], ticks=[-1, 0, 1, 2])
    cbar1.set_label("Occupancy + Path")

    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels([f"{x - GRID_WIDTH_M/2:.0f}" for x in meters_x])
    axes[1].set_yticks(yticks)
    axes[1].set_yticklabels([f"{z:.0f}" for z in meters_z])

    axes[1].set_xlabel("Lateral distance (m)  [left  ⟵  0  ⟶  right]")
    axes[1].set_ylabel("Forward distance (m)  (0 at ego)")
    axes[1].set_title("Drivable Path with Shifting Corridor")

    # Define ticks for the minor grid (every cell)
    axes[1].set_xticks(np.arange(0, NX, 1), minor=True)
    axes[1].set_yticks(np.arange(0, NZ, 1), minor=True)

    # Major grid (every 1m): Solid, thicker, and darker
    axes[1].grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.7)

    # Minor grid (every cell): Thinner, lighter, but still visible
    axes[1].grid(True, which='minor', linestyle='-', linewidth=0.4, alpha=0.4)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[INFO] Saved BEV frame to: {save_path}")
        return path_bev
    else:
        plt.show()
        return path_bev