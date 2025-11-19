import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from parameters.localBevParams import CELL_SIZE_M, GRID_WIDTH_M, GRID_DEPTH_M, NX, NZ

def get_path(bev, a_path):
    """
    Overlay the A* path on the BEV grid.
    Args:
        bev (np.ndarray): 2D array representing the BEV occupancy grid.
                          Values: -1 (unknown), 0 (free), 1 (obstacle)
        a_path (list of tuples): Each tuple is (z, x) representing the path cell coordinates.
    Returns:
        np.ndarray: BEV grid with path overlayed (path cells marked as 2).
    """
    bev_path_overlay = np.copy(bev)
    for (z, x) in a_path:
        bev_path_overlay[z, max(0, x-1):min(NX, x+2)] = 2  # Mark path cells
    
    return bev_path_overlay

def plotAStarPath(bev, a_path, max_forward_m, save_path=False):
    """
    Plots the BEV occupancy grid and overlays the A*-like path.
    Args:
        bev (np.ndarray): 2D array representing the BEV occupancy grid.
                          Values: -1 (unknown), 0 (free), 1 (obstacle)
        a_path (list of tuples): Each tuple is (z, x) representing the path cell coordinates.
        max_forward_m (float): Maximum forward distance in meters for title display.
    """
    # Define custom color map
    cmap = mcolors.ListedColormap([
        (0.15, 0.15, 0.15),  # -1 = gray (unknown)
        (0.10, 0.70, 0.10),  #  0 = green (free)
        (0.85, 0.10, 0.10),  #  1 = red (obstacle)
        (0.10, 0.30, 0.85)   #  2 = blue (path)
    ])

     # Major ticks every 1 m, minor every 0.5 m
    meters_x = np.arange(0, GRID_WIDTH_M + 1e-6, 1.0)
    meters_z = np.arange(0, GRID_DEPTH_M + 1e-6, 1.0)
    xticks = (meters_x / CELL_SIZE_M).astype(int)
    yticks = (meters_z / CELL_SIZE_M).astype(int)

    # ------------------- Plot: A*-like Path Visualization -------------------
    astar_bev = get_path(bev, a_path)  # Copy so we don't overwrite original

    fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    axes = [axes]  # make it list-like for consistency

    im_astar = axes[0].imshow(astar_bev, cmap=cmap, origin="lower", vmin=-1, vmax=2)
    axes[0].set_aspect('equal')  # Keep cells square

    # Add colorbar with occupancy + A* path info
    cbar_astar = fig.colorbar(im_astar, ax=axes[0], ticks=[-1, 0, 1, 2])
    cbar_astar.set_label("Occupancy + A* Path")

    # X and Y ticks in meters
    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels([f"{x - GRID_WIDTH_M/2:.0f}" for x in meters_x])
    axes[0].set_yticks(yticks)
    axes[0].set_yticklabels([f"{z:.0f}" for z in meters_z])

    # Labels and title
    axes[0].set_xlabel("Lateral distance (m)  [left  ⟵  0  ⟶  right]")
    axes[0].set_ylabel("Forward distance (m)  (0 at ego)")
    axes[0].set_title(f"A*-like Path (Speed-limited to {max_forward_m:.1f} m)")

    # Minor ticks for every cell
    axes[0].set_xticks(np.arange(0, NX, 1), minor=True)
    axes[0].set_yticks(np.arange(0, NZ, 1), minor=True)

    # Major grid (1m) and minor grid (per cell)
    axes[0].grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.7)
    axes[0].grid(True, which='minor', linestyle='-', linewidth=0.4, alpha=0.4)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[INFO] Saved BEV frame to: {save_path}")
        return astar_bev
    else:
        plt.show()
        return astar_bev