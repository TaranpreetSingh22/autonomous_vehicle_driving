import numpy as np
from parameters.localBevParams import CELL_SIZE_M as cell_size
from parameters.depthThresholds import max_depth_thresh

def find_drivable_corridor(bev, vehicle_width):
    """
    This function finds a drivable corridor by shifting the corridor to the right or left
    if obstacles are found within the corridor.
    
    Args:
    bev: The BEV grid (NZ, NX) with obstacles (1), free space (0), and unknown (-1).
    vehicle_width: Width of the vehicle in meters.
    max_depth: Max depth (number of rows) to check in the forward direction.
    
    Returns:
    path: List of valid corridor positions (if any) or failure status.
    """
    # =========================
    # === DRIVABLE CORRIDOR WITH SHIFTING (if obstacles) ===
    # =========================
    print("\nFinding drivable corridor with shifting (if obstacles)...")
    # Calculate the vehicle's corridor width in cells
    vehicle_width_cells = int(vehicle_width / cell_size)
    
    # Start from the bottom (near the vehicle) and go upwards
    h, w = bev.shape
    corridor_center = w // 2  # Assume the vehicle is at the center horizontally
    
    # Initial corridor positions (vehicle centered)
    corridor_left = corridor_center - vehicle_width_cells // 2
    corridor_right = corridor_center + vehicle_width_cells // 2
    
    # Store the valid path positions (if found)
    valid_path = []
    
    # # Convert depth thresholds into BEV cell indices
    # start_depth_cell = int(np.floor(min_depth_thresh / cell_size))
    # end_depth_cell   = int(np.ceil(max_depth_thresh / cell_size))

    # # Clamp to BEV size
    NZ, NX = bev.shape
    # start_depth_cell = max(0, start_depth_cell)
    # end_depth_cell   = min(NZ, end_depth_cell)  # Start from the first valid road point
    valid_path = []

    # --- Adaptive corridor start (like BEV logic) ---
    road_rows = np.where(np.any(bev == 0, axis=1))[0]  # rows where road exists
    if road_rows.size > 0:
        start_depth_cell = road_rows.min()  # first road row
    else:
        start_depth_cell = 0                # fallback if no road

    # End at max BEV depth
    end_depth_cell   = int(np.ceil(max_depth_thresh / cell_size))
    end_depth_cell   = min(NZ, end_depth_cell)

    for z in range(start_depth_cell, end_depth_cell):
        # Ensure corridor indices are within bounds
        if 0 <= corridor_left < NX and 0 <= corridor_right < NX:
            if np.all(bev[z, corridor_left:corridor_right+1] == 1):  # Obstacle in corridor
                # Try shifting right
                for shift in range(1, NX - corridor_right):
                    new_corridor_left = corridor_left + shift
                    new_corridor_right = corridor_right + shift
                    if new_corridor_right < NX and np.all(bev[z, new_corridor_left:new_corridor_right+1] == 0):
                        valid_path.append((new_corridor_left, new_corridor_right, z))
                        # valid_path.append((new_corridor_right, z))
                        corridor_left, corridor_right = new_corridor_left, new_corridor_right
                        break
                else:
                    # Try shifting left
                    for shift in range(1, corridor_left + 1):
                        new_corridor_left = corridor_left - shift
                        new_corridor_right = corridor_right - shift
                        if new_corridor_left >= 0 and np.all(bev[z, new_corridor_left:new_corridor_right+1] == 0):
                            valid_path.append((new_corridor_left, new_corridor_right, z))
                            # valid_path.append((new_corridor_right, z))
                            corridor_left, corridor_right = new_corridor_left, new_corridor_right
                            break
            else:
                # No obstacle in current corridor
                if np.any(bev[z, corridor_left:corridor_right+1] == 0):
                    valid_path.append((corridor_left, corridor_right, z))
                    # valid_path.append((corridor_right, z))
                else:
                    break
    
    if not valid_path:
        return "Failure", []
    
    return "Success", valid_path