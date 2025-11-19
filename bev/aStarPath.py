import numpy as np
import heapq
from parameters.localBevParams import CELL_SIZE_M

# =========================
# === A*-LIKE PATH PLANNER ON BEV (SPEED-LIMITED) ===
# =========================

def heuristic(a, b):
    """
    Calculate the Euclidean distance between two points.
    """
    return np.linalg.norm(np.array(a) - np.array(b))

def reconstruct_path(came_from, current):
    """
    Reconstructs the path from the goal back to the start node.
    """
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]  # Reverse to get path from start to goal

def a_star_speed_limited(bev, start, valid_corridor_rows, max_forward_m):
    """
    A* path planner on a Bird's-Eye View (BEV) grid, constrained to a pre-calculated corridor.
    This is a corrected, more efficient implementation of A*.

    Args:
        bev (np.array): 2D grid with values {0: free, 1: obstacle, 2: corridor}.
        start (tuple): The starting (z, x) grid index. Z is the forward axis.
        cell_size (float): The size of each grid cell in meters.
        max_forward_m (float): The maximum forward distance to plan in meters.

    Returns:
        list: A list of (z, x) waypoints from start to goal, or an empty list if no path is found.
    """
    print("\nRunning A*-like path planner on BEV (speed-limited)...")
    
    NZ, NX = bev.shape
    max_forward_cells = int(max_forward_m / CELL_SIZE_M)
    goal_z = min(start[0] + max_forward_cells, valid_corridor_rows.max())

    corridor_cells_at_goal_row = np.where(bev[goal_z, :] == 2)[0]
    if len(corridor_cells_at_goal_row) == 0:
        print(f"No safe corridor found at target depth z={goal_z} for A* goal setting.")
        return []

    goal_x = int(np.mean(corridor_cells_at_goal_row))
    goal = (goal_z, goal_x)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    # The open list stores: (f_cost, node)
    open_list = []
    heapq.heappush(open_list, (heuristic(start, goal), start))

    # came_from[n] is the node immediately preceding it on the cheapest path.
    came_from = {}

    # g_score[n] is the cost of the cheapest path from start to n.
    g_score = {start: 0}

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            return reconstruct_path(came_from, current)

        for dz, dx in directions:
            neighbor = (current[0] + dz, current[1] + dx)

            if not (0 <= neighbor[0] < NZ and 0 <= neighbor[1] < NX):
                continue  # Skip if out of bounds

            # This is the key change: only explore nodes within the corridor (value 2).
            if bev[neighbor] != 2:
                continue

            # tentative_g_score is the distance from start to the neighbor through current
            tentative_g_score = g_score[current] + np.hypot(dz, dx)

            if tentative_g_score < g_score.get(neighbor, float('inf')):
                # This path to neighbor is better than any previous one. Record it!
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score, neighbor))

    print("A* could not find a path to the goal within the corridor.")
    return []