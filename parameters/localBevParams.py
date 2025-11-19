import numpy as np

# ---- BEV parameters (tune as needed) ----
CELL_SIZE_M   = 0.05   # meters per cell (e.g., 0.10 = 10 cm)
GRID_WIDTH_M  = 30.0   # total width (left <-> right), meters
GRID_DEPTH_M  = 25.0   # forward range, meters

NX = int(np.round(GRID_WIDTH_M  / CELL_SIZE_M))   # columns (x)
NZ = int(np.round(GRID_DEPTH_M  / CELL_SIZE_M))   # rows (z)