import numpy as np
from parameters.localBevParams import CELL_SIZE_M, GRID_WIDTH_M, GRID_DEPTH_M

def build_bev_grid(depth_map, road_mask_img, Kinv):
    """
    depth_map: (H,W) meters
    road_mask_img: (H,W) uint8 where 1=road, 0=not-road
    Returns:
      bev: (NZ, NX) int8 with values {-1: unknown, 0: road/free, 1: obstacle}
    """
    print("\nBuilding grid-based BEV (metric) ...")
    H, W = depth_map.shape
    bev = -1 * np.ones((int(GRID_DEPTH_M/CELL_SIZE_M), int(GRID_WIDTH_M/CELL_SIZE_M)), dtype=np.int8)

    # valid depth
    valid = np.isfinite(depth_map) & (depth_map > 0)
    if not np.any(valid):
        return bev

    # pixel grid
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    uv1 = np.stack([uu[valid], vv[valid], np.ones(np.count_nonzero(valid), dtype=np.float32)], axis=0)

    # back-project
    Z = depth_map[valid].astype(np.float32)                         # (N,)
    XYZ = (Kinv @ uv1) * Z[None, :]                                 # (3,N)
    X = XYZ[0, :]                                                   # right  (+)
    Zf = XYZ[2, :]                                                  # forward(+)

    # keep points that fall inside BEV bounds
    half_w = GRID_WIDTH_M / 2.0
    in_bounds = (Zf > 0) & (Zf < GRID_DEPTH_M) & (np.abs(X) < half_w)
    if not np.any(in_bounds):
        return bev

    X = X[in_bounds]
    Zf = Zf[in_bounds]
    road_flags     = (road_mask_img[valid][in_bounds] == 1)  # 1=road -> free
    nonroad_flags  = (road_mask_img[valid][in_bounds] == 0)  # 0=not-road

    # to BEV indices
    ix = np.floor((X + half_w) / CELL_SIZE_M).astype(np.int32)   # [0 .. NX-1]
    iz = np.floor(Zf / CELL_SIZE_M).astype(np.int32)             # [0 .. NZ-1]

    # clamp (safety)
    ix = np.clip(ix, 0, bev.shape[1]-1)
    iz = np.clip(iz, 0, bev.shape[0]-1)

    # 1) mark road cells as free (0)
    bev[iz[road_flags], ix[road_flags]] = 0

    # 2) mark obstacles ONLY where inside road footprint (i.e., cells already free)
    #    Find candidate obstacle cells
    obs_iz = iz[nonroad_flags]
    obs_ix = ix[nonroad_flags]
    # mask to only overwrite where previously set to road/free (0)
    inside_road = (bev[obs_iz, obs_ix] == 0)
    bev[obs_iz[inside_road], obs_ix[inside_road]] = 1

    return bev