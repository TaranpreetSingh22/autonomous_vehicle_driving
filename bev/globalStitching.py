import numpy as np

class GlobalPathProjector:
    # =========================
    # === GLOBAL STITCHING ===
    # =========================
    def __init__(self, map_size_m=(400, 400), resolution=0.5):
        print("\nStitching local BEV paths into global map...")
        self.resolution = resolution
        self.map_size_px = (
            int(map_size_m[0] / resolution),
            int(map_size_m[1] / resolution)
        )
        self.global_map = np.full(self.map_size_px, -1, dtype=np.int8)  # -1 = unknown
        self.pose = np.array([
            self.map_size_px[1] // 2,
            0,
            0.0
        ])

    def update_pose(self, speed_mps, dt, yaw_rate_rad=0.0):
        dx = speed_mps * dt * np.cos(self.pose[2])
        dy = speed_mps * dt * np.sin(self.pose[2])
        self.pose[0] += dx / self.resolution
        self.pose[1] += dy / self.resolution
        self.pose[2] += yaw_rate_rad * dt

    def project_bev(self, local_bev):
        """
        Project full BEV (2D numpy array) into the global map.
        Assumes BEV is aligned forward with ego, origin at bottom center.
        """
        NZ, NX = local_bev.shape
        theta = self.pose[2]
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        
        for z in range(NZ):
            for x in range(NX):
                val = local_bev[z, x]
                if val == -1:
                    continue  # skip unknown
                y_m = z * self.resolution
                x_m = (x - NX // 2) * self.resolution
                local_pt = np.array([x_m, y_m])
                global_pt = R @ local_pt
                global_x = int(self.pose[0] + global_pt[0] / self.resolution)
                global_y = int(self.pose[1] + global_pt[1] / self.resolution)
                if 0 <= global_x < self.map_size_px[1] and 0 <= global_y < self.map_size_px[0]:
                    self.global_map[global_y, global_x] = val

    def get_map(self):
        return self.global_map