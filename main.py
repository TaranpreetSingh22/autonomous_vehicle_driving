import os
import numpy as np
from groundTruth import gtDisparity
from groundTruth import gtDepth
from sgbm.disparity import disparity
from sgbm.depth.depth import get_depth_from_disparity
from computeMetrics import metrics
from roadPlane import roadPlane
from bev import buildBev, corridor, aStarPath, globalStitching
from parameters import intrinsics, vehicleParams, globalBevParams
from plots import plotMetrics, plotGlobalStitch, plotAStarPath, plotBev, plotRoadPlane, plotCorridorBev

def main():
    # Folder containing video frames
    video_folder = r"C:/Users/Taranpreetsingh R/Desktop/graph theory/agv/code/sgbm/agv_impl_copy/video"
    print("Exists?", os.path.exists(video_folder))
    print("image_02/data exists?", os.path.exists(os.path.join(video_folder, "image_02", "data")))
    left_folder = os.path.join(video_folder, "image_02", "data")
    right_folder = os.path.join(video_folder, "image_03", "data")

    # List and sort frame filenames
    left_frames = sorted(os.listdir(left_folder))
    right_frames = sorted(os.listdir(right_folder))

    # Ensure equal number of frames
    assert len(left_frames) == len(right_frames), "Left/Right frame counts mismatch!"

    # Initialize global projector once
    projector = globalStitching.GlobalPathProjector(
        map_size_m=(globalBevParams.GRID_WIDTH_M, globalBevParams.GRID_DEPTH_M),
        resolution=globalBevParams.CELL_SIZE_M
    )

    for i, (lf, rf) in enumerate(zip(left_frames, right_frames)):
        print(f"\n=== Processing frame {i}: {lf} / {rf} ===")

        left_image_path = os.path.join(left_folder, lf)
        right_image_path = os.path.join(right_folder, rf)

        # 1. Compute disparity and depth
        computed_disp = disparity.get_disparity_map(left_image_path, right_image_path)
        computed_depth = get_depth_from_disparity(computed_disp)

        # 2. Road plane detection
        road_mask = roadPlane.detect_road_plane(computed_depth)

        # 3. BEV
        K, Kinv = intrinsics.get_K_and_Kinv(computed_depth)
        bev_map = buildBev.build_bev_grid(computed_depth, road_mask, Kinv)

        # 4. Corridor + A* planning
        vehicle_width = vehicleParams.vehicle_width
        status, path = corridor.find_drivable_corridor(bev_map, vehicle_width)
        path_bev = plotCorridorBev.plotCorridorBev(bev_map, path)

        valid_corridor_rows = np.where(np.any(path_bev == 2, axis=1))[0]
        if valid_corridor_rows.size > 0:
            start_z = valid_corridor_rows.min()
            corridor_cells = np.where(path_bev[start_z, :] == 2)[0]
            start_x = int(np.mean(corridor_cells))
            start_node = (start_z, start_x)

            vehicle_speed_mps = 3.0
            planning_time_s = 1
            max_forward_m = vehicle_speed_mps * planning_time_s

            a_path = aStarPath.a_star_speed_limited(path_bev, start_node, valid_corridor_rows, max_forward_m)
            astar_bev = plotAStarPath.plotAStarPath(bev_map, a_path, max_forward_m)
        else:
            print("A* Failure: no corridor found.")
            continue

        # 5. Global map update
        projector.project_bev(astar_bev)
        projector.update_pose(speed_mps=3.0, dt=1.0)

        # Optional: visualize every N frames
        if i % 10 == 0:
            plotGlobalStitch.plotGlobalStitch(projector)

    # Final stitched map after all frames
    plotGlobalStitch.plotGlobalStitch(projector)

if __name__ == "__main__":
    main()