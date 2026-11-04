"""Reserved LiDAR observation boundary. Not connected to the monocular runtime."""
from .adapters import lidar_observations, observations_frame
from .schema import validate_frame


def lidar_frame(points, timestamp, source_frame_id, calibration, static_map=None):
    """Convert explicitly enabled calibrated Nx3 points into versioned observations.

    static_map is optional Nx3 geometry already in metric map coordinates.
    Enabling this adapter does not enable a fusion subscriber or alter Qt modes.
    """
    if calibration.get("schema_version") != 1 or calibration.get("frame_id") != "map":
        raise ValueError("LiDAR extension requires schema 1 and metric map frame")
    if calibration.get("enabled") is not True:
        raise ValueError("LiDAR extension is disabled")
    if calibration.get("example_only", True):
        raise ValueError("replace example LiDAR calibration with measured extrinsics")
    lidar = calibration["lidar"]
    if source_frame_id != lidar["input_frame_id"]:
        raise ValueError("LiDAR source frame differs from calibration")
    observations = lidar_observations(points, timestamp, lidar["world_from_lidar"], static_map,
                                     lidar["background_distance"], lidar["cluster_distance"],
                                     lidar["min_points"], lidar["ground_z"])
    frame = observations_frame(timestamp, observations, ["lidar"])
    frame["provenance"] = "lidar_extension_observations"
    validate_frame(frame)
    return frame
