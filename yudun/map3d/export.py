import csv


class MapTrackWriter:
    def __init__(self, stream):
        self.writer = csv.writer(stream)
        self.writer.writerow(["timestamp", "frame_id", "scene_sha256", "example_only", "track_id",
                              "category", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps", "status"])

    def write(self, frame):
        mapped = frame.get("map_3d")
        if mapped:
            self.writer.writerows(
                [frame["timestamp"], mapped["frame_id"], mapped["scene_sha256"], mapped["example_only"],
                 track["id"], track["category"], *track["position_m"], *track["velocity_mps"], track["status"]]
                for track in mapped["tracks"]
            )
