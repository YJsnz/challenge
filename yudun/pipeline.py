import copy
import json
from pathlib import Path
from time import perf_counter
from .schema import validate_frame
from .tracking import Tracker
from .risk import RiskEngine


def load_config(path=None):
    path = Path(path) if path else Path(__file__).parent / "configs/warehouse.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("frame_id") != "map":
        raise ValueError("only schema 1 in metric map frame is supported")
    bounds = config["bounds"]
    if len(bounds) != 4 or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        raise ValueError("invalid scene bounds")
    p = config["risk"]
    if p["samples"] < 16 or p["step"] <= 0 or p["horizon"] <= 0 or p["horizon"] / p["step"] > 1000:
        raise ValueError("invalid prediction sampling configuration")
    if len(p["thresholds"]) != 3 or not 0 < p["thresholds"][0] < p["thresholds"][1] < p["thresholds"][2] <= 1:
        raise ValueError("risk thresholds must be ordered within (0,1]")
    if config.get("map_scene"):
        config["map_scene"] = str((path.parent / config["map_scene"]).resolve())
    return config


class Pipeline:
    def __init__(self, config=None, mode="monocular", seed=17):
        self.config = copy.deepcopy(config or load_config())
        self.mode = mode
        self.map_scene = None
        if self.config.get("map_scene"):
            from .map3d.scene import MapScene
            self.map_scene = MapScene(self.config["map_scene"], load_points=False)
            if not self.map_scene.registered:
                raise ValueError("tracking requires a registered 3D map")
        if mode in ("monocular", "monocular_basic") and set(self.config["sensor_positions"]) != {"camera"}:
            raise ValueError("monocular modes require exactly one configured camera")
        self.tracker = Tracker(self.config, mode)
        self.risk = RiskEngine(self.config, seed)
        self.last_heartbeat = {}
        self.origin = None

    def process(self, frame):
        start = perf_counter()
        observations = validate_frame(frame, self.config["frame_id"])
        if self.mode in ("monocular", "monocular_basic") and (set(frame.get("sensors", [])) - {"camera"} or any(o.sensor != "camera" for o in observations)):
            raise ValueError("monocular input cannot contain LiDAR or another sensor")
        if frame.get("reference_only"):
            raise ValueError("reference-only labels are not sensor observations; run model inference first")
        bound_scene = frame.get("map_scene_sha256")
        if self.map_scene and bound_scene and bound_scene != self.map_scene.sha256:
            raise ValueError("observations belong to a different map scene")
        timestamp = frame["timestamp"]
        # Validate ordering before mutating health or tracking state.
        if self.tracker.timestamp is not None and timestamp <= self.tracker.timestamp:
            raise ValueError("non-increasing frame timestamp")
        if self.origin is None:
            self.origin = timestamp
        for sensor in frame.get("sensors", []):
            self.last_heartbeat[sensor] = timestamp
        health = {sensor: ("ONLINE" if sensor in self.last_heartbeat and timestamp - self.last_heartbeat[sensor] <= self.config["health_timeout"] else "OFFLINE")
                  for sensor in self.config["sensor_positions"]}
        tracks = self.tracker.update(timestamp, observations)
        events = self.risk.update(timestamp, tracks)
        result = {"schema_version": 1, "frame_id": self.config["frame_id"], "timestamp": timestamp,
                "mode": self.mode, "provenance": frame.get("provenance", "external_unverified"),
                "health": health, "system_status": "NORMAL" if all(s == "ONLINE" for s in health.values()) else "DEGRADED",
                "tracks": [t.export(timestamp, self.config["tracking"]["acceleration_std"]) for t in tracks],
                "events": events, "observations": frame.get("observations", []),
                "visual_detections": frame.get("visual_detections", []),
                "risk_assessments": self.risk.assessments,
                "processing_ms": 0.0}
        for field in ("map_scene_sha256", "example_calibration"):
            if field in frame:
                result[field] = frame[field]
        if self.map_scene:
            result = self.map_scene.project_result(result)
        result["processing_ms"] = (perf_counter() - start) * 1000
        return result


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on line {number}: {exc}") from exc


def replay(frames, config=None, mode="monocular", seed=17):
    pipeline = Pipeline(config, mode, seed)
    return [pipeline.process(frame) for frame in frames]
