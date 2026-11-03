from dataclasses import dataclass, field
import math
import numpy as np

CLASSES = {"person", "forklift_or_cart", "AGV", "unknown_dynamic_object"}
SENSORS = {"camera", "lidar"}


def covariance(value, size):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (size, size) or not np.isfinite(matrix).all():
        raise ValueError(f"covariance must be finite {size}x{size}")
    if not np.allclose(matrix, matrix.T) or np.linalg.eigvalsh(matrix).min() <= 0:
        raise ValueError("covariance must be symmetric positive definite")
    return matrix


@dataclass
class Observation:
    timestamp: float
    sensor: str
    position: list
    category: str = "unknown_dynamic_object"
    confidence: float = 1.0
    covariance: list = field(default_factory=lambda: [[0.09, 0], [0, 0.09]])
    quality: float = 1.0
    frame_id: str = "map"
    observation_id: str = ""
    radius: float = 0.35

    def __post_init__(self):
        if self.sensor not in SENSORS or self.category not in CLASSES:
            raise ValueError("unsupported sensor or industrial category")
        if not math.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("timestamp must be finite and nonnegative")
        if np.asarray(self.position).shape != (2,) or not np.isfinite(self.position).all():
            raise ValueError("position must contain finite x/y metres")
        covariance(self.covariance, 2)
        if not (0 < self.quality <= 1 and 0 <= self.confidence <= 1):
            raise ValueError("quality must be (0,1]; confidence must be [0,1]")
        if not math.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("radius must be positive")

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


def validate_frame(frame, frame_id="map"):
    if not isinstance(frame, dict):
        raise ValueError("frame must be a JSON object")
    if frame.get("schema_version") != 1 or frame.get("frame_id") != frame_id:
        raise ValueError("schema version or frame_id mismatch; transform observations first")
    timestamp = frame["timestamp"]
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("invalid frame timestamp")
    sensors = frame.get("sensors", [])
    if len(set(sensors)) != len(sensors) or not set(sensors) <= SENSORS:
        raise ValueError("invalid sensor heartbeats")
    observations = [Observation.from_dict(x) for x in frame.get("observations", [])]
    ids = set()
    for obs in observations:
        if obs.frame_id != frame_id or obs.timestamp > timestamp + 1e-6:
            raise ValueError("observation is in wrong frame or in the future")
        if obs.sensor not in sensors:
            raise ValueError("observation requires a sensor heartbeat")
        if obs.observation_id:
            key = (obs.sensor, obs.observation_id)
            if key in ids:
                raise ValueError("duplicate observation_id")
            ids.add(key)
    return observations
