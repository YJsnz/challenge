from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linear_sum_assignment
from .geometry import is_visible

H = np.eye(4)[:2]
MODES = ("visual", "lidar", "fixed_fusion", "quality_fusion", "full")


def motion(dt, acceleration_std):
    f = np.eye(4)
    f[0, 2] = f[1, 3] = dt
    g = np.array([[dt*dt/2, 0], [0, dt*dt/2], [dt, 0], [0, dt]])
    return f, g @ g.T * acceleration_std**2


@dataclass
class Track:
    id: int
    category: str
    x: np.ndarray
    p: np.ndarray
    last_seen: float
    radius: float
    status: str = "VISIBLE"
    hits: int = 1
    sources: set = field(default_factory=set)

    def export(self, timestamp, acceleration_std):
        future = []
        for dt in (0.5, 1.0, 2.0, 3.0):
            f, q = motion(dt, acceleration_std)
            future.append({"dt": dt, "mean": (f @ self.x)[:2].tolist(),
                           "covariance": (f @ self.p @ f.T + q)[:2, :2].tolist()})
        return {"id": self.id, "category": self.category, "state": self.x.tolist(),
                "covariance": self.p.tolist(), "status": self.status,
                "last_seen": self.last_seen, "age": timestamp - self.last_seen,
                "sources": sorted(self.sources), "hits": self.hits,
                "radius": self.radius, "predictions": future}


class Tracker:
    def __init__(self, config, mode="full"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.config, self.mode = config, mode
        self.params = config["tracking"]
        self.tracks, self.next_id, self.timestamp = [], 1, None
        self.seen_observations = {}

    def measurement_covariance(self, obs, age):
        if self.mode == "fixed_fusion":
            r = np.eye(2) * (0.16 if obs.sensor == "camera" else 0.04)
        else:
            r = np.array(obs.covariance, float)
            if self.mode in ("quality_fusion", "full"):
                r /= max(0.05, obs.quality * obs.confidence)
        # Older observations carry additional position uncertainty; no fake retimestamping.
        return r + np.eye(2) * (self.params["max_speed"] * age)**2

    def update(self, timestamp, observations):
        if self.timestamp is not None and timestamp <= self.timestamp:
            raise ValueError("frames must be strictly increasing; reorder before replay")
        dt = 0 if self.timestamp is None else timestamp - self.timestamp
        self.timestamp = timestamp
        f, q = motion(dt, self.params["acceleration_std"])
        for track in self.tracks:
            track.x, track.p = f @ track.x, f @ track.p @ f.T + q
            track.sources.clear()
        # Expire before association, so a new observation cannot resurrect an expired ID.
        self.tracks = [t for t in self.tracks if timestamp - t.last_seen <= self.timeout(t)
                       and np.linalg.eigvalsh(t.p[:2, :2]).max() <= self.params["max_position_variance"]]
        self.seen_observations = {k: v for k, v in self.seen_observations.items()
                                  if timestamp - v <= self.params["max_observation_age"] + 1}
        accepted = []
        for obs in observations:
            age = timestamp - obs.timestamp
            if age > self.params["max_observation_age"] or obs.confidence <= 0:
                continue
            key = (obs.sensor, obs.observation_id)
            if obs.observation_id and key in self.seen_observations:
                continue
            if obs.observation_id:
                self.seen_observations[key] = timestamp
            accepted.append(obs)
        for sensor in ("camera", "lidar"):
            if self.mode == "visual" and sensor != "camera" or self.mode == "lidar" and sensor != "lidar":
                continue
            obs_list = [o for o in accepted if o.sensor == sensor]
            costs = np.full((len(self.tracks), len(obs_list)), 1e9)
            for i, track in enumerate(self.tracks):
                for j, obs in enumerate(obs_list):
                    if track.category != obs.category and "unknown_dynamic_object" not in (track.category, obs.category):
                        continue
                    age = timestamp - obs.timestamp
                    # Compare the measurement at acquisition time to the back-projected state.
                    h = H.copy()
                    h[0, 2] = h[1, 3] = -age
                    delta = np.array(obs.position) - h @ track.x
                    s = h @ track.p @ h.T + self.measurement_covariance(obs, age)
                    distance = float(delta @ np.linalg.solve(s, delta))
                    if distance <= self.params["gate_chi2"]:
                        costs[i, j] = distance
            matched = set()
            if costs.size:
                rows, cols = linear_sum_assignment(costs)
                for i, j in zip(rows, cols):
                    if costs[i, j] >= 1e9:
                        continue
                    track, obs = self.tracks[i], obs_list[j]
                    age = timestamp - obs.timestamp
                    h = H.copy()
                    h[0, 2] = h[1, 3] = -age
                    r = self.measurement_covariance(obs, age)
                    s = h @ track.p @ h.T + r
                    k = np.linalg.solve(s, h @ track.p).T
                    track.x += k @ (np.array(obs.position) - h @ track.x)
                    a = np.eye(4) - k @ h
                    track.p = a @ track.p @ a.T + k @ r @ k.T
                    track.p = (track.p + track.p.T) / 2
                    track.last_seen = max(track.last_seen, obs.timestamp)
                    track.sources.add(sensor)
                    track.status = "VISIBLE"
                    track.hits += 1
                    if obs.category != "unknown_dynamic_object":
                        track.category, track.radius = obs.category, obs.radius
                    matched.add(j)
            for j, obs in enumerate(obs_list):
                if j not in matched:
                    p = np.diag([1., 1., 4., 4.])
                    p[:2, :2] = self.measurement_covariance(obs, timestamp - obs.timestamp)
                    self.tracks.append(Track(self.next_id, obs.category, np.array([*obs.position, 0., 0.]),
                                             p, obs.timestamp, obs.radius, sources={sensor}))
                    self.next_id += 1
        for track in self.tracks:
            if not track.sources:
                track.status = "POSSIBLY_OCCLUDED" if self.occluded(track) else "LOST"
        return self.tracks

    def occluded(self, track):
        return self.mode == "full" and all(
            not is_visible(track.x[:2], position, self.config["obstacles"])
            for position in self.config["sensor_positions"].values())

    def timeout(self, track):
        return self.params["occluded_timeout"] if self.occluded(track) else self.params["visible_timeout"]
