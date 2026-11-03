from itertools import combinations
import numpy as np


class RiskEngine:
    def __init__(self, config, seed=17):
        self.params = config["risk"]
        self.acceleration_std = config["tracking"]["acceleration_std"]
        self.rng = np.random.default_rng(seed)
        self.states = {}
        self.next_event = 1
        self.assessments = []

    def probability(self, a, b):
        p = self.params
        times = np.arange(0, p["horizon"] + p["step"] / 2, p["step"])
        mean, cov = a.x - b.x, a.p + b.p
        samples = self.rng.multivariate_normal(mean, cov, size=p["samples"])
        # A sampled velocity and acceleration persist across the entire trajectory.
        accel = self.rng.normal(0, self.acceleration_std * np.sqrt(2), (p["samples"], 2))
        paths = samples[:, None, :2] + samples[:, None, 2:] * times[None, :, None]
        paths += 0.5 * accel[:, None, :] * times[None, :, None]**2
        safe_distance = a.radius + b.radius + p["safe_margin"]
        prob = float(np.mean(np.linalg.norm(paths, axis=2).min(axis=1) < safe_distance))
        v = mean[2:]
        tca = float(np.clip(-mean[:2] @ v / max(v @ v, 1e-9), 0, p["horizon"]))
        minimum = float(np.linalg.norm(mean[:2] + tca * v))
        return prob, tca, minimum, safe_distance

    def update(self, timestamp, tracks):
        events, active_keys = [], set()
        self.assessments = []
        eligible = [t for t in tracks if t.hits >= 2 and timestamp - t.last_seen <= self.params["max_prediction_age"]]
        for a, b in combinations(eligible, 2):
            if "person" not in (a.category, b.category) or a.category == b.category:
                continue
            key = tuple(sorted((a.id, b.id)))
            active_keys.add(key)
            prob, tca, minimum, safe = self.probability(a, b)
            # Record below-threshold pairs too; calibration on alarms alone is biased.
            self.assessments.append({"timestamp": timestamp, "pair": list(key), "probability": prob,
                                     "horizon": self.params["horizon"], "safe_distance": safe,
                                     "tca": tca, "minimum_distance": minimum})
            level = sum(prob >= threshold for threshold in self.params["thresholds"])
            state = self.states.setdefault(key, {"level": 0, "pending": 0, "count": 0, "clear": 0, "last_end": -1e9, "event_id": None})
            transition = None
            if level > state["level"]:
                state["count"] = state["count"] + 1 if state["pending"] == level else 1
                state["pending"] = level
                state["clear"] = 0
                if state["count"] >= self.params["confirm_frames"] and timestamp - state["last_end"] >= self.params["cooldown"]:
                    transition = "START" if state["level"] == 0 else "ESCALATE"
                    if transition == "START":
                        state["event_id"] = self.next_event
                        self.next_event += 1
                    state["level"], state["count"] = level, 0
            elif level < state["level"] and prob < self.params["thresholds"][state["level"] - 1] - 0.08:
                state["clear"] += 1
                state["count"] = 0
                if state["clear"] >= self.params["clear_frames"]:
                    state["level"], state["clear"] = level, 0
                    transition = "END" if level == 0 else "DEESCALATE"
                    if level == 0:
                        state["last_end"] = timestamp
            else:
                state["clear"], state["count"] = 0, 0
            if state["level"] or transition:
                events.append({"event_id": state["event_id"], "pair": list(key), "timestamp": timestamp,
                               "level": state["level"], "probability": prob, "tca": tca,
                               "minimum_distance": minimum, "safe_distance": safe,
                               "transition": transition or "UPDATE",
                               "reason": "短时轨迹可能进入安全间距",
                               "predicted": a.status != "VISIBLE" or b.status != "VISIBLE"})
        for key in list(self.states):
            if key not in active_keys:
                state = self.states.pop(key)
                if state["level"]:
                    events.append({"event_id": state["event_id"], "pair": list(key), "timestamp": timestamp,
                                   "level": 0, "transition": "UNAVAILABLE", "reason": "轨迹失效，无法继续评估风险"})
        return events
