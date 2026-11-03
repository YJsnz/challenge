"""Explicitly synthetic fixtures, never evidence of hardware or industrial accuracy."""
from dataclasses import asdict
import numpy as np
from .schema import Observation
from .geometry import is_visible

SCENARIOS = ("crossing", "safe", "occluded", "stop", "turn", "sensor_dropout", "low_light", "jitter")


def generate(config, scenario="occluded", seed=7, duration=12.0, hz=10):
    if scenario not in SCENARIOS or duration <= 0 or hz <= 0:
        raise ValueError("invalid synthetic scenario, duration or Hz")
    rng = np.random.default_rng(seed)
    frames = []
    for index, timestamp in enumerate(np.arange(0, duration, 1 / hz)):
        t = float(timestamp)
        person = np.array([0.0, -6 + t])
        vehicle = np.array([-8 + 1.4*t, 0.0])
        if scenario == "safe":
            vehicle[0] = -12 + 1.0*t
        if scenario == "stop":
            vehicle[0] = -8 + 1.4*min(t, 3.0)
        if scenario == "turn" and t > 4:
            vehicle = np.array([-2.4, (t - 4)*1.4])
        truth = [{"id": "person-1", "position": person.tolist(), "category": "person", "radius": 0.35},
                 {"id": "cart-1", "position": vehicle.tolist(), "category": "forklift_or_cart", "radius": 0.6}]
        sensors = list(config["sensor_positions"])
        if scenario == "sensor_dropout" and "lidar" in sensors and 3 <= t < 7:
            sensors.remove("lidar")
        observations = []
        for sensor in sensors:
            for obj in truth:
                position = np.array(obj["position"])
                # Occluded fixture includes a brief shared blind zone to exercise lifecycle.
                blocked = scenario == "occluded" and obj["id"] == "cart-1" and 3.4 <= t < 4.6
                if blocked or not is_visible(position, config["sensor_positions"][sensor], config["obstacles"]):
                    continue
                quality = 0.9
                sigma = 0.25 if sensor == "camera" else 0.12
                if scenario == "low_light" and sensor == "camera" and 2 <= t < 8:
                    sigma, quality = 0.7, 0.2
                if rng.random() < (0.08 if sensor == "camera" else 0.03):
                    continue
                age = float(rng.uniform(0, 0.08)) if scenario == "jitter" and t > 0.1 else 0
                velocity = np.array([0, 1]) if obj["id"] == "person-1" else np.array([1.4, 0])
                measurement = position - age * velocity + rng.normal(0, sigma, 2)
                obs = Observation(t-age, sensor, measurement.tolist(), obj["category"] if sensor == "camera" else "unknown_dynamic_object",
                                  0.9, (np.eye(2) * sigma**2).tolist(), quality, observation_id=f"{index}:{obj['id']}", radius=obj["radius"])
                observations.append(asdict(obs))
        frames.append({"schema_version": 1, "timestamp": t, "frame_id": "map", "sensors": sensors,
                       "observations": observations, "ground_truth": truth,
                       "provenance": "synthetic", "scenario": scenario, "seed": seed})
    return frames
