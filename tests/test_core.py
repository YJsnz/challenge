from dataclasses import asdict
import numpy as np
import pytest
from yudun.schema import Observation, validate_frame
from yudun.geometry import ground_projection, segment_hits_box
from yudun.tracking import Tracker, Track, motion
from yudun.pipeline import Pipeline, load_config, replay
from yudun.risk import RiskEngine
from yudun.simulation import generate, SCENARIOS
from yudun.adapters import lidar_observations
from yudun.evaluation import evaluate


def observation(t, x=0., y=0., sensor="camera", category="person", quality=1., confidence=.9, oid=""):
    return Observation(t, sensor, [x, y], category, confidence=confidence, quality=quality, observation_id=oid)


def frame(t, observations=(), sensors=("camera", "lidar")):
    return {"schema_version": 1, "frame_id": "map", "timestamp": t,
            "sensors": list(sensors), "observations": [asdict(o) for o in observations]}


@pytest.mark.parametrize("change", [{"timestamp": float('nan')}, {"position": [float('inf'), 0]},
                                     {"covariance": [[1, 2], [2, 1]]}, {"quality": 0},
                                     {"category": "R1"}, {"radius": -1}])
def test_invalid_observations(change):
    data = asdict(observation(0))
    data.update(change)
    with pytest.raises(ValueError):
        Observation(**data)


def test_frame_contract():
    with pytest.raises(ValueError):
        validate_frame(frame(0, [observation(1)]))
    with pytest.raises(ValueError):
        validate_frame(frame(0, [observation(0)], sensors=[]))
    with pytest.raises(ValueError):
        validate_frame(frame(0, [observation(0, oid="same"), observation(0, oid="same")]))


def test_ground_projection_metric_and_behind_camera():
    t = np.diag([1., -1., -1., 1.])
    t[2, 3] = 3
    k = [[100, 0, 50], [0, 100, 50], [0, 0, 1]]
    np.testing.assert_allclose(ground_projection([50, 50], k, t), [0, 0])
    np.testing.assert_allclose(ground_projection([150, 50], k, t), [3, 0])
    t = np.eye(4)
    t[2, 3] = 3
    with pytest.raises(ValueError):
        ground_projection([50, 50], k, t)


def test_occlusion_parallel_tangent_rays():
    assert segment_hits_box([0, 0], [5, 0], [2, -1, 3, 1])
    assert not segment_hits_box([0, 2], [5, 2], [2, -1, 3, 1])
    assert segment_hits_box([0, 1], [5, 1], [2, -1, 3, 1])


def test_variable_dt_state_and_covariance():
    f, q = motion(.5, 1)
    np.testing.assert_allclose(f @ [1, 2, 4, -2], [3, 1, 4, -2])
    assert np.linalg.eigvalsh(q).min() > -1e-12
    f2, _ = motion(2, 1)
    assert (f2 @ [1, 2, 4, -2])[0] == 9


def test_single_id_for_two_independent_sensors_and_one_to_one():
    tracker = Tracker(load_config())
    tracks = tracker.update(0, [observation(0), observation(0, sensor="lidar", category="unknown_dynamic_object")])
    assert len(tracks) == 1 and tracks[0].sources == {"camera", "lidar"}
    tracker.update(.1, [observation(.1, .1), observation(.1, 4)])
    assert len(tracker.tracks) == 2


def test_quality_downweights_bad_measurements():
    good, bad = Tracker(load_config()), Tracker(load_config())
    for tracker in (good, bad):
        tracker.update(0, [observation(0)])
    good.update(.1, [observation(.1, .6, quality=1)])
    bad.update(.1, [observation(.1, .6, quality=.05)])
    assert abs(bad.tracks[0].x[0]) < abs(good.tracks[0].x[0])


def test_empty_frames_expire_and_do_not_resurrect():
    tracker = Tracker(load_config())
    tracker.update(0, [observation(0)])
    tracker.update(.2, [])
    assert tracker.tracks[0].status == "LOST"
    assert tracker.tracks[0].p[0, 0] > .09
    assert not tracker.update(1, [])
    assert tracker.update(1.1, [observation(1.1)])[0].id == 2


def test_occluded_id_survives_then_terminates():
    config = load_config()
    config["obstacles"] = [[-1, -2, 1, -1]]
    config["sensor_positions"] = {"camera": [0, -4], "lidar": [.1, -4]}
    tracker = Tracker(config)
    tracker.update(0, [observation(0)])
    tracker.update(1, [])
    assert tracker.tracks[0].status == "POSSIBLY_OCCLUDED"
    assert tracker.update(1.1, [observation(1.1)])[0].id == 1
    assert not tracker.update(4, [])


def test_time_order_stale_dedup_and_health():
    pipeline = Pipeline()
    pipeline.process(frame(0, [observation(0, oid="a")]))
    result = pipeline.process(frame(.1, [observation(0, oid="a")]))
    assert result["tracks"][0]["hits"] == 1
    with pytest.raises(ValueError):
        pipeline.process(frame(.05))
    result = pipeline.process(frame(1, sensors=[]))
    assert result["system_status"] == "DEGRADED"
    assert result["health"] == {"camera": "OFFLINE", "lidar": "OFFLINE"}
    assert result["tracks"] == []


def test_stale_detection_cannot_spawn_track():
    pipeline = Pipeline()
    result = pipeline.process(frame(2, [observation(0)]))
    assert result["tracks"] == []


def test_lidar_clusters_and_background_and_empty():
    rng = np.random.default_rng(1)
    a = rng.normal([1, 1, 1], .03, (20, 3))
    b = rng.normal([4, 1, 1], .03, (20, 3))
    observations = lidar_observations(np.vstack([a, b]), 0)
    assert len(observations) == 2
    assert len(lidar_observations(np.vstack([a, b]), 0, static_map=a)) == 1
    assert lidar_observations([], 0) == []
    assert lidar_observations([[float('nan'), 0, 1]], 0) == []


def risk_track(identity, x, category):
    return Track(identity, category, np.array(x, float), np.eye(4)*.00001, 0, .35, hits=3)


def test_risk_crossing_receding_and_lifecycle():
    config = load_config()
    config["tracking"]["acceleration_std"] = .001
    engine = RiskEngine(config)
    a = risk_track(1, [-2, 0, 1, 0], "person")
    b = risk_track(2, [0, -2, 0, 1], "forklift_or_cart")
    prob, tca, distance, _ = engine.probability(a, b)
    assert prob > .95 and abs(tca-2) < .01 and distance < .01
    assert engine.update(0, [a, b]) == []
    assert engine.update(.1, [a, b])[0]["transition"] == "START"
    assert engine.update(.2, [a, b])[0]["transition"] == "UPDATE"
    assert engine.update(.3, [])[0]["transition"] == "UNAVAILABLE"
    b.x = np.array([0, -4, 0, -1.])
    assert engine.probability(a, b)[0] < .05


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_all_scenarios_finite_and_deterministic(scenario):
    config = load_config()
    frames = generate(config, scenario, duration=3)
    results = replay(frames, config)
    second = replay(frames, config)
    assert len(results) == 30
    for a, b in zip(results, second):
        assert a["tracks"] == b["tracks"]
        assert a["events"] == b["events"]
        for track in a["tracks"]:
            assert np.isfinite(track["state"]).all()
            assert np.linalg.eigvalsh(track["covariance"]).min() > 0


def test_metrics_identity_switch_penalty():
    gt = [{"timestamp": i, "ground_truth": [{"id": "a", "position": [0, 0]}]} for i in range(4)]
    results = [{"timestamp": i, "processing_ms": 1, "events": [], "tracks": [{"id": i//2+1, "state": [0, 0, 0, 0], "predictions": []}]} for i in range(4)]
    metrics = evaluate(gt, results)
    assert metrics["position_rmse_m"] == 0
    assert metrics["id_switches"] == 1
    assert metrics["idf1"] == .5


def test_camera_only_simulation_never_fabricates_lidar():
    config = load_config()
    config["sensor_positions"] = {"camera": [0, -7]}
    frames = generate(config, "sensor_dropout", duration=4)
    assert all(frame["sensors"] == ["camera"] for frame in frames)
    assert all(o["sensor"] == "camera" for frame in frames for o in frame["observations"])
    assert all("lidar" not in frame["health"] for frame in replay(frames, config))
