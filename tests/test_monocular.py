from dataclasses import asdict
import pytest
from yudun.pipeline import Pipeline, load_config, replay
from yudun.schema import Observation
from yudun.simulation import generate
from yudun.tracking import MODES


def test_default_mainline_and_health_are_camera_only():
    config = load_config()
    assert set(config["sensor_positions"]) == {"camera"}
    assert MODES == ("monocular_basic", "monocular")
    result = Pipeline().process({"schema_version": 1, "timestamp": 0., "frame_id": "map", "sensors": ["camera"], "observations": []})
    assert result["mode"] == "monocular"
    assert result["health"] == {"camera": "ONLINE"}
    assert result["system_status"] == "NORMAL"


def test_monocular_mainline_rejects_lidar_and_mixed_configuration():
    config = load_config()
    config["sensor_positions"]["lidar"] = [1, 1]
    with pytest.raises(ValueError, match="exactly one"):
        Pipeline(config)
    with pytest.raises(ValueError, match="cannot contain"):
        Pipeline().process({"schema_version": 1, "timestamp": 0., "frame_id": "map", "sensors": ["lidar"],
                            "observations": [asdict(Observation(0, "lidar", [0, 0]))]})


def test_camera_dropout_causes_degradation_and_track_expiry():
    config = load_config()
    frames = generate(config, "camera_dropout", duration=8)
    results = replay(frames, config)
    outage = [(f, r) for f, r in zip(frames, results) if 6 <= f["timestamp"] < 7]
    assert outage and all(r["health"]["camera"] == "OFFLINE" and not r["tracks"] for _, r in outage)
