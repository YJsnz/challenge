import json
from pathlib import Path
import numpy as np
import pytest
from yudun.lidar import lidar_frame
from yudun.pipeline import Pipeline, load_config
from yudun.schema import validate_frame


def extension():
    path = Path(__file__).resolve().parents[1]/"configs/lidar_extension.example.json"
    config = json.loads(path.read_text())
    config["enabled"], config["example_only"] = True, False
    return config


def test_reserved_adapter_preserves_metric_frame_and_unknown_semantics():
    config = extension()
    # A single compact cluster in sensor space; map translation is [1,-7,2].
    points = np.array([[.01*i, 0, 0] for i in range(10)])
    frame = lidar_frame(points, 12.5, "lidar_frame", config)
    observations = validate_frame(frame)
    assert frame["timestamp"] == 12.5 and frame["sensors"] == ["lidar"]
    assert len(observations) == 1
    np.testing.assert_allclose(observations[0].position, [1.045, -7])
    assert observations[0].category == "unknown_dynamic_object"
    with pytest.raises(ValueError, match="cannot contain"):
        Pipeline().process(frame)


def test_empty_cloud_remains_heartbeat_and_extension_requires_opt_in():
    config = extension()
    assert lidar_frame([], 1, "lidar_frame", config)["observations"] == []
    config["enabled"] = False
    with pytest.raises(ValueError, match="disabled"):
        lidar_frame([], 1, "lidar_frame", config)
    config["enabled"], config["example_only"] = True, True
    with pytest.raises(ValueError, match="example"):
        lidar_frame([], 1, "lidar_frame", config)
    assert set(load_config()["sensor_positions"]) == {"camera"}
