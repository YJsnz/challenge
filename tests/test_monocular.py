from dataclasses import asdict
import json
import numpy as np
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


@pytest.fixture
def camera_calibration(tmp_path):
    model = tmp_path/"model.onnx"
    model.write_bytes(b"test fixture, never loaded as an actual model")
    path = tmp_path/"camera.json"
    data = {"schema_version": 1, "frame_id": "map", "example_only": False, "camera": {
        "model": "model.onnx", "class_names": ["person"], "image_width": 640, "image_height": 480,
        "intrinsics": [[100, 0, 320], [0, 100, 240], [0, 0, 1]],
        "world_from_camera": [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 3], [0, 0, 0, 1]],
        "measurement_covariance": [[.25, 0], [0, .25]]}}
    path.write_text(json.dumps(data))
    return path


def test_camera_calibration_rejects_examples_and_dual_sensor_sections(camera_calibration):
    from yudun.camera import load_camera
    assert load_camera(camera_calibration)["image_width"] == 640
    data = json.loads(camera_calibration.read_text())
    data["lidar"] = {}
    camera_calibration.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="only the camera"):
        load_camera(camera_calibration)
    data.pop("lidar")
    data["example_only"] = True
    camera_calibration.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="verified"):
        load_camera(camera_calibration)


def test_native_video_timestamps_and_capture_cleanup(camera_calibration, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    import yudun.camera as camera
    class Capture:
        frames = 0
        released = False
        def isOpened(self):
            return True
        def get(self, prop):
            return 10.
        def read(self):
            self.frames += 1
            return (True, np.zeros((480, 640, 3), np.uint8)) if self.frames <= 2 else (False, None)
        def release(self):
            self.released = True
    capture = Capture()
    class Detector:
        def __init__(self, *args):
            pass
        def detect(self, image):
            return [{"bbox": [310, 200, 330, 240], "category": "person", "confidence": .9}]
    monkeypatch.setattr(cv2, "VideoCapture", lambda source: capture)
    monkeypatch.setattr(camera, "OnnxDetector", Detector)
    config = load_config()
    config["sensor_positions"] = {"camera": [0, 0]}
    config["obstacles"] = []
    frames = list(camera.camera_frames("fixture.mp4", camera_calibration, config))
    assert [frame[0]["timestamp"] for frame in frames] == [0, .1]
    assert all(frame[0]["sensors"] == ["camera"] for frame in frames)
    assert capture.released
