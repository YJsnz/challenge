import csv
import json
from pathlib import Path

import numpy as np
import pytest

from yudun.camera import camera_frames, load_camera
from yudun.evaluation import save_run
from yudun.map3d.__main__ import project
from yudun.map3d.demo import make_demo
from yudun.map3d.workflows import calibrate, check_camera, read, register_map, write
from yudun.pipeline import Pipeline, load_config, read_jsonl


@pytest.fixture
def demo(tmp_path):
    pytest.importorskip("cv2")
    model = tmp_path/"model.onnx"
    model.touch()  # These tests exercise the calibration and replay, not inference.
    return make_demo(tmp_path/"demo", model=model)


def test_demo_replays_and_exports_map_coordinates(tmp_path, demo):
    config = load_config(demo["scene_config"])
    pipeline = Pipeline(config)
    inputs = list(read_jsonl(demo["input"]))
    with pytest.raises(ValueError, match="different map"):
        pipeline.process({**inputs[0], "map_scene_sha256": "stale"})
    results = [pipeline.process(frame) for frame in inputs]
    assert [f["map_3d"] for f in results] == [f["map_3d"] for f in read_jsonl(demo["results"])]
    save_run(tmp_path/"run", inputs, results, config, 17)
    project(Path(demo["results"]), Path(demo["map_scene"]), tmp_path/"projected")
    for folder in ("run", "projected"):
        with (tmp_path/folder/"map_tracks.csv").open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        assert {row["category"] for row in rows} == {"person", "forklift_or_cart"}
        assert all(row["example_only"] == "True" and float(row["z_m"]) != 0 for row in rows)


def test_demo_calibration_requires_video_and_pins_artifacts(tmp_path, demo):
    with pytest.raises(ValueError, match="verified camera calibration"):
        load_camera(demo["calibration"])
    config = load_config(demo["scene_config"])
    with pytest.raises(ValueError, match="verified camera calibration"):
        next(camera_frames(0, demo["calibration"], config, allow_example=True))
    with pytest.raises(ValueError, match="verified camera calibration"):
        next(camera_frames("http://localhost/video", demo["calibration"], config, allow_example=True))
    camera = load_camera(demo["calibration"], allow_example=True)
    assert camera["example_only"]
    Path(camera["model"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="model hash mismatch"):
        load_camera(demo["calibration"], allow_example=True)


def test_demo_camera_calibration_and_independent_recheck(tmp_path, demo):
    request = Path(demo["map_scene"]).with_name("calibration-request.example.json")
    calibration = calibrate(request, tmp_path/"calibrated")
    check = tmp_path/"check.json"
    write(check, {"calibration": str(calibration), "check_points": read(request)["check_points"]})
    report = check_camera(check)
    assert report["checkpoints"] == 4 and report["example_only"]
    assert report["position_max_m"] < 1e-6


def test_measured_map_requires_independent_registration_checks(tmp_path):
    points = [[0, 0, 0], [4, 0, 0], [0, 5, 0]]
    np.save(tmp_path/"cloud.npy", points)
    request = {"geometry": "cloud.npy", "unit_to_m": 1, "example_only": False,
               "ground_points_m": points, "map_points_m": points}
    path = tmp_path/"register.json"
    write(path, request)
    with pytest.raises(ValueError, match="independent"):
        register_map(path, tmp_path/"registered")
    checks = [[1, 1, 0], [2, 1, 0], [3, 2, 0]]
    request["check_points"] = {"ground_points_m": checks, "map_points_m": checks}
    path.write_text(json.dumps(request), encoding="utf-8")
    assert register_map(path, tmp_path/"registered").is_file()
    assert read(tmp_path/"registered/registration-report.json")["passed"]
