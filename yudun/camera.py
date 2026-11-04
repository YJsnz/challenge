"""Fixed monocular video/USB acquisition; usable without ROS or a web service."""
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from .adapters import OnnxDetector, camera_observation
from .pipeline import Pipeline


def load_camera(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if data.get("schema_version") != 1 or data.get("frame_id") != "map" or data.get("example_only", True):
        raise ValueError("supply verified camera calibration (schema 1, map, example_only=false)")
    if set(data)-{"schema_version", "frame_id", "example_only", "camera"}:
        raise ValueError("monocular calibration must contain only the camera section")
    c = data["camera"]
    c["model"] = str((path.parent/c["model"]).resolve())
    if not Path(c["model"]).is_file():
        raise ValueError("industrial ONNX model does not exist")
    transform = np.asarray(c["world_from_camera"], float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("camera extrinsic must be finite 4x4 world_from_camera")
    rotation = transform[:3, :3]
    if not np.allclose(transform[3], [0, 0, 0, 1]) or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1):
        raise ValueError("camera extrinsic requires a proper rigid rotation")
    k = np.asarray(c["intrinsics"], float)
    if k.shape != (3, 3) or not np.isfinite(k).all() or abs(np.linalg.det(k)) < 1e-9:
        raise ValueError("invalid camera intrinsics")
    from .schema import covariance
    covariance(c["measurement_covariance"], 2)
    if any(not isinstance(c[key], int) or c[key] <= 0 for key in ("image_width", "image_height")):
        raise ValueError("positive calibrated image width/height required")
    return c


def camera_frames(source, calibration, config, mode="monocular", stop=lambda: False):
    import cv2
    c = load_camera(calibration)
    if set(config["sensor_positions"]) != {"camera"}:
        raise ValueError("single-camera scene configuration required")
    if not np.allclose(config["sensor_positions"]["camera"], np.asarray(c["world_from_camera"])[:2, 3], atol=.01):
        raise ValueError("scene camera XY must match calibration world_from_camera")
    detector = OnnxDetector(c["model"], c["class_names"], c.get("input_size", 640))
    capture = cv2.VideoCapture(source)
    try:
        if not capture.isOpened():
            raise ValueError(f"camera/video cannot be opened: {source}")
        file_source = isinstance(source, str)
        if not file_source:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, c["image_width"])
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, c["image_height"])
        fps = capture.get(cv2.CAP_PROP_FPS)
        if file_source and (not np.isfinite(fps) or fps <= 0):
            raise ValueError("video must provide a valid FPS")
        pipeline, start, index = Pipeline(config, mode), time.monotonic(), 0
        while not stop():
            ok, image = capture.read()
            if not ok:
                if file_source:
                    return
                raise RuntimeError("camera frame acquisition stopped")
            timestamp = index/fps if file_source else time.monotonic()-start
            while file_source and not stop():
                remaining = timestamp-(time.monotonic()-start)
                if remaining <= 0:
                    break
                time.sleep(min(.05, remaining))
            if stop():
                return
            if image.shape[:2] != (c["image_height"], c["image_width"]):
                raise ValueError("image resolution differs from calibration; recalibrate or correct input resolution")
            detections, observations = detector.detect(image), []
            rejected = 0
            for i, d in enumerate(detections):
                try:
                    obs = camera_observation(d["bbox"], d["category"], d["confidence"], timestamp,
                           c["intrinsics"], c["world_from_camera"], c["measurement_covariance"],
                           c.get("distortion"), f"camera:{index}:{i}")
                    x, y = obs.position
                    x0, y0, x1, y1 = config["bounds"]
                    if x0 <= x <= x1 and y0 <= y <= y1:
                        observations.append(asdict(obs))
                except ValueError:
                    rejected += 1
            frame = {"schema_version": 1, "frame_id": "map", "timestamp": timestamp,
                     "sensors": ["camera"], "observations": observations, "visual_detections": detections,
                     "provenance": "monocular_video_model" if file_source else "monocular_usb_model",
                     "source_frame": index, "projection_rejections": rejected}
            yield frame, pipeline.process(frame), image
            index += 1
    finally:
        capture.release()
