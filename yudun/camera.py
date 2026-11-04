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
