"""PhysicalAI-SmartSpaces 2025/2026 JSON and video adapter.

Schema source: https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces
Coordinates remain separate from floor-plan pixels and scaleFactor.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import re
from pathlib import Path
import numpy as np
from ..pipeline import load_config, read_jsonl
from ..schema import CLASSES, covariance

CLASS_ORDER = ("person", "forklift_or_cart", "AGV", "unknown_dynamic_object")
AUGMENTED_2026 = {1: 0, 3: 2, 5: 4, 7: 6, 21: 20, 24: 23}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def scene_group(release, scene_id):
    if release == "MTMC_Tracking_2026" and scene_id.startswith("Warehouse_"):
        number = int(scene_id.removeprefix("Warehouse_"))
        scene_id = f"Warehouse_{AUGMENTED_2026.get(number, number):03d}"
    return f"{release}/{scene_id}"


def fields(record):
    normalized = {}
    for name, value in record.items():
        key = name.replace(" ", "_")
        if key in normalized:
            raise ValueError(f"ambiguous duplicate field: {key}")
        normalized[key] = value
    return normalized


def vector(value, size, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return array


def clipped_bbox(box, width, height):
    x1, y1, x2, y2 = vector(box, 4, "bbox")
    x1, x2 = np.clip([x1, x2], 0, width)
    y1, y2 = np.clip([y1, y2], 0, height)
    return [float(x1), float(y1), float(x2), float(y2)] if x2 > x1 and y2 > y1 else None


@dataclass
class Camera:
    id: str
    fps: float
    width: int
    height: int
    homography: np.ndarray
    position_source: np.ndarray

    @classmethod
    def from_calibration(cls, calibration, camera_id):
        cameras = [s for s in calibration["sensors"] if s.get("type") == "camera" and s["id"] == camera_id]
        if len(cameras) != 1:
            raise ValueError(f"expected exactly one calibrated camera {camera_id}")
        sensor = cameras[0]
        attrs = {a["name"]: a["value"] for a in sensor["attributes"]}
        fps = float(attrs["fps"])
        width, height = int(attrs["frameWidth"]), int(attrs["frameHeight"])
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            raise ValueError("invalid calibrated fps or image dimensions")
        h = np.asarray(sensor["homography"], dtype=float)
        if h.shape != (3, 3) or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-12:
            raise ValueError("invalid world-ground-to-image homography")
        xy = sensor["coordinates"]
        return cls(camera_id, fps, width, height, h, vector([xy["x"], xy["y"]], 2, "camera coordinates"))

    def ground(self, pixel):
        # NVIDIA homography maps world ground-plane coordinates to image pixels.
        point = np.linalg.solve(self.homography, [*pixel, 1.])
        if abs(point[2]) < 1e-9:
            raise ValueError("pixel maps to ground-plane infinity")
        return point[:2]/point[2]


class SceneSpec:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.data = read_json(self.path)
        d = self.data
        if d.get("schema_version") != 1 or d["release"] not in ("MTMC_Tracking_2025", "MTMC_Tracking_2026"):
            raise ValueError("only explicit SmartSpaces 2025/2026 specs are supported")
        if d["domain"] not in ("synthetic", "real"):
            raise ValueError("declare source domain synthetic or real")
        if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", d[field]) for field in ("scene_id", "camera_id")):
            raise ValueError("scene_id and camera_id must be simple identifiers")
        self.root = (self.path.parent/d["scene_directory"]).resolve()
        self.calibration_path = self.root/"calibration.json"
        self.gt_path = self.root/"ground_truth.json"
        self.video = (self.root/d["video"]).resolve()
        if not self.video.is_relative_to(self.root):
            raise ValueError("video must be inside the selected scene directory")
        self.camera = Camera.from_calibration(read_json(self.calibration_path), d["camera_id"])
        self.scale = float(d["world_unit_to_m"])
        self.origin = vector(d["map_origin_source"], 2, "map origin")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("world_unit_to_m must be positive")
        self.roi = vector(d["roi_bounds_m"], 4, "ROI")
        if self.roi[0] >= self.roi[2] or self.roi[1] >= self.roi[3]:
            raise ValueError("invalid metric ROI bounds")
        self.class_map = d["class_map"]
        if not self.class_map or not set(self.class_map.values()) <= CLASSES:
            raise ValueError("class_map must explicitly map source classes to industrial classes")
        self.measurement_covariance = covariance(d["measurement_covariance_m2"], 2)
        for key in ("start_frame", "end_frame", "frame_stride"):
            if isinstance(d[key], bool) or not isinstance(d[key], int):
                raise ValueError(f"{key} must be an integer")
        if d["start_frame"] < 0 or d["end_frame"] <= d["start_frame"] or d["frame_stride"] <= 0:
            raise ValueError("invalid [start_frame,end_frame) or frame stride")
        if d["frame_stride"]/self.camera.fps > .5:
            raise ValueError("tracking sample interval must be <=0.5 s; subsample training separately")

    @property
    def sequence_id(self):
        return f"{self.data['release']}/{self.data['scene_id']}/{self.camera.id}"

    def map_xy(self, position):
        return (np.asarray(position, float)[:2]-self.origin)*self.scale

    def in_roi(self, point):
        return self.roi[0] <= point[0] <= self.roi[2] and self.roi[1] <= point[1] <= self.roi[3]

    def config(self):
        config = load_config()
        config["bounds"] = self.roi.tolist()
        config["obstacles"] = self.data.get("obstacles_m", [])
        config["sensor_positions"] = {"camera": self.map_xy(self.camera.position_source).tolist()}
        return config


def iter_gt(path, start, end, stride):
    """Stream large top-level frame dictionary; do not load the whole GT into RAM."""
    try:
        import ijson
    except ImportError as exc:
        raise RuntimeError("SmartSpaces conversion requires pip install -e '.[data]'") from exc
    previous, expected = -1, start
    with Path(path).open("rb") as stream:
        for key, objects in ijson.kvitems(stream, "", use_float=True):
            index = int(key)
            if index <= previous:
                raise ValueError("GT frame keys must be strictly increasing; refusing to reorder annotations silently")
            previous = index
            if index >= end:
                break
            if index < start or (index-start) % stride:
                continue
            if index != expected:
                raise ValueError(f"GT missing frame {expected}; missing labels are not empty scenes")
            if not isinstance(objects, list):
                raise ValueError(f"GT frame {index} must be a list")
            yield index, objects
            expected += stride
    if expected < end:
        raise ValueError(f"GT ended before requested frame {expected}")


def reference_frame(spec, index, objects):
    truth, labels, identities = [], [], set()
    for raw in objects:
        obj = fields(raw)
        source_category = obj["object_type"]
        category = spec.class_map.get(source_category)
        if category is None:
            continue
        identity = str(obj["object_id"])
        if identity in identities:
            raise ValueError(f"duplicate GT identity {identity} at frame {index}")
        identities.add(identity)
        reference_id = f"{spec.data['scene_id']}:{identity}"
        box = obj.get("2d_bounding_box_visible", {}).get(spec.camera.id)
        box = clipped_bbox(box, spec.camera.width, spec.camera.height) if box is not None else None
        if box:
            # Full-frame detection training needs all visible labels, including
            # objects outside the tracking ROI; otherwise they become false negatives.
            labels.append({"category": category, "bbox": box, "reference_id": reference_id})
        xyz = vector(obj["3d_location"], 3, "3D location")
        point = spec.map_xy(xyz)
        if not spec.in_roi(point):
            continue
        dimensions = vector(obj["3d_bounding_box_scale"], 3, "3D dimensions")*spec.scale
        if np.any(dimensions <= 0):
            raise ValueError("GT dimensions must be positive")
        truth.append({"id": reference_id, "category": category,
                      "position": point.tolist(), "radius": float(np.linalg.norm(dimensions[:2])/2),
                      "source_category": source_category})
    return {"schema_version": 1, "timestamp": index/spec.camera.fps, "frame_id": "map",
            "sensors": [], "observations": [], "reference_only": True,
            "ground_truth": truth, "reference_boxes": labels,
            "provenance": f"smartspaces_{spec.data['domain']}_reference",
            "sequence_id": spec.sequence_id, "source_frame": index}


def prepare(spec_path, output):
    spec = SceneSpec(spec_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reference_path = output/"reference.jsonl"
    if reference_path.exists():
        raise FileExistsError("reference.jsonl exists; use a new output directory")
    count = 0
    partial = reference_path.with_suffix(".jsonl.partial")
    with partial.open("x", encoding="utf-8") as stream:
        for index, objects in iter_gt(spec.gt_path, spec.data["start_frame"], spec.data["end_frame"], spec.data["frame_stride"]):
            stream.write(json.dumps(reference_frame(spec, index, objects), ensure_ascii=False, allow_nan=False)+"\n")
            count += 1
    partial.replace(reference_path)
    write_json(output/"scene.json", spec.config())
    write_json(output/"manifest.json", {"schema_version": 1, "sequence_id": spec.sequence_id,
               "scene_group": scene_group(spec.data["release"], spec.data["scene_id"]),
               "spec": str(spec.path), "spec_sha256": sha256(spec.path),
               "calibration_sha256": sha256(spec.calibration_path), "ground_truth_sha256": sha256(spec.gt_path),
               "reference_sha256": sha256(reference_path),
               "frames": count, "domain": spec.data["domain"], "reference_only": True,
               "source_url": "https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces", "license": "CC-BY-4.0"})
    return count


def video_frames(path, indices, camera):
    """Yield exact requested source frames, including gaps, with no silent EOF."""
    import cv2
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or abs(fps-camera.fps) > .05:
            raise ValueError("video FPS differs from calibration; explicitly resynchronise first")
        current, previous = 0, -1
        for index in indices:
            if index <= previous or index < 0:
                raise ValueError("requested video frames must be increasing and nonnegative")
            previous = index
            while current < index:
                if not capture.grab():
                    raise ValueError(f"video ended before frame {index}")
                current += 1
            ok, image = capture.read()
            current += 1
            if not ok or image is None:
                raise ValueError(f"cannot decode frame {index}")
            if image.shape[:2] != (camera.height, camera.width):
                raise ValueError("decoded image dimensions differ from calibration")
            yield index, image
    finally:
        capture.release()


def infer(spec_path, reference_path, model_path, metadata_path, output):
    from ..adapters import OnnxDetector
    from ..schema import Observation
    from dataclasses import asdict
    spec, metadata = SceneSpec(spec_path), read_json(metadata_path)
    reference_manifest = read_json(Path(reference_path).parent/"manifest.json")
    if (reference_manifest["spec_sha256"] != sha256(spec.path)
            or reference_manifest["reference_sha256"] != sha256(reference_path)
            or reference_manifest["calibration_sha256"] != sha256(spec.calibration_path)):
        raise ValueError("reference preparation hashes differ; prepare a new consistent sequence")
    if metadata.get("schema_version") != 1 or metadata.get("format") != "yolov8_detect_raw":
        raise ValueError("expected versioned raw YOLOv8 detect model metadata")
    if not set(metadata["class_names"]) <= set(spec.class_map.values()):
        raise ValueError("scene class_map must cover all model categories for consistent evaluation")
    if sha256(model_path) != metadata["onnx_sha256"]:
        raise ValueError("model and metadata hash mismatch")
    detector = OnnxDetector(model_path, metadata["class_names"], metadata["input_size"],
                            **metadata.get("detector_options", {}))
    refs = list(read_jsonl(reference_path))
    if not refs:
        raise ValueError("empty reference sequence")
    if any(r.get("sequence_id") != spec.sequence_id or not r.get("reference_only") for r in refs):
        raise ValueError("reference/spec mismatch")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("observations already exist; use a new output path")
    partial = output.with_suffix(output.suffix+".partial")
    rejected = total = 0
    with partial.open("x", encoding="utf-8") as stream:
        for reference, (index, image) in zip(refs, video_frames(spec.video, (r["source_frame"] for r in refs), spec.camera)):
            if index/spec.camera.fps != reference["timestamp"]:
                raise ValueError("reference timestamp differs from source video time")
            observations = []
            # Only pixels go into the model; reference labels are copied after inference.
            visual_detections = detector.detect(image)
            for detection_index, detection in enumerate(visual_detections):
                x1, y1, x2, y2 = detection["bbox"]
                try:
                    position = spec.map_xy(spec.camera.ground([(x1+x2)/2, y2]))
                    if not spec.in_roi(position):
                        continue
                    obs = Observation(reference["timestamp"], "camera", position.tolist(), detection["category"],
                                      detection["confidence"], spec.measurement_covariance.tolist(),
                                      quality=max(.05, detection["confidence"]),
                                      observation_id=f"{spec.camera.id}:{index}:{detection_index}",
                                      radius=.35 if detection["category"] == "person" else .6)
                    observations.append(asdict(obs))
                except ValueError:
                    rejected += 1
            frame = {key: reference[key] for key in ("schema_version", "timestamp", "frame_id", "sequence_id", "source_frame", "ground_truth")}
            frame.update({"sensors": ["camera"], "observations": observations,
                          "visual_detections": visual_detections,
                          "provenance": f"smartspaces_{spec.data['domain']}_model_observations"})
            stream.write(json.dumps(frame, ensure_ascii=False, allow_nan=False)+"\n")
            total += 1
    partial.replace(output)
    write_json(output.with_suffix(output.suffix+".manifest.json"), {"schema_version": 1, "frames": total,
               "spec_sha256": sha256(spec.path), "reference_sha256": sha256(reference_path),
               "onnx_sha256": metadata["onnx_sha256"], "model_metadata_sha256": sha256(metadata_path),
               "video_sha256": sha256(spec.video), "projection_rejections": rejected,
               "domain": spec.data["domain"], "ground_truth_used_for_observations": False})
    return total
