"""Calibrated sensor adapters. No competition model is relabelled as industrial."""
from dataclasses import asdict
import numpy as np
from scipy.spatial import cKDTree
from .geometry import ground_projection, transform_points
from .schema import Observation


def camera_observation(bbox, category, confidence, timestamp, intrinsics, world_from_camera,
                       covariance=None, distortion=None, observation_id=""):
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError("invalid xyxy bounding box")
    pixel = np.array([(x1+x2)/2, y2], dtype=float)
    if distortion is not None:
        import cv2
        pixel = cv2.undistortPoints(pixel.reshape(1, 1, 2), np.asarray(intrinsics, float),
                                    np.asarray(distortion, float), P=np.asarray(intrinsics, float)).reshape(2)
    position = ground_projection(pixel, intrinsics, world_from_camera)
    return Observation(timestamp, "camera", position.tolist(), category, confidence,
                       covariance if covariance is not None else [[0.25, 0], [0, 0.25]],
                       max(0.05, min(1.0, confidence)), observation_id=observation_id,
                       radius=0.35 if category == "person" else 0.6)


def lidar_observations(points, timestamp, world_from_lidar=None, static_map=None,
                       background_distance=0.25, cluster_distance=0.4, min_points=5, ground_z=0.15):
    if cluster_distance <= 0 or background_distance < 0 or min_points < 1:
        raise ValueError("invalid clustering parameters")
    points = np.asarray(points, float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if world_from_lidar is not None:
        points = transform_points(points, world_from_lidar)
    points = points[points[:, 2] > ground_z]
    if static_map is not None and len(static_map) and len(points):
        distances, _ = cKDTree(np.asarray(static_map, float)).query(points)
        points = points[distances > background_distance]
    if not len(points):
        return []
    tree, visited, results = cKDTree(points), set(), []
    for root in range(len(points)):
        if root in visited:
            continue
        pending, component = [root], []
        visited.add(root)
        while pending:
            index = pending.pop()
            component.append(index)
            for neighbor in tree.query_ball_point(points[index], cluster_distance):
                if neighbor not in visited:
                    visited.add(neighbor)
                    pending.append(neighbor)
        if len(component) < min_points:
            continue
        cloud = points[component]
        center = cloud.mean(axis=0)
        cov = np.cov(cloud[:, :2], rowvar=False) / len(component) if len(component) > 1 else np.eye(2)*0.09
        cov += np.eye(2)*0.02
        results.append(Observation(timestamp, "lidar", center[:2].tolist(), covariance=cov.tolist(),
                                   quality=min(1.0, len(component)/30),
                                   observation_id=f"{timestamp:.9f}:{len(results)}"))
    return results


class OnnxDetector:
    """OpenCV DNN for YOLOv8 detect exports: [1,4+C,N] or [1,N,4+C]."""
    def __init__(self, model_path, class_names, size=640, confidence=0.4):
        import cv2
        from .schema import CLASSES
        if not set(class_names) <= CLASSES:
            raise ValueError("supply explicit industrial model class names")
        self.cv2, self.net = cv2, cv2.dnn.readNetFromONNX(str(model_path))
        self.names, self.size, self.confidence = class_names, size, confidence

    def detect(self, image):
        cv2, size = self.cv2, self.size
        height, width = image.shape[:2]
        scale = min(size/width, size/height)
        resized = cv2.resize(image, (round(width*scale), round(height*scale)))
        left, top = (size-resized.shape[1])//2, (size-resized.shape[0])//2
        padded = np.full((size, size, 3), 114, dtype=np.uint8)
        padded[top:top+resized.shape[0], left:left+resized.shape[1]] = resized
        self.net.setInput(cv2.dnn.blobFromImage(padded, 1/255, (size, size), swapRB=True))
        output = np.asarray(self.net.forward()).squeeze(axis=0)
        expected = 4 + len(self.names)
        if output.ndim != 2:
            raise ValueError("unsupported model output")
        if output.shape[0] == expected:
            output = output.T
        if output.shape[1] != expected:
            raise ValueError("expected raw YOLOv8 detection output (no objectness or end-to-end NMS)")
        candidates = []
        for row in output:
            label = int(np.argmax(row[4:]))
            score = float(row[4+label])
            if score < self.confidence:
                continue
            cx, cy, w, h = row[:4]
            box = [(cx-w/2-left)/scale, (cy-h/2-top)/scale, w/scale, h/scale]
            candidates.append((box, score, label))
        results = []
        for label in range(len(self.names)):
            subset = [x for x in candidates if x[2] == label]
            keep = cv2.dnn.NMSBoxes([x[0] for x in subset], [x[1] for x in subset], self.confidence, 0.45)
            for index in np.asarray(keep).reshape(-1):
                (x, y, w, h), score, _ = subset[int(index)]
                bbox = [max(0., x), max(0., y), min(float(width), x+w), min(float(height), y+h)]
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    results.append({"bbox": bbox, "category": self.names[label], "confidence": score})
        return results


def observations_frame(timestamp, observations, sensors):
    return {"schema_version": 1, "timestamp": timestamp, "frame_id": "map",
            "sensors": sensors, "observations": [asdict(o) for o in observations], "provenance": "sensor_adapter"}
