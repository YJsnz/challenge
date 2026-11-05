"""Conservative, opt-in operator association for the Warehouse_026 front view."""
from collections import Counter
from dataclasses import dataclass
import numpy as np


PROFILE = "warehouse026_front"
IMAGE_SIZE = (1920, 1080)
EVIDENCE_GAP_S = .40


def detection_boxes(frame, observations):
    """Link direct-camera detections to accepted source observations, including older logs."""
    if (frame.get("provenance") not in {"monocular_video_model", "illustrative_video_demo"}
            or not isinstance(frame.get("source_frame"), int)
            or tuple(frame.get("image_size", IMAGE_SIZE)) != IMAGE_SIZE):
        return {}
    categories = {obs.observation_id: obs.category for obs in observations
                  if obs.sensor == "camera" and obs.timestamp == frame["timestamp"]}
    boxes = {}
    for index, detection in enumerate(frame.get("visual_detections", [])):
        identity = f"camera:{frame['source_frame']}:{index}"
        category = categories.get(identity)
        if not category or category != detection.get("category"):
            continue
        try:
            box = np.asarray(detection.get("bbox"), float)
        except (TypeError, ValueError):
            continue
        if box.shape == (4,) and np.isfinite(box).all() and np.all(box[2:] > box[:2]):
            boxes[identity] = (category, box)
    return boxes


def duplicate_vehicle_boxes(boxes):
    vehicles = {key: box for key, (category, box) in boxes.items() if category == "forklift_or_cart"}
    height = IMAGE_SIZE[1]
    rejected = set()
    for key, box in vehicles.items():
        if box[1] > .025*height or box[3] < .975*height:
            continue
        for other in vehicles.values():
            area = np.prod(other[2:]-other[:2])
            if area > .45*np.prod(box[2:]-box[:2]):
                continue
            intersection = np.maximum(0, np.minimum(box[2:], other[2:])-np.maximum(box[:2], other[:2]))
            if np.prod(intersection) >= .98*area:
                rejected.add(key)
                break
    return rejected


def cabin_position(person, vehicle):
    width, height = vehicle[2:]-vehicle[:2]
    if height < .45*IMAGE_SIZE[1]:
        return None
    left, top, right, bottom = (person-np.tile(vehicle[:2], 2))/[width, height, width, height]
    centre = (left+right)/2
    # Front approach: a compact visible upper body inside the upper part of the mast.
    if (.38 <= centre <= .68 and .12 <= top <= .29 and .27 <= bottom <= .52
            and .13 <= right-left <= .45 and .07 <= bottom-top <= .38):
        return np.array([centre, top])
    return None


@dataclass
class Evidence:
    timestamp: float
    anchor: np.ndarray
    roof: np.ndarray
    hits: int = 1
    confirmed: bool = False


class OperatorAssociator:
    def __init__(self):
        self.evidence = {}

    def update(self, timestamp, boxes, assignments, tracks):
        live = {track.id for track in tracks}
        keys = {assignments[key]: key for key in boxes if key in assignments}
        matched = {track_id: boxes[key] for track_id, key in keys.items()}
        people = {key: box for key, (category, box) in matched.items() if category == "person"}
        vehicles = {key: box for key, (category, box) in matched.items() if category == "forklift_or_cart"}
        candidates = {}
        for person_id, person in people.items():
            for vehicle_id, vehicle in vehicles.items():
                anchor = cabin_position(person, vehicle)
                if anchor is not None:
                    candidates[person_id, vehicle_id] = anchor
        people_count = Counter(p for p, _ in candidates)
        vehicle_count = Counter(v for _, v in candidates)
        current, links = {}, []
        for (person_id, vehicle_id), anchor in candidates.items():
            if people_count[person_id] != 1 or vehicle_count[vehicle_id] != 1:
                continue
            key = person_id, vehicle_id
            vehicle = vehicles[vehicle_id]
            roof = np.array([(vehicle[0]+vehicle[2])/2, vehicle[1]])
            evidence = self.evidence.get(key)
            if (evidence is None or timestamp-evidence.timestamp > EVIDENCE_GAP_S
                    or np.max(np.abs(evidence.anchor-anchor)) > .12):
                evidence = Evidence(timestamp, anchor, roof)
            else:
                evidence.timestamp = timestamp
                evidence.hits += 1
                displacement = np.linalg.norm(evidence.roof-roof)/(vehicle[3]-vehicle[1])
                evidence.confirmed |= evidence.hits >= 3 and displacement >= .025
            current[key] = evidence
            if evidence.confirmed:
                links.append((person_id, vehicle_id, "observed"))
        for key, evidence in self.evidence.items():
            person_id, vehicle_id = key
            if (person_id in live and person_id not in people and vehicle_id in vehicles
                    and not vehicle_count[vehicle_id] and timestamp-evidence.timestamp <= EVIDENCE_GAP_S):
                current[key] = evidence
                if evidence.confirmed:
                    links.append((person_id, vehicle_id, "brief_detection_gap"))
        self.evidence = current
        return [{"person_track_id": p, "vehicle_track_id": v, "evidence": evidence,
                 "person_observation_id": keys.get(p)} for p, v, evidence in links]
