from collections import Counter

from yudun.forklift_operator import PROFILE
from yudun.pipeline import load_config, replay


def frame(index, *, moving=True, rider=True, walk_out=False, duplicate=False, other=True):
    shift = 12*index if moving else 0
    vehicle = [800, 230+shift, 1060, 900+shift]
    detections = [("forklift_or_cart", vehicle, [0, 0])]
    if rider:
        box = [896, 360+shift, 960, 485+shift]
        if walk_out:
            box = [1010, 410, 1100, 925]
        detections.append(("person", box, [.1, 0]))
    if other:
        detections.extend([("person", [1040, 400, 1120, 940], [0, .4]),
                           ("forklift_or_cart", [1300, 220, 1550, 850], [.4, .4])])
    if duplicate:
        detections.append(("forklift_or_cart", [2, 1, 1066, 1078], [0, .2]))
    t = index/15
    return {"schema_version": 1, "frame_id": "map", "timestamp": t, "sensors": ["camera"],
            "provenance": "illustrative_video_demo", "source_frame": index, "image_size": [1920, 1080],
            "observations": [{"timestamp": t, "sensor": "camera", "category": category, "position": position,
                              "radius": .35 if category == "person" else .6,
                              "observation_id": f"camera:{index}:{i}"}
                             for i, (category, _, position) in enumerate(detections)],
            "visual_detections": [{"category": category, "bbox": box} for category, box, _ in detections]}


def paired(frame):
    return {tuple(a["pair"]) for a in frame["risk_assessments"]}


def test_only_confirmed_own_vehicle_pair_is_excluded():
    config = load_config()
    inputs = [frame(i) for i in range(5)]
    baseline = replay(inputs, config)
    optimized = replay(inputs, {**config, "forklift_operator_profile": PROFILE})
    assert (1, 2) in paired(baseline[-1])
    assert (1, 2) in paired(optimized[1])
    assert paired(optimized[-1]) == paired(baseline[-1]) - {(1, 2)}
    assert (1, 3) in paired(optimized[-1]) and (2, 4) in paired(optimized[-1])
    assert {tuple(e["pair"]) for e in optimized[-1]["events"] if e["level"]} >= {(1, 3), (2, 4)}
    assert optimized[-1]["visual_detections"] == baseline[-1]["visual_detections"]
    assert {t["id"] for t in optimized[-1]["tracks"]} == {1, 2, 3, 4}


def test_overlap_alone_is_insufficient_and_evidence_expires():
    config = {**load_config(), "forklift_operator_profile": PROFILE}
    assert not any(f["operator_associations"] for f in replay([frame(i, moving=False) for i in range(5)], config))
    inputs = [frame(i, other=False) for i in range(3)]
    gap = replay(inputs+[frame(3, rider=False, other=False), frame(9, rider=False, other=False)], config)
    assert gap[3]["operator_associations"][0]["evidence"] == "brief_detection_gap"
    assert not gap[-1]["operator_associations"] and (1, 2) in paired(gap[-1])
    exit_cab = replay(inputs+[frame(3, walk_out=True, other=False)], config)
    assert not exit_cab[-1]["operator_associations"] and (1, 2) in paired(exit_cab[-1])
    unrelated = [{**f, "image_size": [1280, 720]} for f in inputs]
    assert not any(f["operator_associations"] for f in replay(unrelated, config))


def test_duplicate_screen_edge_vehicle_keeps_audit_box_but_no_track():
    config = {**load_config(), "forklift_operator_profile": PROFILE}
    inputs = [frame(i, other=False, duplicate=True) for i in range(3)]
    results = replay(inputs, config)
    assert results[-1]["duplicate_vehicle_observations"] == ["camera:2:2"]
    assert Counter(t["category"] for t in results[-1]["tracks"]) == {"forklift_or_cart": 1, "person": 1}
    assert results[-1]["visual_detections"] == inputs[-1]["visual_detections"]
