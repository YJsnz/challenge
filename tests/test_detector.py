import json
import numpy as np
import pytest
from yudun.adapters import OnnxDetector


def detector(monkeypatch, outputs, **options):
    cv2 = pytest.importorskip("cv2")
    class Net:
        def __init__(self):
            self.outputs = iter(outputs)
        def setInput(self, blob):
            pass
        def forward(self):
            rows = np.asarray(next(self.outputs), dtype=np.float32).reshape(-1, 6)
            return rows.T[None]
    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda path: Net())
    return OnnxDetector("unused.onnx", ["forklift_or_cart", "person"], size=64, **options)


def test_per_class_thresholds_apply_to_candidates_and_nms(monkeypatch):
    model = detector(monkeypatch, [[[20, 20, 10, 10, .1, .2], [45, 45, 10, 10, .3, .1]]],
                     class_confidence={"person": .15})
    detections = model.detect(np.zeros((64, 64, 3), np.uint8))
    assert len(detections) == 1 and detections[0]["category"] == "person"
    # Video frames include visual_detections in JSONL; numpy float32 cannot be dumped.
    assert json.loads(json.dumps(detections))[0]["bbox"] == [15, 15, 25, 25]


def test_tiles_merge_duplicates_reject_internal_edges_and_keep_image_edges(monkeypatch):
    outputs = [[] for _ in range(10)]  # full frame, then a 3x3 overlapping grid
    outputs[1] = [[50, 20, 20, 20, .1, .9], [63, 45, 20, 10, .1, .95]]
    outputs[2] = [[18, 20, 20, 20, .1, .8]]  # same global box as the first tile
    outputs[3] = [[0, 45, 20, 10, .1, .9]]   # clipped by an internal left edge
    outputs[9] = [[65, 65, 10, 10, .1, .9]]  # clipped by the actual image boundary
    model = detector(monkeypatch, outputs, tile_size=64)
    results = model.detect(np.zeros((128, 128, 3), np.uint8))
    assert len(results) == 2
    assert sorted(d["bbox"] for d in results) == [[40, 10, 60, 30], [124, 124, 128, 128]]


@pytest.mark.parametrize("options", [
    {"class_confidence": {"truck": .4}}, {"class_confidence": {"person": 0}},
    {"class_confidence": {"person": float("nan")}}, {"tile_size": 0},
    {"tile_overlap": 1}, {"tile_edge_margin": -1}, {"tile_size": 64, "tile_edge_margin": 32},
])
def test_invalid_detector_options_fail_before_inference(monkeypatch, options):
    with pytest.raises(ValueError):
        detector(monkeypatch, [], **options)
