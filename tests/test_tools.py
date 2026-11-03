import importlib.util
from pathlib import Path
import pytest


def load_script(name):
    path = Path(__file__).resolve().parents[1]/"scripts"/(name+".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_group_split_prevents_leakage():
    split = load_script("split_dataset").split
    records = [{"sequence_id": str(i), "scene_group": f"layout-{i//2}"} for i in range(12)]
    result = split(records)
    groups = {key: {r["scene_group"] for r in rows} for key, rows in result.items()}
    assert not groups["train"] & groups["test"]
    assert not groups["train"] & groups["validation"]
    assert not groups["test"] & groups["validation"]
    assert sum(len(v) for v in result.values()) == len(records)
    with pytest.raises(ValueError):
        split(records[:2])


def test_event_recall_lead_and_false_episodes():
    evaluate = load_script("evaluate_events").evaluate_events
    inputs, outputs = [], []
    for index in range(8):
        inputs.append({"timestamp": index, "ground_truth": [{"id": "p", "position": [0, 0]}, {"id": "c", "position": [1, 0]}]})
        event = {"event_id": 1 if index < 4 else 2, "pair": [1, 2], "level": 2}
        outputs.append({"timestamp": index, "tracks": [{"id": 1, "state": [0, 0, 0, 0]}, {"id": 2, "state": [1, 0, 0, 0]}],
                        "events": [event] if index in (1, 2, 5) else []})
    metrics = evaluate(inputs, outputs, {"events": [{"id": "e", "pair": ["p", "c"], "start": 0, "danger_time": 2, "end": 3}]})
    assert metrics["recall"] == 1
    assert metrics["median_lead_seconds"] == 1
    assert metrics["false_alert_episodes"] == 1
