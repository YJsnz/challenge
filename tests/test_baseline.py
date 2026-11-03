import json
import pytest
from yudun.baseline import import_tracks
from yudun.evaluation import evaluate
from yudun.pipeline import read_jsonl


def test_baseline_units_and_unknown_fields_are_not_invented(tmp_path):
    inputs = [{"timestamp": 0., "ground_truth": [{"id": "p", "position": [2, 3]}]},
              {"timestamp": .1, "ground_truth": []}]
    source = tmp_path/"inputs.jsonl"
    source.write_text("".join(json.dumps(f)+"\n" for f in inputs))
    csv = tmp_path/"upstream.csv"
    csv.write_text("t,id,x,y\n0,p,200,300\n")
    mapping = tmp_path/"mapping.json"
    mapping.write_text(json.dumps({"schema_version": 1, "source_name": "upstream_test_fixture",
        "columns": {"timestamp": "t", "track_id": "id", "x": "x", "y": "y"},
        "time_scale_to_seconds": 1., "time_offset_seconds": 0.,
        "map_from_source_xy": [[.01, 0, 0], [0, .01, 0], [0, 0, 1]]}))
    output = tmp_path/"results.jsonl"
    assert import_tracks(source, csv, mapping, output) == 2
    results = list(read_jsonl(output))
    assert results[0]["tracks"][0]["state"][:2] == [2, 3]
    assert results[0]["tracks"][0]["predictions"] == []
    assert results[1]["tracks"] == []
    metrics = evaluate(inputs, results)
    assert metrics["position_rmse_m"] == 0
    assert metrics["processing_p50_ms"] is None
    assert metrics["position_coverage_95"] is None
