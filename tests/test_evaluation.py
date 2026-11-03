import numpy as np
import pytest
from yudun.evaluation import evaluate, uncertainty_score


def fixture_sequence():
    inputs, results = [], []
    for step in range(31):
        timestamp = step/10
        inputs.append({"timestamp": timestamp, "ground_truth": [{"id": "p", "category": "person", "position": [timestamp, 0]}]})
        results.append({"timestamp": timestamp, "processing_ms": 1, "events": [], "tracks": [
            {"id": 1, "category": "person", "state": [timestamp, 0, 1, 0], "covariance": np.eye(4).tolist(),
             "predictions": [{"dt": i/10, "mean": [timestamp+i/10+.2, 0], "covariance": np.eye(2).tolist()} for i in range(1, 31)]}]})
    return inputs, results


def test_ade_fde_and_coverage_use_independent_future_gt():
    inputs, results = fixture_sequence()
    metrics = evaluate(inputs, results)
    for horizon in metrics["forecast_displacement_m"].values():
        assert horizon["fde"] == pytest.approx(.2)
        assert horizon["ade"] == pytest.approx(.2)
        assert horizon["coverage_95"] == 1
    assert metrics["position_coverage_95"] == 1


def test_missing_gt_is_not_empty_gt_and_alignment_is_strict():
    inputs, results = fixture_sequence()
    inputs[0].pop("ground_truth")
    assert evaluate(inputs, results)["annotated_frames"] == 30
    with pytest.raises(ValueError, match="frame counts"):
        evaluate(inputs, results[:-1])
    results[0]["timestamp"] = .01
    with pytest.raises(ValueError, match="timestamps"):
        evaluate(inputs, results)


def test_known_categories_cannot_match_each_other():
    inputs, results = fixture_sequence()
    for result in results:
        result["tracks"][0]["category"] = "forklift_or_cart"
    assert evaluate(inputs, results)["matched_samples"] == 0


def test_incomplete_forecast_horizon_is_not_reported_as_ade():
    inputs, results = fixture_sequence()
    for result in results:
        result["tracks"][0]["predictions"] = [result["tracks"][0]["predictions"][4]]
    horizon = evaluate(inputs, results)["forecast_displacement_m"]["0.5"]
    assert horizon["fde"] == pytest.approx(.2)
    assert horizon["ade"] is None


def test_tiny_covariance_exposes_overconfidence():
    covered, nll = uncertainty_score(np.array([1., 1.]), np.eye(2)*.001)
    assert not covered and nll > 100


def test_nonfinite_result_timestamp_is_rejected():
    inputs, results = fixture_sequence()
    results[0]["timestamp"] = float("nan")
    with pytest.raises(ValueError, match="timestamps"):
        evaluate(inputs, results)
