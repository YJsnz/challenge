import copy
import pytest
from yudun.risk_evaluation import evaluate_probabilities


def example(probability, close):
    inputs, results = [], []
    for i in range(11):
        t = i/10
        inputs.append({"timestamp": t, "ground_truth": [{"id": "p", "position": [0, 0]}, {"id": "c", "position": [1 if close else 4, 0]}]})
        results.append({"timestamp": t, "tracks": [{"id": 1, "state": [0, 0, 0, 0]}, {"id": 2, "state": [1 if close else 4, 0, 0, 0]}],
                        "risk_assessments": [{"pair": [1, 2], "probability": probability, "horizon": 1., "safe_distance": 2.}] if i == 0 else []})
    return inputs, results


def test_correct_and_overconfident_probabilities():
    inputs, results = example(.9, True)
    metrics = evaluate_probabilities(inputs, results)
    assert metrics["samples"] == 1
    assert metrics["brier_score"] == pytest.approx(.01)
    assert metrics["ece"] == pytest.approx(.1)
    inputs, results = example(.9, False)
    assert evaluate_probabilities(inputs, results)["brier_score"] == pytest.approx(.81)


def test_unalerted_pairs_and_zero_probabilities_are_retained():
    inputs, results = example(0, False)
    metrics = evaluate_probabilities(inputs, results)
    assert metrics["samples"] == 1 and metrics["brier_score"] == 0
    assert metrics["bins"][0]["samples"] == 1


def test_missing_future_gt_is_not_counted_as_safe():
    inputs, results = example(.1, False)
    inputs[-1]["ground_truth"] = []
    metrics = evaluate_probabilities(inputs, results)
    assert metrics["samples"] == 0
    assert metrics["excluded"]["incomplete_future_gt"] == 1


def test_invalid_probabilities_are_rejected():
    inputs, results = example(1.2, True)
    with pytest.raises(ValueError):
        evaluate_probabilities(inputs, results)
