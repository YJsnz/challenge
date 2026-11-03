"""Conditional probability reliability diagnostics from independent future GT."""
from bisect import bisect_right
import math
import numpy as np
from .evaluation import match_truth, validate_alignment


def evaluate_probabilities(inputs, results, bins=10, max_gt_gap=.2):
    validate_alignment(inputs, results)
    if not isinstance(bins, int) or bins < 2 or max_gt_gap <= 0:
        raise ValueError("invalid reliability bins or maximum GT gap")
    timestamps = [f["timestamp"] for f in inputs]
    truth = [{g["id"]: g for g in f.get("ground_truth", [])} for f in inputs]
    probabilities, labels = [], []
    excluded = {"unmatched_identity": 0, "incomplete_future_gt": 0}
    missing_assessment_frames = 0
    for index, (frame, result) in enumerate(zip(inputs, results)):
        if "risk_assessments" not in result:
            missing_assessment_frames += 1
        gt, tracks = list(truth[index].values()), result["tracks"]
        identity = {tracks[j]["id"]: gt[i]["id"] for i, j, _ in match_truth(gt, tracks)}
        for assessment in result.get("risk_assessments", []):
            p, horizon, distance = assessment["probability"], assessment["horizon"], assessment["safe_distance"]
            if not all(math.isfinite(x) for x in (p, horizon, distance)) or not 0 <= p <= 1 or horizon <= 0 or distance <= 0:
                raise ValueError("invalid risk probability, horizon or distance")
            pair = [identity.get(i) for i in assessment["pair"]]
            if len(pair) != 2 or any(x is None for x in pair) or pair[0] == pair[1]:
                excluded["unmatched_identity"] += 1
                continue
            end_time = frame["timestamp"]+horizon
            stop = bisect_right(timestamps, end_time+1e-6)
            window_times = timestamps[index:stop]
            if (not window_times or abs(window_times[-1]-end_time) > 1e-6
                    or any(b-a > max_gt_gap+1e-6 for a, b in zip(window_times, window_times[1:]))
                    or any(pair[0] not in future or pair[1] not in future for future in truth[index:stop])):
                excluded["incomplete_future_gt"] += 1
                continue
            minimum = min(float(np.linalg.norm(np.asarray(future[pair[0]]["position"])-future[pair[1]]["position"]))
                          for future in truth[index:stop])
            probabilities.append(p)
            labels.append(int(minimum < distance))
    p, y = np.asarray(probabilities), np.asarray(labels)
    reliability, ece = [], 0.
    for index in range(bins):
        selected = (p >= index/bins) & (p <= 1 if index == bins-1 else p < (index+1)/bins)
        count = int(selected.sum())
        mean_p, observed = (float(p[selected].mean()), float(y[selected].mean())) if count else (None, None)
        if count:
            ece += count/len(p)*abs(mean_p-observed)
        reliability.append({"lower": index/bins, "upper": (index+1)/bins, "samples": count,
                            "mean_probability": mean_p, "observed_frequency": observed})
    return {"samples": len(p), "positive_samples": int(y.sum()),
            "brier_score": float(np.mean((p-y)**2)) if len(p) else None,
            "ece": ece if len(p) else None, "bins": reliability, "excluded": excluded,
            "missing_assessment_frames": missing_assessment_frames,
            "note": "Diagnostics, not a fitted calibrator. Conditional on tracked/matched eligible pairs and complete future GT. Labels use sampled GT separation and each assessment's declared safe distance; continuous-time collisions between GT frames are not verified. Unalerted eligible pairs are included."}
