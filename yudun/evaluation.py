import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment


def validate_alignment(inputs, results):
    if len(inputs) != len(results):
        raise ValueError("input/result frame counts differ; refusing truncated evaluation")
    previous = -1.
    sequence = None
    for frame, result in zip(inputs, results):
        timestamp = frame["timestamp"]
        if not np.isfinite(timestamp) or timestamp <= previous or abs(timestamp-result["timestamp"]) > 1e-6:
            raise ValueError("evaluation requires matching, strictly increasing timestamps")
        previous = timestamp
        if "sequence_id" in frame:
            if sequence is not None and sequence != frame["sequence_id"]:
                raise ValueError("evaluate one sequence at a time")
            sequence = frame["sequence_id"]


def match_truth(gt, tracks, match_distance=1.5):
    if not gt or not tracks:
        return []
    distances = np.array([[np.linalg.norm(np.array(g["position"])-t["state"][:2]) for t in tracks] for g in gt])
    costs = np.where(distances <= match_distance, distances, 1e6)
    for i, g in enumerate(gt):
        for j, t in enumerate(tracks):
            if g.get("category") and t.get("category") and g["category"] != t["category"] and t["category"] != "unknown_dynamic_object":
                costs[i, j] = 1e6
    rows, cols = linear_sum_assignment(costs)
    return [(i, j, float(distances[i, j])) for i, j in zip(rows, cols) if costs[i, j] < 1e6]


def uncertainty_score(error, matrix):
    from .schema import covariance
    matrix = covariance(matrix, 2)
    mahalanobis = float(error @ np.linalg.solve(matrix, error))
    nll = .5*(2*np.log(2*np.pi)+np.linalg.slogdet(matrix)[1]+mahalanobis)
    return mahalanobis <= 5.991464547107979, float(nll)


def evaluate(inputs, results, match_distance=1.5):
    """Global identity assignment; GT comes only from independent input annotations."""
    validate_alignment(inputs, results)
    errors, associations, switches, previous = [], {}, 0, {}
    gt_total = pred_total = matches = 0
    future_errors = {h: [] for h in (0.5, 1., 2.)}
    ade = {h: [] for h in future_errors}
    forecast_coverage = {h: [] for h in future_errors}
    position_coverage, position_nll = [], []
    annotated_frames = 0
    truth_at = {round(f["timestamp"], 6): {o["id"]: o for o in f.get("ground_truth", [])} for f in inputs}
    for frame, result in zip(inputs, results):
        if "ground_truth" not in frame:
            continue  # Missing reference annotations are not known-empty scenes.
        annotated_frames += 1
        gt, tracks = frame.get("ground_truth", []), result["tracks"]
        gt_total += len(gt)
        pred_total += len(tracks)
        if not gt or not tracks:
            continue
        for i, j, distance in match_truth(gt, tracks, match_distance):
            g, track = gt[i], tracks[j]
            errors.append(distance)
            if "covariance" in track:
                covered, nll = uncertainty_score(np.asarray(g["position"])-track["state"][:2], np.asarray(track["covariance"])[:2, :2])
                position_coverage.append(covered)
                position_nll.append(nll)
            matches += 1
            pair = (g["id"], track["id"])
            associations[pair] = associations.get(pair, 0) + 1
            if g["id"] in previous and previous[g["id"]] != track["id"]:
                switches += 1
            previous[g["id"]] = track["id"]
            samples, coverage_samples = {}, {}
            for forecast in track.get("predictions", []):
                h = forecast["dt"]
                if h <= 0 or h in samples:
                    raise ValueError("forecast horizons must be positive and unique")
                target = truth_at.get(round(frame["timestamp"]+h, 6), {}).get(g["id"])
                error = np.asarray(forecast["mean"])-target["position"] if target else None
                samples[h] = float(np.linalg.norm(error)) if error is not None else None
                if error is not None and "covariance" in forecast:
                    coverage_samples[h] = uncertainty_score(error, forecast["covariance"])[0]
            for horizon in future_errors:
                # Fixed 10 Hz prediction grid, including the final horizon; no partial ADE.
                grid = [step/10 for step in range(1, round(horizon*10)+1)]
                if samples.get(horizon) is not None:
                    future_errors[horizon].append(samples[horizon])
                if all(samples.get(h) is not None for h in grid):
                    ade[horizon].append(float(np.mean([samples[h] for h in grid])))
                if horizon in coverage_samples:
                    forecast_coverage[horizon].append(coverage_samples[horizon])
    gt_ids = sorted({p[0] for p in associations})
    track_ids = sorted({p[1] for p in associations})
    counts = np.array([[associations.get((g, t), 0) for t in track_ids] for g in gt_ids])
    idtp = 0
    if counts.size:
        rows, cols = linear_sum_assignment(-counts)
        idtp = int(counts[rows, cols].sum())
    latency = [r["processing_ms"] for r in results if r.get("processing_ms") is not None]
    if not np.isfinite(latency).all() or any(x < 0 for x in latency):
        raise ValueError("processing timings must be finite and nonnegative")
    return {"frames": len(results), "annotated_frames": annotated_frames, "ground_truth_samples": gt_total, "matched_samples": matches,
            "provenance": sorted({f.get("provenance", "external_unverified") for f in inputs}),
            "position_rmse_m": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            "position_p95_m": float(np.percentile(errors, 95)) if errors else None,
            "detection_recall": matches/gt_total if gt_total else None,
            "idf1": 2*idtp/(gt_total+pred_total) if gt_total+pred_total else None,
            "id_switches": switches,
            "forecast_displacement_m": {str(h): {"fde": float(np.mean(v)) if v else None, "samples": len(v),
                    "ade": float(np.mean(ade[h])) if ade[h] else None, "ade_complete_samples": len(ade[h]),
                    "coverage_95": float(np.mean(forecast_coverage[h])) if forecast_coverage[h] else None} for h, v in future_errors.items()},
            "position_coverage_95": float(np.mean(position_coverage)) if position_coverage else None,
            "position_gaussian_nll": float(np.mean(position_nll)) if position_nll else None,
            "uncertainty_samples": len(position_coverage),
            "processing_p50_ms": float(np.percentile(latency, 50)) if latency else None,
            "processing_p95_ms": float(np.percentile(latency, 95)) if latency else None,
            "processing_timing_samples": len(latency),
            "risk_event_starts": sum(e["transition"] == "START" for r in results for e in r["events"]),
            "notes": "ADE uses complete 10 Hz forecast horizons; no GT interpolation. Position error and uncertainty coverage are conditional on spatial/category matching (gate 1.5 m by default). Processing latency excludes capture, inference, transport and UI. HOTA not computed; risk-event metrics require independent annotations."}


def save_run(directory, inputs, results, config, seed):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, frames in (("input.jsonl", inputs), ("results.jsonl", results)):
        with (directory/name).open("w", encoding="utf-8") as stream:
            for frame in frames:
                stream.write(json.dumps(frame, ensure_ascii=False, allow_nan=False)+"\n")
    config_json = json.dumps(config, sort_keys=True, ensure_ascii=False)
    (directory/"config.json").write_text(config_json, encoding="utf-8")
    report = evaluate(inputs, results)
    report.update({"seed": seed, "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
                   "input_sha256": hashlib.sha256((directory/"input.jsonl").read_bytes()).hexdigest(),
                   "version": "0.1.0", "mode": results[0]["mode"] if results else None})
    (directory/"metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (directory/"tracks.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp", "track_id", "category", "x_m", "y_m", "vx_mps", "vy_mps", "status", "age_s", "processing_ms"])
        for frame in results:
            for track in frame["tracks"]:
                writer.writerow([frame["timestamp"], track["id"], track["category"], *track["state"], track["status"], track["age"], frame["processing_ms"]])
    with (directory/"events.jsonl").open("w", encoding="utf-8") as stream:
        for frame in results:
            for event in frame["events"]:
                stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+"\n")
    with (directory/"latency.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp", "processing_ms"])
        writer.writerows((f["timestamp"], f["processing_ms"]) for f in results)
    return report
