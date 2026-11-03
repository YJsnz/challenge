import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment


def evaluate(inputs, results, match_distance=1.5):
    """Global identity assignment; GT comes only from independent input annotations."""
    errors, associations, switches, previous = [], {}, 0, {}
    gt_total = pred_total = matches = 0
    future_errors = {h: [] for h in (0.5, 1., 2.)}
    truth_at = {round(f["timestamp"], 6): {o["id"]: o for o in f.get("ground_truth", [])} for f in inputs}
    for frame, result in zip(inputs, results):
        gt, tracks = frame.get("ground_truth", []), result["tracks"]
        gt_total += len(gt)
        pred_total += len(tracks)
        if not gt or not tracks:
            continue
        distances = np.array([[np.linalg.norm(np.array(g["position"])-t["state"][:2]) for t in tracks] for g in gt])
        gated = np.where(distances <= match_distance, distances, 1e6)
        rows, cols = linear_sum_assignment(gated)
        for i, j in zip(rows, cols):
            if distances[i, j] > match_distance:
                continue
            g, track = gt[i], tracks[j]
            errors.append(float(distances[i, j]))
            matches += 1
            pair = (g["id"], track["id"])
            associations[pair] = associations.get(pair, 0) + 1
            if g["id"] in previous and previous[g["id"]] != track["id"]:
                switches += 1
            previous[g["id"]] = track["id"]
            for forecast in track["predictions"]:
                h = forecast["dt"]
                target = truth_at.get(round(frame["timestamp"]+h, 6), {}).get(g["id"])
                if h in future_errors and target:
                    future_errors[h].append(float(np.linalg.norm(np.array(forecast["mean"])-target["position"])))
    gt_ids = sorted({p[0] for p in associations})
    track_ids = sorted({p[1] for p in associations})
    counts = np.array([[associations.get((g, t), 0) for t in track_ids] for g in gt_ids])
    idtp = 0
    if counts.size:
        rows, cols = linear_sum_assignment(-counts)
        idtp = int(counts[rows, cols].sum())
    latency = [r["processing_ms"] for r in results]
    return {"frames": len(results), "ground_truth_samples": gt_total, "matched_samples": matches,
            "provenance": sorted({f.get("provenance", "external_unverified") for f in inputs}),
            "position_rmse_m": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            "position_p95_m": float(np.percentile(errors, 95)) if errors else None,
            "detection_recall": matches/gt_total if gt_total else None,
            "idf1": 2*idtp/(gt_total+pred_total) if gt_total+pred_total else None,
            "id_switches": switches,
            "forecast_displacement_m": {str(h): {"fde": float(np.mean(v)) if v else None, "samples": len(v)} for h, v in future_errors.items()},
            "processing_p50_ms": float(np.percentile(latency, 50)) if latency else None,
            "processing_p95_ms": float(np.percentile(latency, 95)) if latency else None,
            "risk_event_starts": sum(e["transition"] == "START" for r in results for e in r["events"]),
            "notes": "Processing latency excludes capture, inference, transport and UI. Risk recall/lead time require independently annotated dangerous-event intervals; not inferred from model predictions. HOTA not computed."}


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
