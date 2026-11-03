"""Evaluate alert episodes against independent annotated GT event intervals."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import linear_sum_assignment
from yudun.pipeline import read_jsonl


def evaluate_events(inputs, results, annotations, minimum_level=2):
    truth = annotations["events"]
    for event in truth:
        if not event["start"] <= event["danger_time"] <= event["end"] or len(set(event["pair"])) != 2:
            raise ValueError("invalid independent event annotation")
    episodes, active = [], {}
    for frame, result in zip(inputs, results):
        gt, tracks = frame.get("ground_truth", []), result["tracks"]
        identity = {}
        if gt and tracks:
            distances = np.array([[np.linalg.norm(np.array(g["position"])-t["state"][:2]) for t in tracks] for g in gt])
            rows, cols = linear_sum_assignment(np.where(distances <= 1.5, distances, 1e6))
            identity = {tracks[j]["id"]: gt[i]["id"] for i, j in zip(rows, cols) if distances[i, j] <= 1.5}
        present = set()
        for event in result["events"]:
            key = event["event_id"]
            if event["level"] >= minimum_level:
                present.add(key)
                if key not in active:
                    pair = [identity.get(i) for i in event["pair"]]
                    active[key] = {"start": result["timestamp"], "end": result["timestamp"],
                                   "pair": sorted(pair) if all(pair) else None}
                active[key]["end"] = result["timestamp"]
        for key in list(active):
            if key not in present:
                episodes.append(active.pop(key))
    episodes.extend(active.values())
    matched, leads, false_alarms, unverifiable = set(), [], 0, 0
    for episode in episodes:
        if episode["pair"] is None:
            unverifiable += 1
            continue
        candidates = [(i, event) for i, event in enumerate(truth) if i not in matched
                      and sorted(event["pair"]) == episode["pair"]
                      and episode["start"] <= event["end"] and episode["end"] >= event["start"]]
        if candidates:
            index, event = min(candidates, key=lambda x: x[1]["danger_time"])
            matched.add(index)
            leads.append(event["danger_time"] - episode["start"])
        else:
            false_alarms += 1
    duration = inputs[-1]["timestamp"] - inputs[0]["timestamp"] if len(inputs) > 1 else 0
    return {"annotated_dangerous_events": len(truth), "matched_events": len(matched),
            "recall": len(matched)/len(truth) if truth else None,
            "false_alert_episodes": false_alarms, "unverifiable_alert_episodes": unverifiable,
            "false_alerts_per_hour": false_alarms*3600/duration if duration else None,
            "median_lead_seconds": float(np.median(leads)) if leads else None,
            "minimum_level": minimum_level, "episodes": episodes,
            "note": "Annotation window includes allowed early-warning interval; negative lead means late alert. Unverifiable identity episodes are reported separately, never counted as successful warnings."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("results")
    parser.add_argument("annotations")
    parser.add_argument("--output", default="outputs/event_metrics.json")
    args = parser.parse_args()
    result = evaluate_events(list(read_jsonl(args.input)), list(read_jsonl(args.results)),
                             json.loads(Path(args.annotations).read_text(encoding="utf-8")))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
