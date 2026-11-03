"""Import recorded upstream outputs without rerunning or impersonating upstream B3."""
import csv
import json
from pathlib import Path
import numpy as np
from .datasets.smartspaces import read_json, sha256
from .pipeline import read_jsonl
from .schema import CLASSES


def import_tracks(input_path, csv_path, mapping_path, output):
    mapping = read_json(mapping_path)
    if mapping.get("schema_version") != 1:
        raise ValueError("unsupported CSV mapping version")
    transform = np.asarray(mapping["map_from_source_xy"], dtype=float)
    if transform.shape != (3, 3) or not np.isfinite(transform).all() or not np.allclose(transform[2], [0, 0, 1]):
        raise ValueError("expected finite affine 3x3 map_from_source_xy")
    if abs(np.linalg.det(transform[:2, :2])) < 1e-12:
        raise ValueError("coordinate transform is singular")
    time_scale, time_offset = float(mapping["time_scale_to_seconds"]), float(mapping["time_offset_seconds"])
    if not np.isfinite([time_scale, time_offset]).all() or time_scale <= 0:
        raise ValueError("invalid source clock conversion")
    inputs = list(read_jsonl(input_path))
    times = {round(f["timestamp"], 6): index for index, f in enumerate(inputs)}
    if len(times) != len(inputs) or any(b["timestamp"] <= a["timestamp"] for a, b in zip(inputs, inputs[1:])):
        raise ValueError("input timeline must be unique and increasing")
    columns = mapping["columns"]
    if not {"timestamp", "track_id", "x", "y"} <= columns.keys():
        raise ValueError("CSV mapping requires timestamp, track_id, x and y")
    results = [{"schema_version": 1, "frame_id": "map", "timestamp": f["timestamp"],
                "mode": "external_baseline", "provenance": mapping["source_name"], "tracks": [], "events": []} for f in inputs]
    identities, seen = {}, set()
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(columns.values()) <= set(reader.fieldnames or []):
            raise ValueError("CSV columns differ from mapping")
        for row in reader:
            timestamp = float(row[columns["timestamp"]])*time_scale+time_offset
            index = times.get(round(timestamp, 6))
            if index is None:
                raise ValueError(f"baseline timestamp {timestamp} is not in the reference timeline; align explicitly first")
            identity = row[columns["track_id"]]
            if not identity or (index, identity) in seen:
                raise ValueError("missing or duplicate track identity in a frame")
            seen.add((index, identity))
            identity_number = identities.setdefault(identity, len(identities)+1)
            point = transform @ [float(row[columns["x"]]), float(row[columns["y"]]), 1]
            if not np.isfinite(point).all():
                raise ValueError("non-finite baseline position")
            category = "unknown_dynamic_object"
            if "category" in columns:
                category = mapping["class_map"][row[columns["category"]]]
                if category not in CLASSES:
                    raise ValueError("CSV category must map explicitly to an industrial class")
            # Missing velocity/covariance must not become invented prediction evidence.
            track = {"id": identity_number, "source_id": identity, "category": category,
                     "state": [float(point[0]), float(point[1]), 0., 0.],
                     "velocity_available": False, "predictions": []}
            results[index]["tracks"].append(track)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("baseline output exists")
    with output.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+"\n")
    output.with_suffix(output.suffix+".manifest.json").write_text(json.dumps({"schema_version": 1,
        "input_sha256": sha256(input_path), "csv_sha256": sha256(csv_path), "mapping_sha256": sha256(mapping_path),
        "frames": len(results), "identities": len(identities), "source_name": mapping["source_name"],
        "note": "Recorded external outputs only. No runtime/forecast/covariance inferred. Timeline frames without CSV rows are no-output frames, not interpolated tracks. Source-name claim must be backed by the upstream run logs."}, indent=2), encoding="utf-8")
    return len(results)
