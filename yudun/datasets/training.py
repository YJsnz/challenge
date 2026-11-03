"""Sequence-grouped YOLO exports and explicit local-weight training/export jobs."""
import json
from pathlib import Path
from .smartspaces import (CLASS_ORDER, SceneSpec, iter_gt, read_json, reference_frame,
                          scene_group, sha256, video_frames, write_json)


def make_manifest(spec_paths, output):
    splits = {name: [] for name in ("train", "validation", "test")}
    groups, identities = {}, set()
    for path in spec_paths:
        spec = SceneSpec(path)
        source_split = spec.data.get("source_split")
        if source_split not in ("train", "val", "test"):
            raise ValueError("each spec must declare source_split=train/val/test")
        name = "validation" if source_split == "val" else source_split
        group = scene_group(spec.data["release"], spec.data["scene_id"])
        if group in groups and groups[group] != name:
            raise ValueError(f"original/augmented scene leakage: {group}")
        if spec.sequence_id in identities:
            raise ValueError("duplicate sequence spec")
        groups[group] = name
        identities.add(spec.sequence_id)
        splits[name].append({"sequence_id": spec.sequence_id, "spec": str(spec.path),
                             "scene_group": group, "provenance": f"smartspaces_{spec.data['domain']}"})
    if not splits["train"] or not splits["validation"]:
        raise ValueError("provide at least one training and one independent validation scene")
    output = Path(output)
    if output.exists():
        raise FileExistsError("split manifest already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {"schema_version": 1, "splits": splits})
    return {name: len(rows) for name, rows in splits.items()}


def validate_splits(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    splits = read_json(manifest_path)["splits"]
    if set(splits) != {"train", "validation", "test"} or not splits["train"] or not splits["validation"]:
        raise ValueError("manifest requires nonempty train/validation plus a test list")
    groups, sequence_ids, jobs = {}, set(), []
    for split_name, records in splits.items():
        for record in records:
            spec = SceneSpec((manifest_path.parent/record["spec"]).resolve())
            group = scene_group(spec.data["release"], spec.data["scene_id"])
            if record["scene_group"] != group:
                raise ValueError(f"scene_group must be {group}, including original/augmented scenes")
            if group in groups and groups[group] != split_name:
                raise ValueError(f"scene group leaks across splits: {group}")
            if spec.sequence_id in sequence_ids:
                raise ValueError(f"duplicate sequence: {spec.sequence_id}")
            groups[group] = split_name
            sequence_ids.add(spec.sequence_id)
            jobs.append((split_name, spec))
    return jobs


def export_yolo(manifest_path, output, sample_every=10, max_images_per_sequence=None):
    import cv2
    if sample_every < 1 or max_images_per_sequence is not None and max_images_per_sequence < 1:
        raise ValueError("sampling limits must be positive")
    jobs = validate_splits(manifest_path)
    categories = [name for name in CLASS_ORDER if any(name in s.class_map.values() for _, s in jobs)]
    output = Path(output).resolve()
    staging = output.with_name(output.name+".partial")
    if output.exists() or staging.exists():
        raise FileExistsError("dataset output/partial exists; choose a new output path")
    staging.mkdir(parents=True)
    stats = {name: {"images": 0, "boxes": 0, "negative_images": 0} for name in ("train", "validation", "test")}
    source_metadata = []
    with (staging/"samples.jsonl").open("w", encoding="utf-8") as samples:
        for split_name, spec in jobs:
            image_dir, label_dir = staging/"images"/split_name, staging/"labels"/split_name
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            start, end = spec.data["start_frame"], spec.data["end_frame"]
            step = spec.data["frame_stride"]*sample_every
            if max_images_per_sequence is not None:
                end = min(end, start+step*max_images_per_sequence)
            annotations = iter_gt(spec.gt_path, start, end, step)
            # Independent GT and video iterators preserve exact source frame indexes.
            indices = range(start, end, step)
            for (index, objects), (video_index, image) in zip(annotations, video_frames(spec.video, indices, spec.camera)):
                if index != video_index:
                    raise ValueError("GT/video index mismatch")
                reference = reference_frame(spec, index, objects)
                basename = f"{spec.data['release']}_{spec.data['scene_id']}_{spec.camera.id}_{index:08d}"
                label_lines = []
                for label in reference["reference_boxes"]:
                    x1, y1, x2, y2 = label["bbox"]
                    box = [(x1+x2)/(2*spec.camera.width), (y1+y2)/(2*spec.camera.height),
                           (x2-x1)/spec.camera.width, (y2-y1)/spec.camera.height]
                    label_lines.append(f"{categories.index(label['category'])} "+" ".join(f"{x:.9f}" for x in box))
                if not cv2.imwrite(str(image_dir/(basename+".jpg")), image):
                    raise OSError(f"failed to write image {basename}")
                (label_dir/(basename+".txt")).write_text("\n".join(label_lines)+("\n" if label_lines else ""), encoding="utf-8")
                stats[split_name]["images"] += 1
                stats[split_name]["boxes"] += len(label_lines)
                stats[split_name]["negative_images"] += not label_lines
                samples.write(json.dumps({"image": f"images/{split_name}/{basename}.jpg", "sequence_id": spec.sequence_id,
                              "scene_group": scene_group(spec.data["release"], spec.data["scene_id"]),
                              "source_frame": index, "domain": spec.data["domain"]}, ensure_ascii=False)+"\n")
            source_metadata.append({"sequence_id": spec.sequence_id, "spec_sha256": sha256(spec.path),
                                    "video_sha256": sha256(spec.video), "ground_truth_sha256": sha256(spec.gt_path),
                                    "calibration_sha256": sha256(spec.calibration_path), "class_map": spec.class_map})
    if not stats["train"]["images"] or not stats["validation"]["images"]:
        raise ValueError("train/validation exports must contain images")
    yaml = f"path: {json.dumps(str(output))}\ntrain: images/train\nval: images/validation\n"
    if stats["test"]["images"]:
        yaml += "test: images/test\n"
    yaml += "names: "+json.dumps(categories)+"\n"
    (staging/"data.yaml").write_text(yaml, encoding="utf-8")
    write_json(staging/"dataset.json", {"schema_version": 1, "class_names": categories, "stats": stats,
               "manifest_sha256": sha256(manifest_path), "sources": source_metadata,
               "sample_every": sample_every, "license": "CC-BY-4.0",
               "source_url": "https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces"})
    staging.replace(output)
    return stats


def train_detector(dataset, weights, output, epochs=50, input_size=640, batch=8, seed=17, device="cpu"):
    """Invoked explicitly on Linux; require local initialization weights, no guessed model."""
    from ultralytics import YOLO
    dataset, weights, output = Path(dataset).resolve(), Path(weights).resolve(), Path(output).resolve()
    if not weights.is_file() or weights.suffix != ".pt":
        raise ValueError("supply an existing local YOLOv8 detection .pt checkpoint")
    if output.exists():
        raise FileExistsError("training output already exists; choose a new directory")
    if epochs < 1 or input_size < 32 or input_size % 32 or batch < 1:
        raise ValueError("invalid epochs, batch or input size")
    details = read_json(dataset/"dataset.json")
    if details.get("schema_version") != 1 or not set(details["class_names"]) <= set(CLASS_ORDER):
        raise ValueError("invalid industrial dataset metadata")
    model = YOLO(str(weights), task="detect")
    if model.task != "detect" or getattr(model.model.model[-1], "end2end", False):
        raise ValueError("use a raw one-to-many detection checkpoint, such as YOLOv8; end-to-end heads are unsupported")
    model.train(data=str(dataset/"data.yaml"), epochs=epochs, imgsz=input_size, batch=batch,
                         seed=seed, deterministic=True, device=device, project=str(output.parent),
                         name=output.name, exist_ok=False, pretrained=True)
    # Resolve the checkpoint reported by the trainer rather than assuming a numbered run path.
    best = Path(model.trainer.best)
    if not best.is_file():
        raise RuntimeError("trainer did not produce a best checkpoint")
    write_json(Path(model.trainer.save_dir)/"yudun_training.json", {"schema_version": 1,
               "dataset_metadata_sha256": sha256(dataset/"dataset.json"), "initial_weights_sha256": sha256(weights),
               "best_weights_sha256": sha256(best), "class_names": details["class_names"],
               "epochs_requested": epochs, "input_size": input_size, "seed": seed,
               "device": device, "batch": batch})
    return str(best)


def export_onnx(weights, training_metadata, output):
    from ultralytics import YOLO
    import shutil
    weights, output = Path(weights).resolve(), Path(output).resolve()
    metadata = read_json(training_metadata)
    if not weights.is_file() or sha256(weights) != metadata["best_weights_sha256"]:
        raise ValueError("checkpoint and training metadata do not match")
    if output.exists() or output.with_suffix(output.suffix+".json").exists():
        raise FileExistsError("ONNX output or sidecar already exists")
    model = YOLO(str(weights), task="detect")
    if model.task != "detect" or getattr(model.model.model[-1], "end2end", False):
        raise ValueError("only raw one-to-many detection heads are supported")
    names = [model.names[i] for i in range(len(model.names))]
    if names != metadata["class_names"]:
        raise ValueError("checkpoint classes differ from training metadata")
    exported = Path(model.export(format="onnx", imgsz=metadata["input_size"], batch=1, dynamic=False,
                                 simplify=False, nms=False, half=False, device="cpu"))
    import onnx
    graph = onnx.load(str(exported)).graph
    shapes = [[d.dim_value for d in item.type.tensor_type.shape.dim] for item in graph.output]
    if len(shapes) != 1 or len(shapes[0]) != 3 or shapes[0][0] != 1 or 4+len(names) not in shapes[0][1:]:
        raise ValueError(f"exported output is incompatible with the deployment adapter: {shapes}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != output:
        shutil.copyfile(exported, output)
    write_json(output.with_suffix(output.suffix+".json"), {"schema_version": 1, "format": "yolov8_detect_raw",
               "class_names": names, "input_size": metadata["input_size"], "onnx_sha256": sha256(output),
               "checkpoint_sha256": sha256(weights), "training_metadata_sha256": sha256(training_metadata)})
    return str(output)
