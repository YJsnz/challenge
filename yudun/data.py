"""Dataset commands, independent of Qt/ROS and never executed automatically."""
import argparse
import json
from . import APP_NAME


def main():
    parser = argparse.ArgumentParser(description=f"{APP_NAME} · 公开数据接入与工业检测训练工具")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="stream SmartSpaces GT into reference-only JSONL")
    prepare.add_argument("spec")
    prepare.add_argument("--output", required=True)
    manifest = sub.add_parser("manifest", help="group scene specs using their declared source splits")
    manifest.add_argument("specs", nargs="+")
    manifest.add_argument("--output", required=True)
    infer = sub.add_parser("infer", help="generate camera observations from ONNX inference on video")
    infer.add_argument("spec")
    infer.add_argument("reference")
    infer.add_argument("--model", required=True)
    infer.add_argument("--metadata", required=True)
    infer.add_argument("--output", required=True)
    export = sub.add_parser("export-yolo", help="export grouped train/validation/test images and labels")
    export.add_argument("manifest")
    export.add_argument("--output", required=True)
    export.add_argument("--sample-every", type=int, default=10)
    export.add_argument("--max-images-per-sequence", type=int)
    train = sub.add_parser("train", help="train explicitly supplied local YOLOv8 detector weights")
    train.add_argument("dataset")
    train.add_argument("--weights", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--input-size", type=int, default=640)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--device", default="cpu")
    onnx = sub.add_parser("export-onnx", help="export raw YOLOv8 ONNX with hashed class metadata")
    onnx.add_argument("weights")
    onnx.add_argument("training_metadata")
    onnx.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command in ("prepare", "infer"):
        from .datasets import smartspaces
        if args.command == "prepare":
            result = smartspaces.prepare(args.spec, args.output)
        else:
            result = smartspaces.infer(args.spec, args.reference, args.model, args.metadata, args.output)
    else:
        from .datasets import training
        if args.command == "manifest":
            result = training.make_manifest(args.specs, args.output)
        elif args.command == "export-yolo":
            result = training.export_yolo(args.manifest, args.output, args.sample_every, args.max_images_per_sequence)
        elif args.command == "train":
            result = training.train_detector(args.dataset, args.weights, args.output, args.epochs,
                                            args.input_size, args.batch, args.seed, args.device)
        else:
            result = training.export_onnx(args.weights, args.training_metadata, args.output)
    print(json.dumps({"command": args.command, "result": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
