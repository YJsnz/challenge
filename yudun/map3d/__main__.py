import argparse
import json
from pathlib import Path

from ..pipeline import read_jsonl
from .cloud import load_cloud
from .demo import make_demo
from .export import MapTrackWriter
from .scene import MapScene
from .workflows import calibrate, check_camera, register_map, write


def inspect(path):
    if path.suffix.lower() == ".json":
        scene = MapScene(path)
        cloud, total = scene.cloud, scene.original_points
        metadata = {"frame_id": scene.frame_id, "registered": scene.registered,
                    "example_only": scene.example_only, "units": "m"}
    else:
        cloud, _, total = load_cloud(path)
        metadata = {"registered": False, "units": "source"}
    return {**metadata, "source_points": total, "sampled_points": len(cloud),
            "sample_bounds": [cloud.min(axis=0).tolist(), cloud.max(axis=0).tolist()]}


def project(results, map_path, output):
    scene = MapScene(map_path, load_points=False)
    output.mkdir(parents=True, exist_ok=False)
    with (output/"results.jsonl").open("x", encoding="utf-8") as stream, \
         (output/"map_tracks.csv").open("x", newline="", encoding="utf-8-sig") as csv:
        writer = MapTrackWriter(csv)
        for frame in read_jsonl(results):
            mapped = scene.project_result(frame)
            stream.write(json.dumps(mapped, ensure_ascii=False, allow_nan=False)+"\n")
            writer.write(mapped)


def main():
    parser = argparse.ArgumentParser(description="Metric 3D map registration, camera calibration and replay")
    commands = parser.add_subparsers(dest="command", required=True)
    info = commands.add_parser("inspect")
    info.add_argument("path", type=Path)
    for name in ("register", "calibrate", "check"):
        command = commands.add_parser(name)
        command.add_argument("request", type=Path)
        command.add_argument("--output", required=True, type=Path)
    export = commands.add_parser("project")
    export.add_argument("results", type=Path)
    export.add_argument("--map-scene", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    demo = commands.add_parser("demo")
    demo.add_argument("--output", required=True, type=Path)
    demo.add_argument("--source-calibration", type=Path, help="Warehouse_026 publisher calibration")
    demo.add_argument("--model", type=Path, help="yolo11s forklift/person ONNX")
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            print(json.dumps(inspect(args.path), indent=2))
        elif args.command == "register":
            print(register_map(args.request, args.output))
        elif args.command == "calibrate":
            print(calibrate(args.request, args.output))
        elif args.command == "check":
            report = check_camera(args.request)
            write(args.output, report)
            print(json.dumps(report, indent=2))
        elif args.command == "project":
            project(args.results, args.map_scene, args.output)
            print(args.output.resolve())
        elif args.command == "demo":
            print(json.dumps(make_demo(args.output, args.source_calibration, args.model), indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"map3d: {exc}\n")


if __name__ == "__main__":
    main()
