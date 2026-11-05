import argparse
import json
from pathlib import Path
from . import APP_NAME
from .pipeline import load_config, replay, read_jsonl
from .simulation import generate, SCENARIOS
from .tracking import MODES
from .evaluation import save_run


def main():
    parser = argparse.ArgumentParser(description=f"{APP_NAME} · 固定单目相机定位与辅助预警研究原型")
    parser.add_argument("--config", help="scene JSON (default: packaged warehouse scene)")
    sub = parser.add_subparsers(dest="command", required=True)
    simulate = sub.add_parser("simulate", help="generate explicitly synthetic inputs")
    simulate.add_argument("--scenario", choices=SCENARIOS, default="occluded")
    simulate.add_argument("--seed", type=int, default=7)
    simulate.add_argument("--output", default="datasets/generated/demo.jsonl")
    run = sub.add_parser("run", help="offline tracking and evaluation")
    run.add_argument("input")
    run.add_argument("--mode", choices=MODES, default="monocular")
    run.add_argument("--seed", type=int, default=17)
    run.add_argument("--output", default="outputs/run")
    bench = sub.add_parser("benchmark", help="synthetic regression matrix, not B3 evidence")
    bench.add_argument("--repeats", type=int, default=2)
    bench.add_argument("--output", default="outputs/benchmark")
    gui = sub.add_parser("gui", help="native Qt panel adapted from the monocular project")
    sources = gui.add_mutually_exclusive_group()
    sources.add_argument("--input", help="optional external JSONL replay")
    sources.add_argument("--ros", action="store_true", help="subscribe to one ROS 2 camera")
    sources.add_argument("--video", help="run inference on a calibrated local video")
    sources.add_argument("--camera-index", type=int, help="run inference on a USB camera index")
    gui.add_argument("--calibration", help="fixed camera model and calibration JSON")
    gui.add_argument("--map-scene", help="3D map specification JSON")
    gui.add_argument("--allow-demo-calibration", action="store_true", help="allow an illustrative calibration for a local recording")
    gui.add_argument("--start-frame", type=int, default=0, help="start a local video at a zero-based frame index")
    baseline = sub.add_parser("import-baseline", help="import explicitly mapped recorded upstream CSV outputs")
    baseline.add_argument("input")
    baseline.add_argument("csv")
    baseline.add_argument("mapping")
    baseline.add_argument("--output", required=True)
    evaluation = sub.add_parser("evaluate", help="evaluate aligned outputs, including imported baselines")
    evaluation.add_argument("input")
    evaluation.add_argument("results")
    evaluation.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "simulate":
        frames = generate(config, args.scenario, args.seed)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(f, ensure_ascii=False)+"\n" for f in frames), encoding="utf-8")
        print(f"Synthetic frames: {len(frames)} -> {path.resolve()}")
    elif args.command == "run":
        frames = list(read_jsonl(args.input))
        results = replay(frames, config, args.mode, args.seed)
        print(json.dumps(save_run(args.output, frames, results, config, args.seed), indent=2, ensure_ascii=False))
    elif args.command == "benchmark":
        if not 1 <= args.repeats <= 100:
            parser.error("repeats must be 1..100")
        summary = []
        for scenario in SCENARIOS:
            for seed in range(args.repeats):
                frames = generate(config, scenario, seed)
                for mode in MODES:
                    results = replay(frames, config, mode, seed)
                    metrics = save_run(Path(args.output)/scenario/str(seed)/mode, frames, results, config, seed)
                    summary.append({"scenario": scenario, "mode": mode, **metrics})
            print(f"Completed {scenario}", flush=True)
        Path(args.output, "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    elif args.command == "gui":
        from .qt.main import launch
        if (args.video is not None or args.camera_index is not None) and not args.calibration:
            parser.error("direct video/USB inference requires --calibration")
        if args.allow_demo_calibration and args.video is None:
            parser.error("--allow-demo-calibration requires --video")
        if args.start_frame < 0 or (args.start_frame and args.video is None):
            parser.error("--start-frame requires --video and a nonnegative index")
        if args.map_scene:
            config["map_scene"] = str(Path(args.map_scene).resolve())
        return launch(config, args.input, args.ros, args.calibration,
                      args.video if args.video is not None else args.camera_index,
                      args.allow_demo_calibration, args.start_frame)
    elif args.command == "import-baseline":
        from .baseline import import_tracks
        print(f"Imported {import_tracks(args.input, args.csv, args.mapping, args.output)} baseline frames")
    elif args.command == "evaluate":
        from .evaluation import evaluate
        report = evaluate(list(read_jsonl(args.input)), list(read_jsonl(args.results)))
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
