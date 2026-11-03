import argparse
import json
from pathlib import Path
from .pipeline import load_config, replay, read_jsonl
from .simulation import generate, SCENARIOS
from .tracking import MODES
from .evaluation import save_run


def main():
    parser = argparse.ArgumentParser(description="域盾 · 多模态空间感知与辅助预警研究原型")
    parser.add_argument("--config", help="scene JSON (default: packaged warehouse scene)")
    sub = parser.add_subparsers(dest="command", required=True)
    simulate = sub.add_parser("simulate", help="generate explicitly synthetic inputs")
    simulate.add_argument("--scenario", choices=SCENARIOS, default="occluded")
    simulate.add_argument("--seed", type=int, default=7)
    simulate.add_argument("--output", default="datasets/generated/demo.jsonl")
    run = sub.add_parser("run", help="offline tracking and evaluation")
    run.add_argument("input")
    run.add_argument("--mode", choices=MODES, default="full")
    run.add_argument("--seed", type=int, default=17)
    run.add_argument("--output", default="outputs/run")
    bench = sub.add_parser("benchmark", help="synthetic regression matrix, not B3 evidence")
    bench.add_argument("--repeats", type=int, default=2)
    bench.add_argument("--output", default="outputs/benchmark")
    gui = sub.add_parser("gui", help="native Qt panel adapted from the monocular project")
    gui.add_argument("--input", help="optional external JSONL replay")
    gui.add_argument("--ros", action="store_true", help="subscribe to ROS 2 raw sensor topics")
    gui.add_argument("--calibration", help="real camera and LiDAR calibration JSON")
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
        launch(config, args.input, args.ros, args.calibration)
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
