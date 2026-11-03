import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yudun.pipeline import read_jsonl
from yudun.risk_evaluation import evaluate_probabilities


def main():
    parser = argparse.ArgumentParser(description="Evaluate all eligible pair probabilities using independent future GT")
    parser.add_argument("input")
    parser.add_argument("results")
    parser.add_argument("--output", default="outputs/risk_reliability.json")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--max-gt-gap", type=float, default=.2)
    args = parser.parse_args()
    report = evaluate_probabilities(list(read_jsonl(args.input)), list(read_jsonl(args.results)), args.bins, args.max_gt_gap)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
