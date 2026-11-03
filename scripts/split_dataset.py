"""Split complete sequences by scene group; frames from a group never cross splits."""
import argparse
import json
from pathlib import Path
import random


def split(records, seed=17):
    groups = sorted({r["scene_group"] for r in records})
    if len(groups) < 3:
        raise ValueError("at least three independent scene groups are required")
    random.Random(seed).shuffle(groups)
    train_count = min(len(groups)-2, max(1, round(len(groups)*.6)))
    val_count = min(len(groups)-train_count-1, max(1, round(len(groups)*.2)))
    assignment = {g: "train" if i < train_count else "validation" if i < train_count+val_count else "test"
                  for i, g in enumerate(groups)}
    return {name: [r for r in records if assignment[r["scene_group"]] == name]
            for name in ("train", "validation", "test")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="JSON list of sequence_id, path, scene_group, provenance")
    parser.add_argument("--output", default="datasets/splits.json")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    records = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    ids = [r["sequence_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sequence_id")
    result = {"seed": args.seed, "splits": split(records, args.seed)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
