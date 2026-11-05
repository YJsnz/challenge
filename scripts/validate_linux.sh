#!/usr/bin/env bash
# Run inside Linux or WSL, with the project environment already installed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python="${PYTHON:-python3}"
output="${1:-outputs/linux-validation/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$output"
output="$(realpath "$output")"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"

"$python" - "$output/environment.json" <<'PY'
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

if sys.platform != "linux":
    raise SystemExit("Run this validation inside Linux or WSL")
versions = {name: importlib.metadata.version(name) for name in
            ("yudun", "numpy", "scipy", "PyQt5", "opencv-python", "pytest", "ijson")}
record = {"platform": platform.platform(), "python": sys.version,
          "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
          "versions": versions}
Path(sys.argv[1]).write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
PY
"$python" -m pip check | tee "$output/pip-check.log"
"$python" -m pip freeze --all > "$output/requirements.txt"
"$python" -m pytest -q -ra --junitxml="$output/pytest.xml" --basetemp="$output/pytest-temp" | tee "$output/pytest.log"
"$python" -m yudun simulate --scenario camera_dropout --output "$output/camera-dropout.jsonl" | tee "$output/simulate.log"
"$python" -m yudun run "$output/camera-dropout.jsonl" --mode monocular --output "$output/replay" | tee "$output/replay.log"
"$python" -m yudun benchmark --repeats 2 --output "$output/benchmark" | tee "$output/benchmark.log"
"$python" -m build --no-isolation --outdir "$output/dist" | tee "$output/build.log"

# Import the built wheel outside the checkout, so source imports cannot mask
# missing package data in the distributable.
"$python" -m pip install --no-deps --target "$output/installed" "$output"/dist/*.whl | tee "$output/wheel-install.log"
export PYTHONPATH="$output/installed"
cd "$output"
"$python" - <<'PY' | tee wheel-smoke.log
import os
from importlib.resources import files
from pathlib import Path
import yudun
from yudun.pipeline import load_config

assert Path(yudun.__file__).resolve().is_relative_to(Path(os.environ["PYTHONPATH"]))
assert load_config()["sensor_positions"].keys() == {"camera"}
assert files("yudun").joinpath("qt/vendor/LICENSE").is_file()
assert files("yudun").joinpath("assets/warehouse026-demo-plan.png").is_file()
assert files("yudun").joinpath("assets/warehouse026-demo-layout.json").is_file()
assert files("yudun").joinpath("assets/zhaowei-logo.png").is_file()
print("Built wheel imports with its camera-only default scene, Qt license and map demo assets")
PY
"$python" -m yudun --help > wheel-cli-help.txt
"$python" -m yudun.map3d --help > wheel-map-cli-help.txt
printf 'Linux validation completed: %s\n' "$output"
