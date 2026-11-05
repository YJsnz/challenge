#!/usr/bin/env bash
# Launch the original Warehouse_026 recording with the native Qt panel.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

start_frame=1335
check=false
for option in "$@"; do
    case "$option" in
        --full) start_frame=0 ;;
        --check) check=true ;;
        -h|--help)
            printf 'Usage: bash scripts/launch_demo.sh [--full] [--check]\n'
            printf 'Starts Warehouse_026 at frame 1335. --full starts at frame 0. --check runs one frame without opening Qt.\n'
            exit 0 ;;
        *) printf 'Unknown option: %s\n' "$option" >&2; exit 2 ;;
    esac
done

python="${YUDUN_PYTHON:-$HOME/.venvs/yudun/bin/python}"
if [[ ! -x "$python" ]]; then
    printf 'Missing Linux Python environment: %s\nSee docs/WSL.md or set YUDUN_PYTHON.\n' "$python" >&2
    exit 1
fi
scene=datasets/raw/MTMC_Tracking_2026/test/Warehouse_026
video="$scene/videos/Camera_0000.mp4"
model=models/deployment/yolo11s-forklift-person.onnx
for asset in "$video" "$model" "$scene/calibration.json"; do
    if [[ ! -s "$asset" ]]; then
        printf 'Missing demo asset: %s\nSee docs/RESOURCE_DOWNLOADS.md.\n' "$asset" >&2
        exit 1
    fi
done

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
"$python" -c 'import cv2; from PyQt5 import QtWidgets'
mkdir -p outputs/quick-demo
run_dir="$(mktemp -d outputs/quick-demo/run.XXXXXX)"
bundle="$run_dir/scene"
"$python" -m yudun.map3d demo --output "$bundle" --source-calibration "$scene/calibration.json" --model "$model" > "$run_dir/manifest.json"
printf 'Demo scene: %s\nStarting source frame: %s\n' "$bundle" "$start_frame"

if "$check"; then
    "$python" - "$video" "$bundle" "$start_frame" <<'PY'
from contextlib import closing
from pathlib import Path
import sys
from yudun.camera import camera_frames
from yudun.pipeline import load_config

video, directory, index = sys.argv[1:]
scene = Path(directory)
with closing(camera_frames(video, scene/"camera.example.json", load_config(scene/"tracking_scene.json"),
                           allow_example=True, start_frame=int(index))) as stream:
    frame, result, _ = next(stream)
assert frame["source_frame"] == int(index) and result["map_3d"]["example_only"]
print(f"Check passed: frame {index}, {len(frame['visual_detections'])} detections, registered demo map")
PY
else
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
    exec "$python" -m yudun gui --video "$video" --calibration "$bundle/camera.example.json" \
        --allow-demo-calibration --start-frame "$start_frame"
fi
