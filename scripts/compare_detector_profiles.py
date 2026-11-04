"""Compare fixed frames without inventing detection GT or accuracy metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import cv2
from yudun.adapters import OnnxDetector


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    if sys.platform != 'linux':
        raise SystemExit('Run in Linux or WSL')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--profile', default='configs/detector.quality.json')
    parser.add_argument('--frames', default='0,450,900,300,750,1350')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    indices = [int(index) for index in args.frames.split(',')]
    if not indices or min(indices) < 0 or len(set(indices)) != len(indices):
        parser.error('frame indices must be unique and nonnegative')
    metadata = json.loads(Path(args.metadata).read_text(encoding='utf-8-sig'))
    model_hash = digest(args.model)
    if model_hash != metadata['onnx_sha256']:
        raise ValueError('model and metadata hash mismatch')
    profile = json.loads(Path(args.profile).read_text(encoding='utf-8-sig'))
    profiles = {'baseline': {}, 'class_thresholds': {'class_confidence': profile['class_confidence']},
                'quality': profile}
    detectors = {name: OnnxDetector(args.model, metadata['class_names'], metadata['input_size'], **options)
                 for name, options in profiles.items()}
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('choose a new comparison output directory')
    output.mkdir(parents=True)
    report = {'schema_version': 1, 'commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
              'platform': platform.platform(), 'opencv': cv2.__version__, 'video': str(Path(args.video).resolve()),
              'video_sha256': digest(args.video), 'onnx_sha256': model_hash, 'metadata_sha256': digest(args.metadata),
              'profiles': profiles, 'frames': [], 'ground_truth_available': False,
              'note': 'Detection counts are not accuracy. Latency includes preprocessing, inference and NMS; excludes video decoding and UI.'}
    capture = cv2.VideoCapture(args.video)
    try:
        if not capture.isOpened():
            raise ValueError('video cannot be opened')
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if max(indices) >= count:
            raise ValueError(f'frame index exceeds video length ({count})')
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if not ok:
                raise ValueError(f'failed to decode frame {index}')
            record = {'source_frame': index, 'detections': {}}
            for name, detector in detectors.items():
                started = time.perf_counter()
                detections = detector.detect(image)
                elapsed = (time.perf_counter()-started)*1000
                record['detections'][name] = {'elapsed_ms': elapsed, 'objects': detections}
                preview = image.copy()
                for detection in detections:
                    x1,y1,x2,y2 = map(int, detection['bbox'])
                    colour = (0,200,0) if detection['category'] == 'person' else (0,180,255)
                    cv2.rectangle(preview, (x1,y1), (x2,y2), colour, 2)
                    cv2.putText(preview, f"{detection['category']} {detection['confidence']:.2f}",
                                (x1,max(20,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .55, colour, 2)
                if not cv2.imwrite(str(output/f'{name}-{index}.jpg'), preview):
                    raise OSError('failed to write preview')
                print(index, name, len(detections), f'{elapsed:.1f} ms', flush=True)
            report['frames'].append(record)
            (output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n', encoding='utf-8')
    finally:
        capture.release()


if __name__ == '__main__':
    main()
