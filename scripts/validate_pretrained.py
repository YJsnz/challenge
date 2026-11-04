"""Linux CPU smoke checks for the pinned downloads; not an accuracy evaluation."""
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import onnx
from ultralytics import YOLO
from yudun.adapters import OnnxDetector

if sys.platform != 'linux':
    raise SystemExit('Run inside Linux or WSL')
root = Path(__file__).resolve().parents[1]
output = root/'outputs/wsl-validation/models'
output.mkdir(parents=True, exist_ok=True)
manifest = json.loads((root/'configs/resource_downloads.json').read_text())
class_map = {'forklift': 'forklift_or_cart', 'person': 'person'}
video = root/'datasets/raw/MTMC_Tracking_2026/test/Warehouse_026/videos/Camera_0000.mp4'
capture = cv2.VideoCapture(str(video))
assert capture.isOpened()
fps, total = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
frames = []
for index in (0, total//4, total//2):
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    success, frame = capture.read()
    assert success
    frames.append((index, frame))
capture.release()
report = {'video': str(video.relative_to(root)), 'fps': fps, 'frame_count': total,
          'video_shape': list(frames[0][1].shape), 'device': 'cpu', 'models': []}
for name in ('yolo11s-forklift-person', 'keremberke-yolov8n'):
    weights = root/'models/pretrained'/name/'best.pt'
    item = next(item for item in manifest if item['path'] == str(weights.relative_to(root)))
    checkpoint_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert checkpoint_hash == item['sha256']
    record = {'name': name, 'source_url': item['url'], 'checkpoint_sha256': checkpoint_hash}
    try:
        model = YOLO(str(weights), task='detect')
        names = [model.names[index] for index in range(len(model.names))]
        assert set(names) == set(class_map), names
        record['original_class_names'] = names
        exported = Path(model.export(format='onnx', imgsz=640, batch=1, dynamic=False,
                                     simplify=False, nms=False, half=False, device='cpu', opset=12))
        graph = onnx.load(str(exported))
        onnx.checker.check_model(graph)
        record['output_shapes'] = [[dim.dim_value for dim in value.type.tensor_type.shape.dim]
                                   for value in graph.graph.output]
        target = root/'models/deployment'/(name+'.onnx')
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(exported, target)
        metadata = {'schema_version': 1, 'format': 'yolov8_detect_raw',
                    'class_names': [class_map[label] for label in names], 'input_size': 640,
                    'onnx_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                    'checkpoint_sha256': checkpoint_hash, 'source_url': item['url'],
                    'provenance': 'external_pretrained; no project training performed'}
        target.with_suffix('.onnx.json').write_text(json.dumps(metadata, indent=2)+'\n')
        detector = OnnxDetector(target, metadata['class_names'])
        record['frames'] = []
        for index, frame in frames:
            start = time.perf_counter()
            detections = detector.detect(frame)
            elapsed = (time.perf_counter()-start)*1000
            record['frames'].append({'frame_index': index, 'inference_ms': elapsed,
                                     'detections': [{**detection, 'bbox': list(map(float, detection['bbox']))}
                                                    for detection in detections]})
            reference = model.predict(frame, device='cpu', imgsz=640, rect=False,
                                      conf=.4, iou=.45, verbose=False)[0].boxes.data.cpu().numpy()
            if len(reference) != len(detections):
                raise AssertionError('PyTorch and ONNX detection counts differ')
            remaining, differences = list(detections), []
            for x1, y1, x2, y2, confidence, label in reference:
                category = metadata['class_names'][int(label)]
                candidates = [detection for detection in remaining if detection['category'] == category]
                best = min(candidates, key=lambda detection: sum((a-b)**2 for a,b in
                           zip(detection['bbox'], (x1,y1,x2,y2))))
                difference = max(abs(float(a)-float(b)) for a,b in zip(best['bbox'], (x1,y1,x2,y2)))
                if difference >= .1 or abs(best['confidence']-float(confidence)) >= .001:
                    raise AssertionError('PyTorch and ONNX coordinates or confidence differ')
                differences.append(difference)
                remaining.remove(best)
            record['frames'][-1]['max_backend_box_difference_px'] = max(differences, default=0)
            preview = frame.copy()
            for detection in detections:
                x1,y1,x2,y2 = map(int, detection['bbox'])
                cv2.rectangle(preview, (x1,y1), (x2,y2), (0,220,60), 2)
                cv2.putText(preview, f"{detection['category']} {detection['confidence']:.2f}",
                            (x1,max(20,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,220,60), 2)
            assert cv2.imwrite(str(output/f'{name}-{index}.jpg'), preview)
        record['status'] = 'passed'
        record['onnx'] = str(target.relative_to(root))
    except Exception as error:
        record['status'] = 'failed'
        record['error'] = f'{type(error).__name__}: {error}'
    report['models'].append(record)
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(record, indent=2), flush=True)
if not all(model['status'] == 'passed' for model in report['models']):
    raise SystemExit('A pretrained model smoke check failed; inspect report.json')
