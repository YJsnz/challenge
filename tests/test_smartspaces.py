import json
import numpy as np
import pytest
from yudun.datasets.smartspaces import Camera, SceneSpec, fields, clipped_bbox, reference_frame, scene_group, iter_gt, infer, sha256
from yudun.pipeline import Pipeline


def calibration():
    return {"sensors": [{"type": "camera", "id": "Camera_0000", "coordinates": {"x": 0, "y": 0},
                         "attributes": [{"name": "fps", "value": "30"}, {"name": "frameWidth", "value": "640"}, {"name": "frameHeight", "value": "480"}],
                         "homography": [[.01, 0, -3.2], [0, .01, -2.4], [0, 0, 1]]}]}


@pytest.fixture
def spec(tmp_path):
    (tmp_path/"calibration.json").write_text(json.dumps(calibration()))
    settings = {"schema_version": 1, "release": "MTMC_Tracking_2026", "scene_id": "Warehouse_000", "domain": "synthetic",
                "scene_directory": ".", "camera_id": "Camera_0000", "video": "videos/Camera_0000.mp4",
                "world_unit_to_m": .01, "map_origin_source": [100, 200], "roi_bounds_m": [-10, -10, 10, 10],
                "start_frame": 0, "end_frame": 6, "frame_stride": 3,
                "measurement_covariance_m2": [[.25, 0], [0, .25]], "class_map": {"Person": "person"}}
    path = tmp_path/"spec.json"
    path.write_text(json.dumps(settings))
    return SceneSpec(path)


def test_actual_space_keys_and_documented_underscore_keys():
    assert fields({"object type": "Person", "3d location": [0, 0, 0]}) == {"object_type": "Person", "3d_location": [0, 0, 0]}
    with pytest.raises(ValueError):
        fields({"object type": "Person", "object_type": "Person"})


def test_homography_direction_and_explicit_units(spec):
    camera = Camera.from_calibration(calibration(), "Camera_0000")
    np.testing.assert_allclose(camera.ground([420, 440]), [1, 2])
    np.testing.assert_allclose(spec.map_xy([200, 400, 3]), [1, 2])
    reverse = calibration()
    reverse['sensors'][0]['homography'] = [[100,0,320],[0,100,240],[0,0,1]]
    np.testing.assert_allclose(Camera.from_calibration(reverse,'Camera_0000','ground_to_image').ground([420,440]),[1,2])
    with pytest.raises(ValueError,match='homography_direction'):
        Camera.from_calibration(reverse,'Camera_0000','guess')


def test_published_warehouse026_homography_matches_intrinsics_and_extrinsics():
    # Numeric calibration facts from the pinned NVIDIA CC-BY-4.0 dataset.
    k = np.array([[999.5598304777992,0,960],[0,988.7114606584821,540],[0,0,1]])
    extrinsic = np.array([[-.15112446150792366,.9870559881787692,-.053683184225040696,.19325946321014653],
                         [.530628984951175,.03518355999453721,-.8468736846429169,3.0487452647145012],
                         [-.8340229670102792,-.15646918140090382,-.5290776078683733,1.9046793883261441]])
    homography = [[-.0002879703995466277,.0010222173909751242,-1.8640921155465346],
                  [.00188085306944875,.00006777851931885563,-2.140242963070378],
                  [.00002841507783552547,.0004531775983272045,.007928729107611037]]
    data = calibration()
    data['sensors'][0]['homography'] = homography
    camera = Camera.from_calibration(data,'Camera_0000')
    for x,y in [(-5,-1),(-10,2),(-3,0)]:
        pixel = k@extrinsic@np.array([x,y,0,1])
        np.testing.assert_allclose(camera.ground(pixel[:2]/pixel[2]),[x,y],atol=1e-10)


def test_reference_never_becomes_an_observation(spec):
    obj = {"object type": "Person", "object id": 7, "3d location": [200, 400, 10],
           "3d bounding box scale": [50, 60, 170], "2d bounding box visible": {"Camera_0000": [20, 10, 80, 90]}}
    frame = reference_frame(spec, 3, [obj])
    assert frame["timestamp"] == .1
    assert frame["observations"] == [] and frame["sensors"] == []
    assert frame["ground_truth"][0]["position"] == [1, 2]
    assert frame["reference_boxes"][0]["category"] == "person"
    with pytest.raises(ValueError, match="reference-only"):
        Pipeline().process(frame)


def test_bbox_clipping_and_augmented_scene_group():
    assert clipped_bbox([-10, 0, 700, 480], 640, 480) == [0, 0, 640, 480]
    assert clipped_bbox([5, 5, 5, 6], 640, 480) is None
    assert scene_group("MTMC_Tracking_2026", "Warehouse_001") == scene_group("MTMC_Tracking_2026", "Warehouse_000")
    assert scene_group("MTMC_Tracking_2025", "Warehouse_001") != scene_group("MTMC_Tracking_2025", "Warehouse_000")


def test_full_image_training_labels_include_objects_outside_tracking_roi(spec):
    obj = {"object type": "Person", "object id": 8, "3d location": [10000, 10000, 10],
           "3d bounding box scale": [50, 60, 170], "2d bounding box visible": {"Camera_0000": [20, 10, 80, 90]}}
    frame = reference_frame(spec, 0, [obj])
    assert frame["ground_truth"] == []
    assert len(frame["reference_boxes"]) == 1


def test_streaming_gt_gaps_and_bounds(tmp_path):
    pytest.importorskip("ijson")
    path = tmp_path/"gt.json"
    path.write_text(json.dumps({"0": [], "1": [], "2": [], "3": []}))
    assert [i for i, _ in iter_gt(path, 0, 4, 3)] == [0, 3]
    path.write_text(json.dumps({"0": [], "2": []}))
    with pytest.raises(ValueError, match="missing frame"):
        list(iter_gt(path, 0, 3, 1))


def test_inference_observations_come_from_model_not_reference_boxes(spec, tmp_path, monkeypatch):
    import yudun.adapters
    import yudun.datasets.smartspaces as adapter
    received_options = []
    class FakeDetector:
        def __init__(self, *args, **kwargs):
            received_options.append(kwargs)
        def detect(self, image):
            return [{"bbox": [320, 240, 420, 440], "category": "person", "confidence": .9}]
    monkeypatch.setattr(yudun.adapters, "OnnxDetector", FakeDetector)
    monkeypatch.setattr(adapter, "video_frames", lambda path, indices, camera: ((i, np.zeros((480, 640, 3), np.uint8)) for i in indices))
    reference = reference_frame(spec, 0, [{"object type": "Person", "object id": 7, "3d location": [200, 400, 10],
                "3d bounding box scale": [50, 60, 170], "2d bounding box visible": {"Camera_0000": [20, 10, 80, 90]}}])
    reference_path = tmp_path/"reference.jsonl"
    reference_path.write_text(json.dumps(reference)+"\n")
    spec.video.parent.mkdir()
    spec.video.write_bytes(b"mock video for provenance hash")
    model = tmp_path/"model.onnx"
    model.write_bytes(b"model mocked; never executed")
    metadata = tmp_path/"model.json"
    metadata.write_text(json.dumps({"schema_version": 1, "format": "yolov8_detect_raw", "class_names": ["person"], "input_size": 640, "onnx_sha256": sha256(model),
                                    "detector_options": {"class_confidence": {"person": .15}}}))
    (tmp_path/"manifest.json").write_text(json.dumps({"spec_sha256": sha256(spec.path), "reference_sha256": sha256(reference_path), "calibration_sha256": sha256(spec.calibration_path)}))
    output = tmp_path/"observations.jsonl"
    assert infer(spec.path, reference_path, model, metadata, output) == 1
    assert received_options == [{"class_confidence": {"person": .15}}]
    converted = json.loads(output.read_text())
    assert converted["ground_truth"][0]["position"] == [1, 2]
    assert converted["observations"][0]["position"] != [1, 2]
    assert converted["observations"][0]["observation_id"] == "Camera_0000:0:0"
    assert not converted.get("reference_only")


def test_training_split_guard_rejects_original_augmented_overlap(spec, tmp_path):
    from yudun.datasets.training import validate_splits
    other = dict(spec.data)
    other["scene_id"] = "Warehouse_001"
    path = tmp_path/"other.json"
    path.write_text(json.dumps(other))
    manifest = tmp_path/"split.json"
    group = scene_group("MTMC_Tracking_2026", "Warehouse_000")
    manifest.write_text(json.dumps({"splits": {"train": [{"spec": str(spec.path), "scene_group": group}],
        "validation": [{"spec": str(path), "scene_group": group}], "test": []}}))
    with pytest.raises(ValueError, match="leaks across"):
        validate_splits(manifest)
    other["scene_id"] = "Warehouse_002"
    path.write_text(json.dumps(other))
    data = json.loads(manifest.read_text())
    data["splits"]["validation"][0]["scene_group"] = scene_group("MTMC_Tracking_2026", "Warehouse_002")
    manifest.write_text(json.dumps(data))
    assert len(validate_splits(manifest)) == 2
