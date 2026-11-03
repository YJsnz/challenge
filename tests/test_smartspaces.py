import json
import numpy as np
import pytest
from yudun.datasets.smartspaces import Camera, SceneSpec, fields, clipped_bbox, reference_frame, scene_group, iter_gt
from yudun.pipeline import Pipeline


def calibration():
    return {"sensors": [{"type": "camera", "id": "Camera_0000", "coordinates": {"x": 0, "y": 0},
                         "attributes": [{"name": "fps", "value": "30"}, {"name": "frameWidth", "value": "640"}, {"name": "frameHeight", "value": "480"}],
                         "homography": [[100, 0, 320], [0, 100, 240], [0, 0, 1]]}]}


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
