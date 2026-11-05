"""Linux offscreen widget tests; not executed on the Windows editing host."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPixmap, QColor
from yudun.pipeline import load_config, replay
from yudun.simulation import generate
from yudun.qt.vendor.zoomable import ZoomableDraggableLabel
from yudun.qt.main import WarehouseProjectionWidget


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_original_zoom_component_preserves_view_on_new_frame(app):
    label = ZoomableDraggableLabel()
    label.resize(400, 300)
    image = QPixmap(800, 600)
    image.fill(QColor("blue"))
    label.setPixmap(image)
    label.scale_factor = 1.5
    label.setPixmap(image)
    assert label.scale_factor == 1.5
    label.close()


def test_warehouse_projection_renders_finite_tracks(app):
    config = load_config()
    widget = WarehouseProjectionWidget(config)
    widget.resize(640, 480)
    frames = replay(generate(config, "crossing", duration=1), config)
    widget.set_frame(frames[-1])
    assert not widget.grab().isNull()
    widget.close()


def test_camera_only_dataset_status_does_not_require_lidar(app, monkeypatch):
    from yudun.qt.main import RadarStationMainWindow
    config = load_config()
    config["sensor_positions"] = {"camera": [0, -7]}
    monkeypatch.setattr(RadarStationMainWindow, "reload", lambda self: None)
    window = RadarStationMainWindow(config)
    # Constructor schedules a replay on the next event-loop turn; test the direct
    # render path without entering that loop or starting any ROS source.
    frame = replay(generate(config, "crossing", duration=.1), config)[0]
    window.render(frame)
    assert not hasattr(window, "lidar_status")
    assert window.mode.count() == 2
    window.close()


def test_native_3d_map_tabs_and_xyz_render(app, monkeypatch, tmp_path):
    pytest.importorskip("cv2")
    from yudun.map3d.demo import make_demo
    from yudun.pipeline import read_jsonl
    from yudun.qt.main import RadarStationMainWindow
    demo = make_demo(tmp_path/"demo")
    monkeypatch.setattr(RadarStationMainWindow, "reload", lambda self: None)
    window = RadarStationMainWindow(load_config(demo["scene_config"]))
    frame = list(read_jsonl(demo["results"]))[-1]
    window.render(frame)
    assert not window.windowIcon().isNull() and window.logo.hasAlphaChannel()
    assert window.views.currentWidget() is window.map_view
    assert window.views.count() == 3 and "尺寸未实测" in window.map_label.text()
    assert window.table.item(0, 6).text() != "—"
    window.table.selectRow(0)
    selected = frame["tracks"][0]["id"]
    assert window.map_view.selected_track_id == window.field_view.selected_track_id == selected
    window.render({**frame, "tracks": list(reversed(frame["tracks"]))})
    assert window.selected_track_id == selected
    window.render({**frame, "tracks": [t for t in frame["tracks"] if t["id"] != selected]})
    assert window.selected_track_id is None
    assert not window.map_view.grab().isNull()
    window.set_display_option("show_people", False)
    assert not window.map_view.show_people and not window.field_view.show_people
    window.load_map(tmp_path/"demo/cloud.npy")
    assert "未配准" in window.map_label.text()
    assert window.views.count() == 2 and "map_scene" not in window.config
    window.close()
