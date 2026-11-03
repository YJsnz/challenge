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
