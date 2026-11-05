"""Native Qt point cloud and track view with perspective projection."""
import numpy as np
from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import QWidget


class MapView(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(360, 340)
        self.xyz = self.rgb = self.frame = self.scene = self.camera_pose = None
        self.show_people = self.show_vehicles = True
        self.show_predictions = self.show_uncertainty = True
        self.anchor = np.zeros(3)
        self.basis = np.eye(3)
        self.radius = 1.
        self.last = None
        self._background = None
        self.reset_view()

    def set_cloud(self, xyz, rgb, scene=None):
        self.xyz, self.rgb, self.scene = xyz, rgb, scene
        self.frame = self.camera_pose = None
        self.basis = scene.map_from_ground[:3, :3] if scene and scene.registered else np.eye(3)
        self.anchor = (xyz.min(axis=0)+xyz.max(axis=0))/2
        self.radius = max(.1, np.linalg.norm(xyz.max(axis=0)-xyz.min(axis=0))/2)
        if rgb is None:
            height = xyz @ self.basis[:, 2]
            shade = (height-height.min()) / max(1e-6, np.ptp(height))
            self.rgb = np.asarray([69, 112, 129]+shade[:, None]*np.array([106, 106, 104]), dtype=np.uint8)
        self.reset_view()

    def set_frame(self, frame):
        mapped = (frame or {}).get("map_3d")
        self.frame = mapped if self.scene and mapped and mapped["scene_sha256"] == self.scene.sha256 else None
        self.update()

    def reset_view(self):
        self.azimuth, self.elevation = -.90, .69
        self.distance = 1.9*self.radius
        self.focus = self.anchor.copy()
        self._background = None
        self.update()

    def _view(self):
        a, e = self.azimuth, self.elevation
        outward = self.basis @ [np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)]
        right = self.basis @ [-np.sin(a), np.cos(a), 0]
        up = np.cross(outward, right)
        return right, up, outward

    def _project(self, xyz):
        right, up, outward = self._view()
        local = np.asarray(xyz) - self.focus
        depth = self.distance-local @ outward
        focal = .97*min(self.width(), self.height())
        factor = focal/np.maximum(depth, 1e-6)
        screen = np.column_stack([self.width()/2+(local @ right)*factor,
                                  self.height()/2-(local @ up)*factor])
        return screen, depth

    def _cloud_image(self):
        width, height = self.width(), self.height()
        image = np.empty((height, width, 3), dtype=np.uint8)
        image[:] = (20, 38, 52)
        screen, depth = self._project(self.xyz)
        valid = ((depth > 1e-6) & (screen[:, 0] >= 0) & (screen[:, 0] < width)
                 & (screen[:, 1] >= 0) & (screen[:, 1] < height))
        x, y = screen[valid].astype(int).T
        # Nearest sample wins when several points project to the same pixel.
        order = np.argsort(depth[valid])
        _, nearest = np.unique((y*width+x)[order], return_index=True)
        chosen = order[nearest]
        image[y[chosen], x[chosen]] = self.rgb[valid][chosen]
        return QImage(image.data, width, height, image.strides[0], QImage.Format_RGB888).copy()

    def _polyline(self, painter, xyz):
        screen, depth = self._project(xyz)
        path = QPainterPath()
        previous_visible = False
        for (x, y), z in zip(screen, depth):
            if z > 1e-6:
                if previous_visible:
                    path.lineTo(float(x), float(y))
                else:
                    path.moveTo(float(x), float(y))
            previous_visible = z > 1e-6
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)

    def _marker(self, painter, xyz, label, colour, filled=True):
        screen, depth = self._project([xyz])
        if depth[0] <= 1e-6:
            return
        p = QPointF(*screen[0])
        painter.setPen(QPen(colour, 2))
        painter.setBrush(colour if filled else QColor("#142634"))
        painter.drawEllipse(p, 5, 5)
        painter.setPen(QColor("#e6f1f6"))
        painter.drawText(p+QPointF(8, -6), label)

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.xyz is None:
            painter.fillRect(self.rect(), QColor("#142634"))
            painter.setPen(QColor("#bacdd6"))
            painter.drawText(self.rect(), Qt.AlignCenter, "打开已配准地图或 PCD / PLY / NPY 点云")
            return
        if self._background is None or self._background.size() != self.size():
            self._background = self._cloud_image()
        painter.drawImage(0, 0, self._background)
        painter.setRenderHint(QPainter.Antialiasing)
        if self.camera_pose is not None:
            origin = self.camera_pose[:3, 3]
            forward = self.camera_pose[:3, 2]
            painter.setPen(QPen(QColor("#f5c979"), 2))
            self._polyline(painter, [origin, origin+forward])
            self._marker(painter, origin, "相机", QColor("#f5c979"))
        if self.frame:
            circle = np.linspace(0, 2*np.pi, 49)
            for track in self.frame["tracks"]:
                person = track["category"] == "person"
                if not (self.show_people if person else self.show_vehicles):
                    continue
                color = QColor("#62e5bc" if person else "#72b9ff")
                guide = QColor(color)
                guide.setAlpha(110)
                xyz = np.asarray(track["position_m"])
                if self.show_uncertainty:
                    values, vectors = np.linalg.eigh(track["covariance_m2"])
                    ellipse = (np.column_stack([np.cos(circle), np.sin(circle)])
                               * (2.448*np.sqrt(np.maximum(values[-2:], 0))))
                    painter.setPen(QPen(guide, 1))
                    self._polyline(painter, xyz+ellipse @ vectors[:, -2:].T)
                if self.show_predictions:
                    painter.setPen(QPen(guide, 1, Qt.DashLine))
                    self._polyline(painter, [xyz]+[p["position_m"] for p in track["predictions"]])
                visible = track["status"] == "VISIBLE"
                name = "人员" if person else "车辆"
                self._marker(painter, xyz, f"#{track['id']} {name}"+("" if visible else " · 预测"), color, visible)
        painter.setPen(QColor("#b5ccd8"))
        painter.drawText(14, self.height()-14, "左键旋转  ·  右键平移  ·  滚轮缩放  ·  双击复位")

    def mousePressEvent(self, event):
        self.last = event.pos()

    def mouseMoveEvent(self, event):
        if self.last is None or self.xyz is None:
            return
        delta = event.pos()-self.last
        self.last = event.pos()
        if event.buttons() & Qt.LeftButton:
            self.azimuth -= .009*delta.x()
            self.elevation = float(np.clip(self.elevation+.009*delta.y(), -1.45, 1.45))
        elif event.buttons() & Qt.RightButton:
            right, up, _ = self._view()
            self.focus += self.distance/max(1, min(self.width(), self.height()))*(-delta.x()*right+delta.y()*up)
        self._background = None
        self.update()

    def mouseReleaseEvent(self, event):
        self.last = None

    def wheelEvent(self, event):
        self.distance = float(np.clip(self.distance*np.exp(-event.angleDelta().y()/960),
                                      .15*self.radius, 12*self.radius))
        self._background = None
        self.update()

    def mouseDoubleClickEvent(self, event):
        self.reset_view()
