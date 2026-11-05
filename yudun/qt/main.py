"""Industrial adaptation of the original monocular radar Qt panel."""
from datetime import datetime
from importlib.resources import files
from pathlib import Path
import math
import sys
import time
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QPointF, QRectF, QSignalBlocker
from PyQt5.QtGui import QColor, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog,
    QFrame, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QMessageBox, QPushButton,
    QSizePolicy, QSlider, QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget)
from .vendor.panel_base import RadarPanelBase, StatusIndicator
from .vendor.zoomable import ZoomableDraggableLabel
from .workers import ReplayWorker, RosWorker, CameraWorker
from .map_view import MapView
from .theme import WORKSTATION_STYLE
from .. import APP_NAME
from ..camera import camera_config, load_camera
from ..evaluation import save_run
from ..map3d.cloud import load_cloud
from ..map3d.scene import MapScene

CLASS_NAMES = {"person": "人员", "forklift_or_cart": "叉车 / 小车", "AGV": "AGV", "unknown_dynamic_object": "未知动态目标"}
STATUS_NAMES = {"VISIBLE": "观测", "POSSIBLY_OCCLUDED": "遮挡预测", "LOST": "失去观测"}
EVENT_NAMES = {"START": "风险触发", "ESCALATE": "风险升级", "DEESCALATE": "风险降低",
               "END": "风险解除", "UNAVAILABLE": "评估中断"}
SCENE_NAMES = {"occluded": "遮挡交汇", "crossing": "交叉通行", "safe": "错时通行", "stop": "车辆停车",
               "turn": "车辆转弯", "camera_dropout": "相机断流", "low_light": "低照度", "jitter": "时间抖动"}
MODE_NAMES = {"monocular": "单目质量与遮挡预测", "monocular_basic": "单目基础跟踪"}


def section(title):
    panel = QFrame()
    panel.setObjectName("section")
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(12, 10, 12, 12)
    layout.setSpacing(8)
    heading = QLabel(title)
    heading.setObjectName("sectionHeading")
    header = QHBoxLayout()
    header.addWidget(heading)
    header.addStretch()
    layout.addLayout(header)
    return panel, layout, header


class WarehouseProjectionWidget(QWidget):
    """Replaces competition field dimensions/colours with metric warehouse state."""
    def __init__(self, config):
        super().__init__()
        self.config, self.frame = config, None
        self.selected_track_id = None
        self.show_predictions = self.show_uncertainty = True
        self.show_people = self.show_vehicles = True
        self.setMinimumSize(360, 340)

    def project(self, x, y):
        x0, y0, x1, y1 = self.config["bounds"]
        scale = min((self.width()-40)/(x1-x0), (self.height()-40)/(y1-y0))
        return QPointF(self.width()/2 + (x-(x0+x1)/2)*scale,
                       self.height()/2 - (y-(y0+y1)/2)*scale)

    def set_frame(self, frame):
        self.frame = frame
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#edf2f5"))
        x0, y0, x1, y1 = self.config["bounds"]
        painter.setPen(QPen(QColor("#d4dfe5"), 1))
        for x in range(math.ceil(x0), math.floor(x1)+1):
            painter.drawLine(self.project(x, y0), self.project(x, y1))
        for y in range(math.ceil(y0), math.floor(y1)+1):
            painter.drawLine(self.project(x0, y), self.project(x1, y))
        painter.setPen(QPen(QColor("#90a7b4"), 1))
        for box in self.config["obstacles"]:
            rect = QRectF(self.project(box[0], box[3]), self.project(box[2], box[1]))
            painter.fillRect(rect, QColor("#bdcdd5"))
            painter.drawRect(rect)
            painter.drawText(rect, Qt.AlignCenter, "货架")
        for sensor, position in self.config["sensor_positions"].items():
            p = self.project(*position)
            painter.setBrush(QColor("#627f91"))
            painter.drawRect(QRectF(p.x()-3, p.y()-3, 6, 6))
            painter.drawText(p+QPointF(7, 4), sensor.upper())
        if self.frame:
            for track in self.frame["tracks"]:
                is_person = track["category"] == "person"
                if is_person and not self.show_people or not is_person and not self.show_vehicles:
                    continue
                color = QColor("#168879" if is_person else "#286c9b")
                position = track["state"][:2]
                if self.show_uncertainty:
                    values, vectors = np.linalg.eigh(np.asarray(track["covariance"])[:2, :2])
                    ellipse = QPainterPath()
                    for i, angle in enumerate(np.linspace(0, 2*np.pi, 65)):
                        offset = vectors @ (2.448*np.sqrt(np.maximum(values, 0))*[np.cos(angle), np.sin(angle)])
                        point = self.project(*(np.asarray(position)+offset))
                        ellipse.moveTo(point) if i == 0 else ellipse.lineTo(point)
                    ellipse.closeSubpath()
                    fill = QColor(color)
                    fill.setAlpha(28)
                    painter.fillPath(ellipse, fill)
                    painter.setPen(QPen(color, 1))
                    painter.drawPath(ellipse)
                if self.show_predictions:
                    previous = position
                    painter.setPen(QPen(color, 1, Qt.DashLine))
                    for prediction in track["predictions"]:
                        painter.drawLine(self.project(*previous), self.project(*prediction["mean"]))
                        previous = prediction["mean"]
                point = self.project(*position)
                if track["id"] == self.selected_track_id:
                    painter.setBrush(Qt.NoBrush)
                    painter.setPen(QPen(QColor("#ad6a27"), 2))
                    painter.drawEllipse(point, 12, 12)
                painter.setPen(QPen(QColor("white"), 2))
                painter.setBrush(color)
                painter.drawEllipse(point, 6 if is_person else 8, 6 if is_person else 8)
                painter.setPen(QColor("#17394b"))
                painter.drawText(point+QPointF(11, -6), f"#{track['id']} {STATUS_NAMES.get(track['status'], track['status'])}")
        painter.end()


class RadarStationMainWindow(RadarPanelBase):
    """Uses the original 2:2:1 panel layout without referee/camera singletons."""
    def __init__(self, config, input_path=None, use_ros=False, calibration=None, camera_source=None,
                 allow_demo_calibration=False):
        super().__init__()
        self.config, self.input_path, self.calibration = config, input_path, calibration
        self.camera_source, self.stream_name = camera_source, "ROS 2 相机"
        self.allow_demo_calibration = allow_demo_calibration
        self.use_ros = use_ros
        self.frames, self.inputs, self.index = [], [], 0
        self.current_frame = None
        self.selected_track_id = None
        self.worker = None
        self.live = False
        self.last_receipt = 0.
        self.video = None
        self.setWindowTitle(f"{APP_NAME} · 单目定位与安全预警")
        self.logo = QPixmap()
        self.logo.loadFromData(files("yudun").joinpath("assets/zhaowei-logo.png").read_bytes())
        self.setWindowIcon(QIcon(self.logo))
        self.resize(1540, 940)
        self.init_ui()  # Original monocular RadarStationMainWindow three-panel assembly.
        self.centralWidget().setObjectName("workspace")
        workspace = self.centralWidget().layout()
        workspace.setContentsMargins(16, 10, 16, 12)
        workspace.setSpacing(12)
        for column, stretch in enumerate((10, 12, 5)):
            workspace.setStretch(column, stretch)
        self.setMenuWidget(self.create_masthead())
        self.setStyleSheet(WORKSTATION_STYLE)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.advance)
        self.health_timer = QTimer(self)
        self.health_timer.timeout.connect(self.check_freshness)
        self.health_timer.start(250)
        self.statusBar().showMessage("原生 Qt 面板 · 合成数据仅用于流程验证")
        if self.config.get("map_scene"):
            self.load_map(self.config["map_scene"])
        QTimer.singleShot(0, self.start_live if use_ros or camera_source is not None else self.reload)

    def create_masthead(self):
        header = QWidget()
        header.setObjectName("masthead")
        row = QHBoxLayout(header)
        row.setContentsMargins(18, 9, 18, 10)
        row.setSpacing(12)
        seal = QLabel()
        seal.setAccessibleName(APP_NAME)
        seal.setAlignment(Qt.AlignCenter)
        seal.setFixedSize(50, 50)
        seal.setPixmap(self.logo.scaled(50, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        name = QLabel(APP_NAME)
        name.setObjectName("wordmark")
        title = QLabel("仓储人车预警台")
        title.setObjectName("mastheadTitle")
        row.addWidget(seal)
        row.addWidget(name)
        row.addSpacing(12)
        row.addWidget(title)
        row.addStretch(1)
        caption = QLabel("当前输入")
        caption.setObjectName("inputCaption")
        self.source_label = QLabel("等待数据")
        self.source_label.setObjectName("inputSource")
        source = QVBoxLayout()
        source.setSpacing(1)
        source.addWidget(caption)
        source.addWidget(self.source_label)
        row.addLayout(source)
        return header

    def create_image_panel(self):
        panel, layout, header = section("相机画面")
        self.resolution = QLabel("等待画面")
        self.resolution.setObjectName("viewBadge")
        header.addWidget(self.resolution)
        self.camera_label = ZoomableDraggableLabel()
        self.camera_label.setObjectName("cameraCanvas")
        self.camera_label.setMinimumSize(300, 220)
        self.camera_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.camera_label.setText("暂无相机画面\n可打开参考视频或启动相机接入")
        layout.addWidget(self.camera_label, 3)
        metrics = QHBoxLayout()
        self.metrics = []
        for label, value in (("观测轨迹", "—"), ("预测轨迹", "—"), ("跟踪与预警", "— ms")):
            tile = QFrame()
            tile.setObjectName("metricTile")
            column = QVBoxLayout(tile)
            column.setContentsMargins(10, 7, 10, 7)
            column.setSpacing(1)
            caption = QLabel(label)
            caption.setObjectName("caption")
            number = QLabel(value)
            number.setObjectName("metricValue")
            column.addWidget(caption)
            column.addWidget(number)
            self.metrics.append(number)
            metrics.addWidget(tile, 1)
        self.metrics[-1].setToolTip("核心处理耗时，范围为跟踪、预警和地图坐标转换")
        layout.addLayout(metrics)
        controls = QHBoxLayout()
        open_video = QPushButton("打开参考视频")
        open_video.clicked.connect(self.open_video)
        controls.addWidget(open_video)
        reset = QPushButton("图像适应窗口")
        reset.clicked.connect(self.camera_label.fit_to_window)
        controls.addWidget(reset)
        layout.addLayout(controls)
        self.video_hint = QLabel("滚轮缩放 · 左键拖拽 · 参考视频随回放进度显示")
        self.video_hint.setObjectName("caption")
        self.video_hint.setWordWrap(True)
        layout.addWidget(self.video_hint)
        return panel

    def create_field_projection_panel(self):
        panel, layout, _ = section("场地态势")
        self.field_view = WarehouseProjectionWidget(self.config)
        self.map_view = MapView()
        self.views = QTabWidget()
        self.views.addTab(self.field_view, "地面 XY")
        self.views.addTab(self.map_view, "三维地图")
        self.plan_view = ZoomableDraggableLabel()
        self.plan_view.setObjectName("planCanvas")
        self.plan_view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.map_label = QLabel("未加载三维地图")
        self.map_label.setObjectName("caption")
        self.map_label.setTextFormat(Qt.PlainText)
        self.map_label.setWordWrap(True)
        layout.addWidget(self.views, 3)
        layout.addWidget(self.map_label)
        options = QHBoxLayout()
        for label, attr in (("人员", "show_people"), ("车辆", "show_vehicles"), ("预测", "show_predictions"), ("不确定范围", "show_uncertainty")):
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.toggled.connect(lambda value, name=attr: self.set_display_option(name, value))
            options.addWidget(cb)
        layout.addLayout(options)
        heading = QLabel("目标轨迹")
        heading.setObjectName("trackHeading")
        layout.addWidget(heading)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["ID", "类别", "地面 XY (m)", "速度 m/s", "状态", "观测龄 s", "地图 XYZ (m)"])
        self.table.horizontalHeaderItem(6).setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(26)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.itemSelectionChanged.connect(self.select_track)
        self.table.setShowGrid(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setMaximumHeight(220)
        layout.addWidget(self.table, 1)
        self.play_button = QPushButton("播放")
        self.play_button.clicked.connect(self.toggle_play)
        self.seek = QSlider(Qt.Horizontal)
        self.seek.valueChanged.connect(self.seek_to)
        self.speed = QComboBox()
        for text, value in (("0.5×", .5), ("1×", 1.), ("2×", 2.)):
            self.speed.addItem(text, value)
        self.speed.setCurrentIndex(1)
        self.time_label = QLabel("0.0 s")
        self.time_label.setObjectName("timeReadout")
        row = QHBoxLayout()
        for widget in (self.play_button, self.seek, self.speed, self.time_label):
            row.addWidget(widget)
        layout.addLayout(row)
        return panel

    def create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        control, controls, header = section("运行控制")
        self.camera_status = StatusIndicator("相机")
        header.addWidget(self.camera_status)
        self.scenario, self.mode = QComboBox(), QComboBox()
        for key, name in SCENE_NAMES.items():
            self.scenario.addItem(name, key)
        for key in MODE_NAMES:
            self.mode.addItem(MODE_NAMES[key], key)
        for widget in (self.scenario, self.mode):
            controls.addWidget(widget)
            widget.currentIndexChanged.connect(self.reload)
        open_file = QPushButton("观测文件")
        open_file.setToolTip("打开观测 JSONL")
        open_file.clicked.connect(self.open_input)
        self.open_file_button = open_file
        synthetic = QPushButton("合成回放")
        synthetic.clicked.connect(self.use_synthetic)
        self.synthetic_button = synthetic
        self.calibration_button = QPushButton("相机标定")
        self.calibration_button.setToolTip("选择相机标定 JSON")
        self.calibration_button.clicked.connect(self.choose_calibration)
        self.map_button = QPushButton("三维地图")
        self.map_button.setToolTip("打开三维地图或点云")
        self.map_button.clicked.connect(self.choose_map)
        for buttons in ((open_file, synthetic), (self.calibration_button, self.map_button)):
            actions = QHBoxLayout()
            for button in buttons:
                actions.addWidget(button, 1)
            controls.addLayout(actions)
        self.source_kind = QComboBox()
        self.source_kind.addItem("USB 相机 0", "usb")
        self.source_kind.addItem("本地视频推理", "video")
        self.source_kind.addItem("ROS 2 相机", "ros")
        self.source_kind.setCurrentIndex(2 if self.use_ros else 1 if isinstance(self.camera_source, str) else 0)
        controls.addWidget(self.source_kind)
        self.live_button = QPushButton("启动相机接入")
        self.live_button.setObjectName("primary")
        self.live_button.clicked.connect(self.toggle_live)
        controls.addWidget(self.live_button)
        self.export_button = QPushButton("导出轨迹、事件与评测")
        self.export_button.clicked.connect(self.export_run)
        controls.addWidget(self.export_button)
        layout.addWidget(control)
        risk, risk_layout, _ = section("风险记录")
        self.risk_label = QLabel()
        self.risk_label.setWordWrap(True)
        self.set_risk_status("等待轨迹数据", "pending")
        self.event_log = QTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setPlaceholderText("风险变化将在这里按时间记录")
        self.event_log.document().setMaximumBlockCount(300)
        risk_layout.addWidget(self.risk_label)
        risk_layout.addWidget(self.event_log)
        layout.addWidget(risk, 1)
        note = QLabel("辅助预警原型 · 预测位置存在不确定性\n风险概率仍需独立数据校准")
        note.setObjectName("caption")
        note.setWordWrap(True)
        layout.addWidget(note)
        return panel

    def set_display_option(self, name, value):
        for view in (self.field_view, self.map_view):
            setattr(view, name, value)
            view.update()

    def select_track(self):
        rows = self.table.selectionModel().selectedRows()
        item = self.table.item(rows[0].row(), 0) if rows else None
        self.selected_track_id = item.data(Qt.UserRole) if item else None
        for view in (self.field_view, self.map_view):
            view.selected_track_id = self.selected_track_id
            view.update()

    def set_risk_status(self, text, state):
        colours = {"pending": ("#f0eee6", "#6e5529"),
                   "normal": ("#e9f2ef", "#315c55"),
                   "alert": ("#faeae5", "#932f25")}
        background, foreground = colours[state]
        if self.risk_label.property("state") != state:
            self.risk_label.setProperty("state", state)
            self.risk_label.setStyleSheet(
                f"background: {background}; color: {foreground}; padding: 9px; border-radius: 4px;"
            )
        self.risk_label.setText(text)

    def stop_worker(self):
        if self.worker is not None:
            self.worker.requestInterruption()
            if not self.worker.wait(3000):
                self.statusBar().showMessage("工作线程仍在退出，请稍后重试")
                return False
            # Disconnect queued old results before starting a new session.
            try:
                self.worker.disconnect()
            except TypeError:
                pass
            self.worker.deleteLater()
            self.worker = None
        return True

    def reload(self):
        if self.live or not self.stop_worker():
            return
        self.timer.stop()
        self.table.clearSelection()
        self.play_button.setText("播放")
        self.play_button.setEnabled(False)
        self.statusBar().showMessage("正在计算回放…")
        self.worker = ReplayWorker(self.config, self.mode.currentData(), self.scenario.currentData(), self.input_path)
        self.worker.ready.connect(self.replay_ready)
        self.worker.failed.connect(self.show_error)
        self.worker.start()

    def replay_ready(self, inputs, frames):
        self.inputs, self.frames, self.index = inputs, frames, 0
        self.seek.setMaximum(max(0, len(frames)-1))
        self.seek.setValue(0)
        self.play_button.setEnabled(bool(frames))
        self.event_log.clear()
        synthetic = frames and all("synthetic" in f.get("provenance", "") for f in frames)
        self.source_label.setText("合成数据回放 · 流程验证" if synthetic or not self.input_path else "外部数据回放 · 来源请核验")
        self.statusBar().showMessage(f"回放已加载：{len(frames)} 帧")
        if frames:
            self.render(frames[0], rebuild_events=True)

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("播放")
        elif self.frames and not self.live:
            if self.index >= len(self.frames)-1:
                self.seek.setValue(0)
            self.play_button.setText("暂停")
            self.schedule_next()

    def schedule_next(self):
        if self.index >= len(self.frames)-1:
            self.play_button.setText("播放")
            return
        dt = self.frames[self.index+1]["timestamp"]-self.frames[self.index]["timestamp"]
        self.timer.start(max(1, round(1000*dt/self.speed.currentData())))

    def advance(self):
        if self.index < len(self.frames)-1:
            self.seek.setValue(self.index+1)
            self.schedule_next()

    def seek_to(self, index):
        if not self.live and self.frames:
            self.index = index
            self.render(self.frames[index], rebuild_events=True)
            self.render_video(self.video_time())

    def video_time(self):
        if not self.frames:
            return 0.
        # Converted dataset frames keep their original source-video timestamps.
        if self.inputs and "source_frame" in self.inputs[self.index]:
            return self.frames[self.index]["timestamp"]
        return self.frames[self.index]["timestamp"]-self.frames[0]["timestamp"]

    def render(self, frame, rebuild_events=False):
        self.current_frame = frame
        self.field_view.set_frame(frame)
        self.map_view.set_frame(frame)
        self.time_label.setText(f"{frame['timestamp']:.1f} s")
        for name, title, indicator in (("camera", "相机", self.camera_status),):
            indicator.update_status(frame["health"].get(name) == "ONLINE")
            if name not in frame["health"]:
                indicator.label.setText(f"{title}：本序列未配置")
                indicator.indicator.setStyleSheet("color: gray; font-size: 16px;")
            elif not self.live:
                state = "回放有数据" if indicator.connected else "回放无数据"
                indicator.label.setText(f"{title}：{state}")
        predicted = sum(t["status"] != "VISIBLE" for t in frame["tracks"])
        for label, value in zip(self.metrics, (len(frame["tracks"])-predicted, predicted, f"{frame['processing_ms']:.1f} ms")):
            label.setText(str(value))
        blocker = QSignalBlocker(self.table)
        self.table.setRowCount(len(frame["tracks"]))
        mapped = self.map_view.frame
        positions = {t["id"]: t["position_m"] for t in mapped["tracks"]} if mapped else {}
        if self.map_view.scene and self.map_view.scene.registered:
            self.map_label.setText(self.map_description +
                                   (" · 当前轨迹仅作示例" if mapped and mapped["example_only"] else ""))
        selected_row = None
        for row, t in enumerate(frame["tracks"]):
            if t["id"] == self.selected_track_id:
                selected_row = row
            data = [t["id"], CLASS_NAMES.get(t["category"], t["category"]), f"{t['state'][0]:.2f}, {t['state'][1]:.2f}",
                    f"{math.hypot(*t['state'][2:]):.2f}", STATUS_NAMES.get(t["status"], t["status"]), f"{t['age']:.2f}",
                    ", ".join(f"{v:.2f}" for v in positions[t["id"]]) if t["id"] in positions else "—"]
            for col, value in enumerate(data):
                item = QTableWidgetItem(str(value))
                if col == 0:
                    item.setData(Qt.UserRole, t["id"])
                if col in (0, 2, 3, 5):
                    item.setTextAlignment(Qt.AlignCenter)
                if col == 4:
                    item.setForeground(QColor("#216d62" if t["status"] == "VISIBLE" else "#85501c"))
                self.table.setItem(row, col, item)
        self.table.clearSelection()
        if selected_row is not None:
            self.table.selectRow(selected_row)
        blocker.unblock()
        self.select_track()
        active = [e for e in frame["events"] if e["level"] > 0]
        degraded = frame["system_status"] != "NORMAL"
        if active:
            summary = "\n".join(
                f"{['','提示','预警','紧急人工关注'][e['level']]} · #{e['pair'][0]} / #{e['pair'][1]}\n"
                f"风险估计 {e['probability']:.0%} · 最近点 {e['tca']:.1f} s · 间距 {e['minimum_distance']:.2f} m"
                for e in active
            )
            self.set_risk_status(f"感知降级，风险评估受限\n{summary}" if degraded else summary, "alert")
        elif degraded:
            self.set_risk_status("感知降级，风险评估受限", "pending")
        else:
            self.set_risk_status("当前未触发风险事件", "normal")
        if rebuild_events:
            self.event_log.clear()
            history = [e for f in self.frames[:self.index+1] for e in f["events"] if e["transition"] != "UPDATE"][-100:]
        else:
            history = [e for e in frame["events"] if e["transition"] != "UPDATE"]
        for event in history:
            # Plain text avoids interpreting dataset strings as Qt rich text.
            transition = EVENT_NAMES.get(event["transition"], event["transition"])
            self.event_log.insertPlainText(f"{event['timestamp']:.1f}s · #{event['pair'][0]}/#{event['pair'][1]} · {transition}\n{event['reason']}\n")

    def display_image(self, image):
        image = np.ascontiguousarray(image).copy()
        h, w, _ = image.shape
        self.resolution.setText(f"{w} × {h}")
        # Overlay model outputs only; independent GT is never drawn as detections.
        if self.current_frame:
            import cv2
            for detection in self.current_frame.get("visual_detections", []):
                box = detection.get("bbox", [])
                if len(box) != 4 or not np.isfinite(box).all():
                    continue
                x1, y1, x2, y2 = [int(v) for v in box]
                category = detection.get("category")
                colour = (182, 228, 109) if category == "person" else (252, 188, 119)
                caption = {"person": "PERSON", "forklift_or_cart": "FORKLIFT / CART", "AGV": "AGV"}.get(category, "OBJECT")
                text = f"{caption} {detection.get('confidence', 0):.2f}"
                scale = max(.45, w / 1800)
                thickness = max(1, round(scale * 2))
                (label_width, label_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
                left = max(0, min(x1, w-label_width-14))
                top = max(0, y1-label_height-baseline-10)
                cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness+1)
                cv2.rectangle(image, (left, top), (left+label_width+14, top+label_height+baseline+10), (52, 38, 20), -1)
                cv2.putText(image, text, (left+7, top+label_height+4), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, colour, thickness, cv2.LINE_AA)
        qimage = QImage(image.data, w, h, image.strides[0], QImage.Format_BGR888).copy()
        self.camera_label.setPixmap(QPixmap.fromImage(qimage))

    def open_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开统一观测数据", "", "JSONL (*.jsonl)")
        if path:
            self.input_path = path
            self.reload()

    def use_synthetic(self):
        self.input_path = None
        self.reload()

    def choose_calibration(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择已验证的实机标定", "", "JSON (*.json)")
        if path:
            self.calibration = path
            self.statusBar().showMessage(f"标定文件：{path}；将在启动实时接入时加载")

    def choose_map(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开三维地图或点云", "",
                                             "三维地图 (*.json *.pcd *.ply *.npy)")
        if not path:
            return
        scale = 1.
        if Path(path).suffix.lower() != ".json":
            units, ok = QInputDialog.getItem(self, "点云单位", "源坐标单位",
                                            ["米 (m)", "厘米 (cm)", "毫米 (mm)"], editable=False)
            if not ok:
                return
            scale = {"米 (m)": 1., "厘米 (cm)": .01, "毫米 (mm)": .001}[units]
        try:
            self.load_map(path, scale)
        except (ValueError, KeyError, OSError, TypeError) as exc:
            self.show_error(str(exc))
        else:
            self.reload()

    def load_map(self, path, scale=1.):
        scene = MapScene(path) if Path(path).suffix.lower() == ".json" else None
        if scene:
            xyz, rgb, total = scene.cloud, scene.colours, scene.original_points
        else:
            xyz, rgb, total = load_cloud(path)
            xyz *= scale
        self.map_view.set_cloud(xyz, rgb, scene)
        self.config = {**self.config}
        if scene and scene.registered:
            self.config["map_scene"] = str(scene.path)
            self.map_description = (f"{scene.frame_id} · 米 · 已配准 · " +
                                    ("示意地图 / 尺寸未实测" if scene.example_only else "目标落地点") +
                                    f" · {len(xyz):,}/{total:,} 点")
        else:
            self.config.pop("map_scene", None)
            self.map_description = f"点云预览 · 米 · 未配准 · {len(xyz):,}/{total:,} 点"
        self.map_label.setText(self.map_description)
        old_tab = self.views.indexOf(self.plan_view)
        if old_tab >= 0:
            self.views.removeTab(old_tab)
        if scene and scene.data.get("reference_image"):
            pixmap = QPixmap(str(scene.path.parent/scene.data["reference_image"]))
            if not pixmap.isNull():
                self.plan_view.setPixmap(pixmap)
                self.views.addTab(self.plan_view, "参考俯视图")
        self.views.setCurrentWidget(self.map_view)

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开对应原始视频（数据集回放保持原帧号）", "", "Video (*.mp4 *.avi *.mkv)")
        if not path:
            return
        import cv2
        if self.video:
            self.video.release()
        self.video = cv2.VideoCapture(path)
        if not self.video.isOpened():
            self.show_error("视频无法打开")
        else:
            self.render_video(self.video_time())

    def render_video(self, elapsed):
        if self.video and self.video.isOpened() and not self.live:
            import cv2
            self.video.set(cv2.CAP_PROP_POS_MSEC, max(0, elapsed)*1000)
            ok, image = self.video.read()
            if ok:
                self.display_image(image)

    def toggle_live(self):
        if self.live:
            if not self.stop_worker():
                return
            self.live = False
            self.set_live_controls(False)
            self.reload()
        else:
            self.start_live()

    def start_live(self):
        kind = self.source_kind.currentData()
        if kind != "ros" and not self.calibration:
            self.show_error("请先选择已验证的相机标定与模型 JSON")
            return
        if kind == "video" and not isinstance(self.camera_source, str):
            path, _ = QFileDialog.getOpenFileName(self, "选择固定相机视频进行推理", "", "Video (*.mp4 *.avi *.mkv)")
            if not path:
                return
            self.camera_source = path
        elif kind == "usb":
            self.camera_source = self.camera_source if isinstance(self.camera_source, int) else 0
        elif kind == "ros":
            self.camera_source = None
        if kind == "ros" and self.config.get("map_scene"):
            self.show_error("三维地图接入请选择已绑定标定的本地视频或 USB 相机")
            return
        if kind != "ros":
            try:
                camera = load_camera(self.calibration, allow_example=self.allow_demo_calibration and kind == "video")
                config = camera_config(self.config, camera)
                path = config.get("map_scene")
                if path and (not self.map_view.scene or self.map_view.scene.path != Path(path).resolve()):
                    self.load_map(path)
                if path and self.map_view.scene:
                    self.map_view.camera_pose = self.map_view.scene.map_from_ground @ np.asarray(camera["world_from_camera"])
                self.config = self.field_view.config = config
            except (ValueError, KeyError, OSError, TypeError) as exc:
                self.show_error(str(exc))
                return
        if not self.stop_worker():
            return
        self.timer.stop()
        self.table.clearSelection()
        self.frames, self.inputs = [], []
        self.event_log.clear()
        self.live, self.last_receipt = True, time.monotonic()
        self.set_live_controls(True)
        self.statusBar().showMessage("相机接入中 · 等待模型输出")
        self.stream_name = self.source_kind.currentText()
        self.source_label.setText(f"{self.stream_name} · 等待数据")
        directory = Path("outputs/live")/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.worker = (CameraWorker(self.config, self.mode.currentData(), self.calibration, self.camera_source, directory,
                                    self.allow_demo_calibration)
                       if self.camera_source is not None else RosWorker(self.config, self.mode.currentData(), self.calibration, directory))
        self.worker.result.connect(self.accept_live)
        self.worker.image.connect(self.display_image)
        self.worker.failed.connect(self.live_error)
        self.worker.start()

    def set_live_controls(self, live):
        for widget in (self.mode, self.scenario, self.open_file_button, self.synthetic_button, self.seek, self.play_button, self.calibration_button, self.map_button, self.source_kind):
            widget.setEnabled(not live)
        self.live_button.setText("停止相机接入" if live else "启动相机接入")
        self.export_button.setEnabled(not live)
        self.video_hint.setText("滚轮缩放 · 左键拖拽 · 画面来自当前推理输入" if live else
                                "滚轮缩放 · 左键拖拽 · 参考视频随回放进度显示")

    def accept_live(self, result):
        self.last_receipt = time.monotonic()
        self.render(result)
        source = "录像推理 + 示例标定 · 非实测定位" if result.get("example_calibration") else self.stream_name
        self.source_label.setText(source)
        self.statusBar().showMessage(f"{source} · 日志已写入 outputs/live")

    def live_error(self, text):
        self.source_label.setText("相机接入失败；请停止接入后检查配置")
        self.camera_status.update_status(False)
        self.set_risk_status("输入不可用，风险评估已暂停", "pending")
        self.show_error(text)

    def check_freshness(self):
        if self.live and time.monotonic()-self.last_receipt > 1:
            self.camera_status.update_status(False)
            self.set_risk_status("数据已中断，风险结果失效", "pending")
            self.source_label.setText("相机断流、视频结束或处理停滞 · 请检查输入")

    def export_run(self):
        if not self.frames:
            return
        directory = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if directory:
            try:
                save_run(directory, self.inputs, self.frames, self.config, 17)
                self.statusBar().showMessage(f"导出完成：{directory}")
            except (ValueError, OSError) as exc:
                self.show_error(str(exc))

    def show_error(self, text):
        self.statusBar().showMessage(text)
        QMessageBox.warning(self, APP_NAME, text)

    def closeEvent(self, event):
        self.timer.stop()
        if not self.stop_worker():
            event.ignore()
            return
        if self.video:
            self.video.release()
        self.health_timer.stop()
        event.accept()


def launch(config, input_path=None, use_ros=False, calibration=None, camera_source=None, allow_demo_calibration=False):
    app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    window = RadarStationMainWindow(config, input_path, use_ros, calibration, camera_source, allow_demo_calibration)
    window.show()
    return app.exec_()
