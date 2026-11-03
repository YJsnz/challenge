"""Industrial adaptation of the original monocular radar Qt panel."""
from datetime import datetime
from pathlib import Path
import math
import sys
import time
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QPointF, QRectF
from PyQt5.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPushButton,
    QSlider, QSplitter, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget)
from .vendor.panel_base import RadarPanelBase, StatusIndicator
from .vendor.zoomable import ZoomableDraggableLabel
from .workers import ReplayWorker, RosWorker
from ..evaluation import save_run

CLASS_NAMES = {"person": "人员", "forklift_or_cart": "叉车 / 小车", "AGV": "AGV", "unknown_dynamic_object": "未知动态目标"}
STATUS_NAMES = {"VISIBLE": "观测", "POSSIBLY_OCCLUDED": "遮挡预测", "LOST": "失去观测"}
SCENE_NAMES = {"occluded": "遮挡交汇", "crossing": "交叉通行", "safe": "错时通行", "stop": "车辆停车",
               "turn": "车辆转弯", "sensor_dropout": "雷达断流", "low_light": "低照度", "jitter": "时间抖动"}
MODE_NAMES = {"full": "融合与遮挡推断", "visual": "仅视觉", "lidar": "仅激光",
              "fixed_fusion": "固定协方差融合", "quality_fusion": "质量自适应融合"}


class WarehouseProjectionWidget(QWidget):
    """Replaces competition field dimensions/colours with metric warehouse state."""
    def __init__(self, config):
        super().__init__()
        self.config, self.frame = config, None
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
                painter.setPen(QPen(QColor("white"), 2))
                painter.setBrush(color)
                painter.drawEllipse(point, 6 if is_person else 8, 6 if is_person else 8)
                painter.setPen(QColor("#17394b"))
                painter.drawText(point+QPointF(11, -6), f"#{track['id']} {STATUS_NAMES.get(track['status'], track['status'])}")
        painter.end()


class RadarStationMainWindow(RadarPanelBase):
    """Uses the original 2:2:1 panel layout without referee/camera singletons."""
    def __init__(self, config, input_path=None, use_ros=False, calibration=None):
        super().__init__()
        self.config, self.input_path, self.calibration = config, input_path, calibration
        self.frames, self.inputs, self.index = [], [], 0
        self.worker = None
        self.live = False
        self.last_receipt = 0.
        self.video = None
        self.setWindowTitle("域盾 · 多模态空间感知与主动安全预警")
        self.resize(1540, 940)
        self.init_ui()  # Original monocular RadarStationMainWindow three-panel assembly.
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.advance)
        self.health_timer = QTimer(self)
        self.health_timer.timeout.connect(self.check_freshness)
        self.health_timer.start(250)
        self.statusBar().showMessage("原生 Qt 面板 · 合成数据仅用于流程验证")
        QTimer.singleShot(0, self.start_live if use_ros else self.reload)

    def create_image_panel(self):
        panel = QGroupBox("图像与点云")
        layout = QVBoxLayout(panel)
        self.camera_label = ZoomableDraggableLabel()
        self.camera_label.setMinimumSize(300, 220)
        self.camera_label.setText("等待相机画面\n合成回放不提供真实视频")
        self.tracker_label = ZoomableDraggableLabel()
        self.tracker_label.setMinimumSize(300, 220)
        self.tracker_label.setText("等待原始点云\n下方为雷达坐标系 XY 投影")
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.camera_label)
        splitter.addWidget(self.tracker_label)
        layout.addWidget(splitter)
        controls = QHBoxLayout()
        open_video = QPushButton("打开参考视频")
        open_video.clicked.connect(self.open_video)
        controls.addWidget(open_video)
        reset = QPushButton("图像适应窗口")
        reset.clicked.connect(lambda: [label.fit_to_window() for label in (self.camera_label, self.tracker_label)])
        controls.addWidget(reset)
        layout.addLayout(controls)
        layout.addWidget(QLabel("滚轮缩放 · 左键拖拽；视频仅为回放参考，不参与定位。"))
        return panel

    def create_field_projection_panel(self):
        panel = QGroupBox("场地实时投影与轨迹")
        layout = QVBoxLayout(panel)
        self.field_view = WarehouseProjectionWidget(self.config)
        layout.addWidget(self.field_view, 3)
        options = QHBoxLayout()
        for label, attr in (("人员", "show_people"), ("车辆", "show_vehicles"), ("预测", "show_predictions"), ("不确定范围", "show_uncertainty")):
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.toggled.connect(lambda value, name=attr: self.set_display_option(name, value))
            options.addWidget(cb)
        layout.addLayout(options)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["ID", "类别", "x / y (m)", "速度 m/s", "状态", "观测龄 s"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
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
        row = QHBoxLayout()
        for widget in (self.play_button, self.seek, self.speed, self.time_label):
            row.addWidget(widget)
        layout.addLayout(row)
        return panel

    def create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        status = QGroupBox("设备状态")
        status_layout = QVBoxLayout(status)
        self.camera_status, self.lidar_status = StatusIndicator("相机"), StatusIndicator("激光雷达")
        self.source_label = QLabel("合成回放 · 流程验证")
        self.source_label.setWordWrap(True)
        self.performance = QLabel("等待数据")
        for widget in (self.camera_status, self.lidar_status, self.source_label, self.performance):
            status_layout.addWidget(widget)
        layout.addWidget(status)
        control = QGroupBox("运行控制")
        controls = QVBoxLayout(control)
        self.scenario, self.mode = QComboBox(), QComboBox()
        for key, name in SCENE_NAMES.items():
            self.scenario.addItem(name, key)
        for key in ("full", "visual", "lidar", "fixed_fusion", "quality_fusion"):
            self.mode.addItem(MODE_NAMES[key], key)
        for widget in (self.scenario, self.mode):
            controls.addWidget(widget)
            widget.currentIndexChanged.connect(self.reload)
        open_file = QPushButton("打开观测 JSONL")
        open_file.clicked.connect(self.open_input)
        self.open_file_button = open_file
        controls.addWidget(open_file)
        synthetic = QPushButton("切回合成场景")
        synthetic.clicked.connect(self.use_synthetic)
        self.synthetic_button = synthetic
        controls.addWidget(synthetic)
        self.calibration_button = QPushButton("选择传感器标定 JSON")
        self.calibration_button.clicked.connect(self.choose_calibration)
        controls.addWidget(self.calibration_button)
        self.live_button = QPushButton("启动 ROS 2 实时接入")
        self.live_button.clicked.connect(self.toggle_live)
        controls.addWidget(self.live_button)
        self.export_button = QPushButton("导出轨迹、事件与评测")
        self.export_button.clicked.connect(self.export_run)
        controls.addWidget(self.export_button)
        layout.addWidget(control)
        risk = QGroupBox("交汇风险与事件")
        risk_layout = QVBoxLayout(risk)
        self.risk_label = QLabel("当前未触发风险事件")
        self.risk_label.setWordWrap(True)
        self.event_log = QTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.document().setMaximumBlockCount(300)
        risk_layout.addWidget(self.risk_label)
        risk_layout.addWidget(self.event_log)
        layout.addWidget(risk, 1)
        note = QLabel("辅助预警研究原型\n预测位置不等于观测位置。\n风险概率尚需独立数据校准。")
        note.setWordWrap(True)
        layout.addWidget(note)
        return panel

    def set_display_option(self, name, value):
        setattr(self.field_view, name, value)
        self.field_view.update()

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
        self.source_label.setText("外部数据回放 · 来源请核验" if self.input_path else "合成回放 · 流程验证")
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
            self.render_video(self.frames[index]["timestamp"]-self.frames[0]["timestamp"])

    def render(self, frame, rebuild_events=False):
        self.field_view.set_frame(frame)
        self.time_label.setText(f"{frame['timestamp']:.1f} s")
        self.camera_status.update_status(frame["health"]["camera"] == "ONLINE")
        self.lidar_status.update_status(frame["health"]["lidar"] == "ONLINE")
        self.performance.setText(f"活动轨迹 {len(frame['tracks'])} · 核心处理 {frame['processing_ms']:.1f} ms")
        self.table.setRowCount(len(frame["tracks"]))
        for row, t in enumerate(frame["tracks"]):
            data = [t["id"], CLASS_NAMES.get(t["category"], t["category"]), f"{t['state'][0]:.2f}, {t['state'][1]:.2f}",
                    f"{math.hypot(*t['state'][2:]):.2f}", STATUS_NAMES.get(t["status"], t["status"]), f"{t['age']:.2f}"]
            for col, value in enumerate(data):
                self.table.setItem(row, col, QTableWidgetItem(str(value)))
        active = [e for e in frame["events"] if e["level"] > 0]
        self.risk_label.setText("\n".join(f"{['','提示','预警','紧急人工关注'][e['level']]}：#{e['pair'][0]} / #{e['pair'][1]}\n"
                                f"风险估计 {e['probability']:.0%}，最近点 {e['tca']:.1f} s，间距 {e['minimum_distance']:.2f} m" for e in active)
                                or ("感知降级，不能据此判断安全" if frame["system_status"] != "NORMAL" else "当前未触发风险事件"))
        self.risk_label.setStyleSheet("color: #a53726; font-weight: bold" if active else "color: #385666")
        if rebuild_events:
            self.event_log.clear()
            history = [e for f in self.frames[:self.index+1] for e in f["events"] if e["transition"] != "UPDATE"][-100:]
        else:
            history = [e for e in frame["events"] if e["transition"] != "UPDATE"]
        for event in history:
            # Plain text avoids interpreting dataset strings as Qt rich text.
            self.event_log.insertPlainText(f"{event['timestamp']:.1f}s · #{event['pair'][0]}/#{event['pair'][1]} · {event['transition']}\n{event['reason']}\n")

    def display_image(self, image):
        image = np.ascontiguousarray(image)
        h, w, _ = image.shape
        qimage = QImage(image.data, w, h, image.strides[0], QImage.Format_BGR888).copy()
        self.camera_label.setPixmap(QPixmap.fromImage(qimage))

    def display_cloud(self, points):
        pixmap = QPixmap(640, 400)
        pixmap.fill(QColor("#152b38"))
        painter = QPainter(pixmap)
        painter.setPen(QColor("#60c7bb"))
        for x, y, z in np.asarray(points).reshape(-1, 3)[:20000]:
            if np.isfinite([x, y, z]).all() and abs(x) < 20 and abs(y) < 12:
                painter.drawPoint(QPointF(320+x*15, 200-y*15))
        painter.setPen(QColor("white"))
        painter.drawText(12, 22, "原始雷达坐标系 XY · 不作为世界定位结果")
        painter.end()
        self.tracker_label.setPixmap(pixmap)

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

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开与回放起点对齐的参考视频", "", "Video (*.mp4 *.avi *.mkv)")
        if not path:
            return
        import cv2
        if self.video:
            self.video.release()
        self.video = cv2.VideoCapture(path)
        if not self.video.isOpened():
            self.show_error("视频无法打开")
        else:
            self.render_video(0)

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
        if not self.stop_worker():
            return
        self.timer.stop()
        self.frames, self.inputs = [], []
        self.event_log.clear()
        self.live, self.last_receipt = True, time.monotonic()
        self.set_live_controls(True)
        self.source_label.setText("ROS 2 实时接入 · 等待数据")
        directory = Path("outputs/live")/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.worker = RosWorker(self.config, self.mode.currentData(), self.calibration, directory)
        self.worker.result.connect(self.accept_live)
        self.worker.image.connect(self.display_image)
        self.worker.cloud.connect(self.display_cloud)
        self.worker.failed.connect(self.live_error)
        self.worker.start()

    def set_live_controls(self, live):
        for widget in (self.mode, self.scenario, self.open_file_button, self.synthetic_button, self.seek, self.play_button, self.calibration_button):
            widget.setEnabled(not live)
        self.live_button.setText("停止实时接入" if live else "启动 ROS 2 实时接入")
        self.export_button.setEnabled(not live)

    def accept_live(self, result):
        self.last_receipt = time.monotonic()
        self.render(result)
        self.source_label.setText("ROS 2 实时接入 · 原始记录已写入 outputs/live")

    def live_error(self, text):
        self.source_label.setText("ROS 2 接入失败；请停止接入后检查配置")
        self.camera_status.update_status(False)
        self.lidar_status.update_status(False)
        self.show_error(text)

    def check_freshness(self):
        if self.live and time.monotonic()-self.last_receipt > 1:
            self.camera_status.update_status(False)
            self.lidar_status.update_status(False)
            self.risk_label.setText("数据已中断，风险结果失效")
            self.source_label.setText("ROS 2 断流或处理停滞 · 请检查节点")

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
        QMessageBox.warning(self, "域盾", text)

    def closeEvent(self, event):
        self.timer.stop()
        if not self.stop_worker():
            event.ignore()
            return
        if self.video:
            self.video.release()
        event.accept()


def launch(config, input_path=None, use_ros=False, calibration=None):
    app = QApplication(sys.argv[:1])
    app.setApplicationName("域盾")
    window = RadarStationMainWindow(config, input_path, use_ros, calibration)
    window.show()
    return app.exec_()
