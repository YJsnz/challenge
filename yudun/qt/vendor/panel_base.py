# Adapted from HKUST ENTERPRIZE RM2026-Radar. Copyright (c) 2025 HKUST ENTERPRIZE Team. See LICENSE.
from PyQt5.QtWidgets import QWidget, QMainWindow, QHBoxLayout, QLabel

class RadarPanelBase(QMainWindow):
    def init_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QHBoxLayout(central_widget)

        # 左侧区域：相机图像和雷达点云
        left_panel = self.create_image_panel()

        # 中间区域：场地投影和车辆坐标
        center_panel = self.create_field_projection_panel()

        # 右侧区域：控制面板和状态监控
        right_panel = self.create_control_panel()

        # 添加到主布局
        main_layout.addWidget(left_panel, 2)  # 左侧占2/5
        main_layout.addWidget(center_panel, 2)  # 中间占2/5
        main_layout.addWidget(right_panel, 1)  # 右侧占1/5


class StatusIndicator(QWidget):
    """设备状态指示器"""

    def __init__(self, device_name):
        super().__init__()
        self.device_name = device_name
        self.connected = False
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.indicator = QLabel("●")
        self.indicator.setFixedSize(20, 20)
        self.label = QLabel(self.device_name)

        layout.addWidget(self.indicator)
        layout.addWidget(self.label)
        layout.addStretch()

        self.update_status(False)

    def update_status(self, connected):
        """更新连接状态"""
        self.connected = connected
        if connected:
            self.indicator.setStyleSheet("color: green; font-size: 16px;")
            self.label.setText(f"{self.device_name}: 已连接")
        else:
            self.indicator.setStyleSheet("color: red; font-size: 16px;")
            self.label.setText(f"{self.device_name}: 未连接")

