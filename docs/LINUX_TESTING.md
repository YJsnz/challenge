# Linux 构建与验证计划

本文是后续执行步骤，当前 Qt/ROS 版本尚未执行这些检查。2026-11-03 用户明确要求 Windows 仅修改代码，Linux 测试之后进行。详见 [VALIDATION.md](VALIDATION.md)。

## 1 离线核心与 Qt

目标基线 Ubuntu 22.04、Python 3.10；不要复用 ZIP 中 build/install/log。

```bash
sudo apt install python3-venv python3-pip python3-pyqt5 libgl1 libxcb-xinerama0
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[qt,test]'
QT_QPA_PLATFORM=offscreen python -m pytest -q
python -m yudun simulate --scenario occluded
python -m yudun run datasets/generated/demo.jsonl --output outputs/linux-smoke
python -m yudun gui
```

检查原三栏布局、滚轮缩放/拖拽、刷新保留缩放、场地米制投影、目标筛选、预测虚线与协方差范围、播放/暂停/拖动、模式切换、导出、文件错误提示、线程退出。测试报告必须记录实际 Qt/Python/NumPy/OpenCV 版本。

原面板使用 PyQt5；不要在同一进程混用 PyQt5、PyQt6 或 PySide6。OpenCV 若引入其他 Qt 插件目录，应在干净虚拟环境排查，不照搬上游 LD_PRELOAD/用户绝对路径。

## 2 构建独立 ROS 包

已安装 ROS 2 Humble 后：

```bash
source /opt/ros/humble/setup.bash
sudo apt install python3-colcon-common-extensions ros-humble-cv-bridge ros-humble-sensor-msgs-py
source .venv/bin/activate
cd ros2
colcon build --symlink-install --packages-select yudun_interfaces yudun_ros
source install/setup.bash
cd ..
ros2 interface show yudun_interfaces/msg/DetectionArray
ros2 run yudun_ros fusion
```

核心 Python 包需对运行 ROS 节点的同一解释器可见。若 colcon 选取了系统 Python，应在该环境安装 yudun 或明确使用一致的虚拟环境；避免 NumPy/OpenCV 与 cv_bridge ABI 不一致。

## 3 真值、标定与模型

`configs/calibration.example.json` 只展示字段，`example_only=true` 会被节点拒绝。用真实内外参替换后保存独立文件，确认 `schema_version=1`、`frame_id=map`、`example_only=false`。同时将场景配置中的 `sensor_positions` 更新为真实外参 XY，障碍框更新为实测货架布局。

相机外参是 **world_from_camera**，不是 solvePnP 原始的 camera_from_world；若来自 PnP，应先取逆。像素采用光学系 x 右、y 下、z 前。地面交点模型要求近似平坦地面，斜坡/多层平台需补充模型。2x2 measurement_covariance 是实测位置误差，不是随意填写的精度保证。畸变参数与图像分辨率必须匹配。

LiDAR 输入需与 `frame_id` 一致，外参为 world_from_lidar；静态地图 NPY 为 map 系 Nx3 点，`allow_pickle=False`。尚未提供现场 GICP 自动标定流程；可复用上游独立定位节点，但需单独修复/编译验证和建立仓储地图。固定地面阈值适用于首轮平地实验，不能替代复杂地面分割。

工业 ONNX 模型不包含在仓库中。适配器只接受显式工业类别列表与原始 YOLOv8 detect 输出，不会自动下载权重或将比赛装甲板模型改名为人员/叉车。相对模型/地图路径相对于标定 JSON 所在目录解析。

## 4 Qt 直接接入 ROS

先运行实际相机与雷达驱动，重映射到标定配置中的 raw topic，再执行：

```bash
source /opt/ros/humble/setup.bash
source .venv/bin/activate
source ros2/install/setup.bash
python -m yudun --config /path/to/warehouse.json gui --ros --calibration /path/to/calibration.json
```

Qt 内部通过后台 ROS executor 启动传感器适配与融合，结果用 Qt signal/slot 送到面板，**没有 HTTP 或网页控制链路**。不要再启动同名独立 fusion/sensors 节点，否则会重复发布。若已有外部适配器输出统一 detections，可不传 `--calibration`，只启动 Qt 内部融合。

| 话题 | 消息 | QoS |
|---|---|---|
| `/sensors/camera/image` | sensor_msgs/Image | 传感器 best effort，depth 5 |
| `/sensors/lidar/points` | sensor_msgs/PointCloud2 | 传感器 best effort，depth 5 |
| `/perception/visual_detections` | yudun_interfaces/DetectionArray，sensor=camera | 传感器 best effort |
| `/perception/lidar_detections` | yudun_interfaces/DetectionArray，sensor=lidar | 传感器 best effort |
| `/tracking/fused_tracks` | yudun_interfaces/FusedTrackArray | reliable，depth 10 |
| `/safety/risk_events` | yudun_interfaces/RiskEventArray | reliable，depth 10 |

DetectionArray 与每项 detection 的采集时间戳必须相同；空数组也是有效数据帧。当前融合默认 10Hz，每个 tick 只取每个源最新完整观测，拒绝重复/乱序/未来/过旧数据，输出时间使用 ROS 时钟。坐标转换在传感器适配层完成；不在核心中悄悄变换坐标。静态外参方案不使用动态 TF，移动传感器不属于当前支持范围。

播放 rosbag 时对节点设置 `use_sim_time=true`，暂停时不老化；倒退 seek 必须重启融合节点清空历史。Qt 当前主要支持 JSONL 的任意拖动；ROS bag 倒退不自动重置。Jazzy/Humble 跨发行版通信不作为默认兼容保证，应先验证消息类型与 RMW/QoS。

## 5 验收顺序

1. B1 视觉、B2 点云与真正的上游 B3 在同一录制数据上留存 CSV。
2. 用独立参考定位验证轴向、米/厘米、时钟偏差、误差和协方差覆盖率。
3. 回放遮挡、转弯、启停、弱光、同类多目标与交汇，复查身份切换和失败案例。
4. 断开相机/雷达，确认降级、失效事件、过期轨迹退出及 Qt 不继续显示有效风险结论。
5. 在独立 GT 上计算 IDF1、位置误差与多时域位移误差，另行计算 HOTA/轨迹 ADE 与概率校准；对危险窗口和良性片段统计召回、误报/小时与提前量。
6. 记录端到端延迟（不只核心处理时间）、连续 8 小时稳定性和恢复行为，再对照方案目标。

实时推理与点云处理目前共享单线程 executor，必须在 Linux 实测负载后决定独立进程/线程拆分；Python 点云适配器是功能基线，未承诺高密点云实时吞吐。曝光/增益、海康 SDK 控制、通用三维视图和声光硬件联动不在当前 Qt 接线范围内。
