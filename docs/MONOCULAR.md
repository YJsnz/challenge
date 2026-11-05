# 照微固定单目相机方案

2026-11-04 按用户要求改为固定单目主线。现已完成首轮 WSL Linux 软件验证：59 项测试、32 组合成回归、Qt 原生窗口及两套 YOLO 的导出/真实视频冒烟检查通过。模型场景效果、实机相机、现场标定与正式实验仍待验证，详见 [实际验证记录](WSL_VALIDATION_2026-11-04.md)。

LiDAR 继续作为预留扩展接口，独立配置且默认关闭。接口契约与后续接入顺序见 [LIDAR_ROADMAP.md](LIDAR_ROADMAP.md)。

## 主线流程

一台固定 RGB 相机 → 工业目标检测 → 底边接地点估计 → 相机内外参与地面交点 → map 平面位置及协方差 → 目标关联与常速跟踪 → 货架可见性与短时遮挡状态 → 未来轨迹分布 → 交汇风险事件 → 原生 Qt 与日志。

单目位置是基于固定视角、已知地面和标定的估计，不是通用单目绝对深度恢复。相机移动、分辨率变化、斜坡、多层地面、车辆底边遮挡会影响结果。完全不可见时仅做有限预测，扩大位置不确定性；超过 visible_timeout/occluded_timeout 或协方差上限后结束轨迹。没有观测不能证明通道安全。

## Linux 软件入口

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[qt,test,data]'
QT_QPA_PLATFORM=offscreen python -m pytest -q
python -m yudun simulate --scenario camera_dropout
python -m yudun run datasets/generated/demo.jsonl --mode monocular --output outputs/mono
python -m yudun gui --input datasets/generated/demo.jsonl
```

这些命令在 Linux 或 WSL Ubuntu 内执行，不使用 Windows 原生 Python。WSL 环境步骤见 [操作说明](WSL.md)。默认场景只配置 camera，未配置传感器不会被伪造在线；单目 Pipeline 会拒绝 LiDAR 观测和双传感器场景配置。

## 相机标定与实际推理

检测器支持在 `camera.detector_options` 配置分类别阈值和可选重叠分块；SmartSpaces 模型 sidecar 使用同名字段。全帧与仓库分块预设、正负例及 CPU 耗时见 [检测改进记录](DETECTION_REVIEW_2026-11-04.md)。分块默认关闭，不能把一个场景的改善直接推广到其他场景。

复制 configs/calibration.example.json 为 configs/camera.local.json，填入实际工业 ONNX 权重、类别顺序、相机内参、畸变、world_from_camera 和测量协方差。image_width/image_height 必须是完成标定的图像尺寸。示例的 example_only=true 会拒绝实机使用，完成实际标定后才能设为 false。

world_from_camera 是从相机光学坐标到 map 的刚性变换，PnP 返回的反方向变换需要取逆。默认地面 z=0。场景 sensor_positions.camera 应与外参平移的 XY 一致，障碍矩形为同一 map 系下的实测货架范围；先验证几个独立地面控制点的误差，再进行人车测试。

```bash
python -m yudun gui --camera-index 0 --calibration configs/camera.local.json
python -m yudun gui --video /path/to/fixed-camera.mp4 --calibration configs/camera.local.json
```

可在 Qt 中选择 USB 相机 0、本地视频推理或 ROS 相机。CLI 可指定其他 USB 索引。视频按采集帧号/FPS计算时间；USB 按实际读帧时刻计算。读取失败停止更新，Qt 在断流后将风险结果标为失效。每次运行的 input.jsonl/results.jsonl 保存至独立 outputs/live 目录，可后续离线评测。USB 当前使用 OpenCV 采集，海康 SDK 曝光/增益控制尚未实现。

## 可选 ROS 相机输入

仅在已有工业相机 ROS 驱动时需要 ROS 2 Humble。构建 ros2/src 中 yudun_interfaces 与 yudun_ros，并为同一 Python 环境安装 yudun。

```bash
source /opt/ros/humble/setup.bash
source ros2/install/setup.bash
python -m yudun gui --ros --calibration configs/camera.local.json
```

主线订阅 /sensors/camera/image，发布 /perception/visual_detections、/tracking/fused_tracks 和 /safety/risk_events；不订阅或启动雷达节点。Qt 内部启动视觉与跟踪，不要再启动同名独立节点。历史 ROS 包/消息名含 fusion 或 FusedTrack，仅为现有接口兼容；lidar_status 在历史消息字段中为 NOT_CONFIGURED。

## 数据与实验

优先使用 SmartSpaces 单相机视频、标定和独立 GT。按 docs/SMARTSPACES_WORKFLOW.md 准备、训练、推理；同一场景的不同相机不同时送入 Pipeline。训练与验证仍按原场景和增强版本分组，不能跨分区泄漏。

当前对比组：M0=monocular_basic（基础跟踪）；M1=monocular（测量质量降权、可见性状态机与有限遮挡预测）。风险引擎对两组保持一致，以比较感知差异对预警的影响。分别报告正常/低照度、无遮挡/短遮挡/长遮挡、启停/转弯和相机中断场景。

定位报告独立参考误差与有效区域；跟踪报告 IDF1、ID Switch 与召回；预测报告 ADE/FDE 和协方差覆盖率；风险报告危险事件召回、误报/小时、提前量及概率可靠性。先验证原型运行，再以数据判断改进是否有效，不将程序模块名称作为算法创新证据。

当前不要求用双传感器 B1–B5 证明单目方案。历史双传感器代码与审计保留，可通过 Git 复查；实验表格、申报书和答辩材料应以此单目范围重新组织。原始 V1.0 DOCX 保留为历史方案，未同步重写。
