# 域盾

现成的人员/叉车权重与真实仓储短片见 [模型和视频筛选记录](docs/READY_MODELS_AND_VIDEOS.md)，本地路径、完整下载清单与核对命令见 [资源下载说明](docs/RESOURCE_DOWNLOADS.md)。

当前采用固定单目主线，同时[保留 LiDAR 接口与后续升级规划](docs/LIDAR_ROADMAP.md)。雷达扩展独立配置、默认关闭。

面向仓储人车交汇通道的固定单目相机定位、遮挡跟踪和辅助风险预警研究原型。主线仅使用一台 RGB 相机，依靠已知地面与标定估计目标平面位置。

主界面复用并改造原单目工程的 **PyQt5 面板**：左侧相机与定位说明，中间场地投影与轨迹，右侧相机状态、运行控制与风险事件。支持 USB 相机、本地视频推理、JSONL 回放和可选 ROS 相机输入。原作者许可见 [NOTICE.md](NOTICE.md) 和 [Qt 组件许可](yudun/qt/vendor/LICENSE)。

当前是**已完成首轮 WSL Linux 软件验证的开发版本**：当前 69 项测试通过，已有 32 组合成回归记录，原生 Qt 窗口、安装包构建和两套 YOLO 的真实视频推理已跑通。[首轮检测改进](docs/DETECTION_REVIEW_2026-11-04.md)加入分类别阈值和可选分块，仓库样例有所改善，跨场景仍有误报。ROS、现场标定、工业相机和验收实验尚未验证。环境搭建见 [WSL 操作说明](docs/WSL.md)，原始结果见 [验证记录](docs/WSL_VALIDATION_2026-11-04.md)。

## 在 Linux 开始

本轮已验证 Ubuntu 24.04、Python 3.12（WSL 2）；原计划 Ubuntu 22.04、Python 3.10 仍需另行验证。USB/视频/JSONL 主线无需 ROS，工业相机的可选 ROS 驱动需使用其支持的发行版。

```bash
git clone https://github.com/YJsnz/challenge.git
cd challenge
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[qt,test]'
python -m yudun gui
```

Qt 初始显示合成场景与明确的数据来源标记，不提供虚构摄像画面。可切换场景与感知模式、播放/拖动进度、导出轨迹和事件，或打开独立观测 JSONL。参考视频从回放起点对齐，仅显示，不参与定位推理。

```bash
python -m yudun simulate --scenario occluded --output datasets/generated/demo.jsonl
python -m yudun run datasets/generated/demo.jsonl --mode monocular --output outputs/run
python -m yudun gui --input datasets/generated/demo.jsonl
```

单目方案的标定、实机命令、Linux 验证与实验分组见 [单目主线说明](docs/MONOCULAR.md)。Windows 宿主机负责编辑与 Git，运行验证在 WSL Ubuntu 内执行。旧双传感器设计保存在历史文档和 Git 中，不作为当前默认部署步骤。

```bash
python -m yudun gui --camera-index 0 --calibration configs/camera.local.json
python -m yudun gui --video /path/to/fixed-camera.mp4 --calibration configs/camera.local.json
```

公开仓储数据的准备、YOLO 训练集导出、工业模型训练/导出、模型观测与 GT 隔离、Qt 视频对齐及基线评测，见 [SmartSpaces 操作流程](docs/SMARTSPACES_WORKFLOW.md)。工具入口为 `python -m yudun.data`，数据处理需安装 `.[data]`，训练工具为可选的 `.[train]`。

## 代码组成

| 路径 | 内容 |
|---|---|
| `docs/audit/` | 来源、依赖、权重、13 项静态发现、调用图、完整哈希清单 |
| `yudun/schema.py` | 带版本、时戳、坐标系、单位与协方差的观测接口 |
| `yudun/adapters.py` | 工业 YOLOv8 ONNX 与像素地面投影；旧点云函数仅供历史复现 |
| `yudun/camera.py` | 不依赖 ROS 的固定相机/视频采集、推理与定位 |
| `yudun/tracking.py` | 时戳驱动 CV 滤波、质量降权、马氏距离与一对一关联、可见性生命周期 |
| `yudun/risk.py` | 轨迹采样、最近点时间与间距、风险概率、分级滞回和冷却 |
| `yudun/qt/` | 改造的原 Qt 面板、后台回放与直接 ROS 信号接入 |
| `ros2/src/` | 可选相机消息、视觉适配与跟踪节点，主线不订阅点云 |
| `yudun/datasets/` | SmartSpaces 流式标注、显式单位转换、分组 YOLO 导出与模型推理 |
| `scripts/` | 上游审计、按场景划分数据集、独立危险事件及风险概率评测 |
| `tests/` | 核心、评测工具和 Linux Qt 离屏回归测试 |

## 数据和实验

输入采用 JSONL，所有位置为米、速度为米/秒、时间为秒，统一 `map` 坐标。空观测列表仍需按时送入，表示当前帧无目标；传感器无数据时不发送该源的 heartbeat。详情与独立 GT/危险事件标注模板见 [datasets/README.md](datasets/README.md)。

输出包括 `input.jsonl`、`results.jsonl`、`tracks.csv`、`events.jsonl`、`latency.csv`、配置和 `metrics.json`，记录随机种子、配置与输入哈希。核心处理时间不包括相机采集、模型推理、传输或 GUI 延迟。

公开模式：`monocular_basic` 为基础 CV 跟踪，`monocular` 为质量加权与有限遮挡预测。两者均只接受 camera 观测。实验用相同图像/检测结果比较定位、跟踪、预测与预警。旧融合模式仅作为内部历史兼容代码，不在 CLI 或 Qt 提供选择；原 B3 CSV 导入器仍可读取真实历史记录，但不再是单目主线必做实验。

```bash
python -m pytest -q
python -m yudun benchmark --repeats 10 --output outputs/benchmark
python scripts/evaluate_events.py outputs/run/input.jsonl outputs/run/results.jsonl datasets/events.json
```

合成回归不能证明工业泛化、实机定位精度或预警效果。独立 GT 不进入估计器。已有 ADE/FDE、协方差覆盖率及概率可靠性诊断代码；HOTA、校准器拟合、长时稳定性、跨场景统计与实际预警提前量仍需正式实验。

## 上游保存与提交

原压缩包、解压目录、模型、实采数据、输出日志和个人开发方案不入 Git；审计清单保留其来源与哈希。后续在 Linux 复查原工程时，将两份原始 ZIP 放回项目根目录后解压到 `upstream/`，再运行 `python scripts/audit.py`；不要直接运行未知启动脚本。Qt 所需许可组件已随代码保存，不依赖上游目录存在。

每个功能步骤单独 commit。当前仓库不预设新增代码的开源许可；`qt/vendor` 及上游的原有许可独立保留。
