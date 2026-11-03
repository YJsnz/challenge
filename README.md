# 域盾

面向仓储人车交汇通道的相机与三维激光雷达感知、遮挡跟踪和辅助风险预警研究原型。

主界面复用并改造单目雷达工程的原生 **PyQt5 面板**：左侧相机/点云，中间场地投影与轨迹，右侧设备状态、运行控制与风险事件。无需网页或 HTTP 服务。原作者许可见 [NOTICE.md](NOTICE.md) 和 [Qt 组件许可](yudun/qt/vendor/LICENSE)。

当前状态是**待 Linux 验证的开发版本**。源码审计与离线软件模块已落地；Qt/ROS 集成、真实传感器、工业检测模型和验收实验未完成 Linux/实机验证，不能视为已经达到开发方案中的实测目标。

## 在 Linux 开始

建议主线 Ubuntu 22.04、ROS 2 Humble、系统 Python 3.10。无 ROS 时可先运行合成/JSONL 回放。

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
python -m yudun run datasets/generated/demo.jsonl --mode full --output outputs/run
python -m yudun gui --input datasets/generated/demo.jsonl
```

实机与 ROS 2 的构建、标定、话题和验收步骤见 [Linux 测试说明](docs/LINUX_TESTING.md)。当前 Windows 主机只继续编辑与提交代码；不在此构建 Qt/ROS。

## 代码组成

| 路径 | 内容 |
|---|---|
| `docs/audit/` | 来源、依赖、权重、13 项静态发现、调用图、完整哈希清单 |
| `yudun/schema.py` | 带版本、时戳、坐标系、单位与协方差的观测接口 |
| `yudun/adapters.py` | 工业 YOLOv8 ONNX、像素地面投影、点云背景剔除与聚类 |
| `yudun/tracking.py` | 时戳驱动 CV 滤波、质量降权、马氏距离与一对一关联、可见性生命周期 |
| `yudun/risk.py` | 轨迹采样、最近点时间与间距、风险概率、分级滞回和冷却 |
| `yudun/qt/` | 改造的原 Qt 面板、后台回放与直接 ROS 信号接入 |
| `ros2/src/` | 自定义消息、原始传感器适配节点、融合节点 |
| `scripts/` | 上游审计、按场景划分数据集、独立危险事件评测 |
| `tests/` | 核心、评测工具和 Linux Qt 离屏回归测试 |

## 数据和实验

输入采用 JSONL，所有位置为米、速度为米/秒、时间为秒，统一 `map` 坐标。空观测列表仍需按时送入，表示当前帧无目标；传感器无数据时不发送该源的 heartbeat。详情与独立 GT/危险事件标注模板见 [datasets/README.md](datasets/README.md)。

输出包括 `input.jsonl`、`results.jsonl`、`tracks.csv`、`events.jsonl`、`latency.csv`、配置和 `metrics.json`，记录随机种子、配置与输入哈希。核心处理时间不包括相机采集、模型推理、传输或 GUI 延迟。

感知模式：`visual`、`lidar`、`fixed_fusion`、`quality_fusion`、`full`。其中 `fixed_fusion` 是新写的固定协方差参照，**不是原项目 B3**；B3 必须在 Linux 跑原代码并保留真实输出后单独比较。

```bash
python -m pytest -q
python -m yudun benchmark --repeats 10 --output outputs/benchmark
python scripts/evaluate_events.py outputs/run/input.jsonl outputs/run/results.jsonl datasets/events.json
```

合成回归不能证明工业泛化、实机定位精度或预警效果。独立 GT 不进入估计器。HOTA、概率校准、长时稳定性、跨场景统计与实际预警提前量仍需正式实验。

## 上游保存与提交

原压缩包、解压目录、模型、实采数据、输出日志和个人开发方案不入 Git；审计清单保留其来源与哈希。后续在 Linux 复查原工程时，将两份原始 ZIP 放回项目根目录后解压到 `upstream/`，再运行 `python scripts/audit.py`；不要直接运行未知启动脚本。Qt 所需许可组件已随代码保存，不依赖上游目录存在。

每个功能步骤单独 commit。当前仓库不预设新增代码的开源许可；`qt/vendor` 及上游的原有许可独立保留。
