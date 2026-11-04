# WSL Linux 验证记录：2026-11-04

已在现有 Ubuntu 24.04.4 / WSL 2 中完成首轮软件验证。安装与复现步骤见 [WSL 操作说明](WSL.md)，结构化结果、模型来源和构建产物哈希见 [验证证据](validation/wsl-2026-11-04.json)。本轮没有原生运行 Windows Python 或 Windows 构建。

## 环境与受测版本

WSL 内核 `6.6.87.2-microsoft-standard-WSL2`，Python 3.12.3，CPU Intel Core i7-14650HX。核心依赖 NumPy 2.5.4、SciPy 1.18.1、PyQt5 5.15.11、OpenCV 5.0.0、pytest 9.2.0、ijson 3.6.0。模型导出环境为 Ultralytics 8.4.202、PyTorch 2.15.0+cpu、ONNX 1.23.2。

核心测试及打包基于 `eca7393472332f119fba1d041e8ad70c669c4dea`；随后仅添加 WSL 指南及验证工具。模型和 Qt 冒烟脚本内容对应 `02d4431`，未修改应用算法。Linux 虚拟环境为 `$HOME/.venvs/yudun`。

## 已完成项目

| 检查 | 结果 | 本机输出 |
|---|---|---|
| 核心、数据、评测、单目、LiDAR 预留接口及 Qt 离屏回归 | **59 passed，0 skipped** | `outputs/wsl-validation/baseline/pytest.xml`、`pytest.log` |
| 依赖完整性 | `pip check` 通过，含模型依赖安装后复查 | `baseline/pip-check.log`、`model-requirements.txt` |
| 相机断流合成回放 | 120 帧完成，输出轨迹、事件与评测 | `baseline/replay/` |
| 合成回归矩阵 | 八场景 × 两模式 × 两种子，共 **32 组**，每组 120 帧 | `baseline/benchmark/summary.json` |
| sdist / wheel 构建 | 均成功 | `baseline/dist/yudun-0.1.0.tar.gz`、`yudun-0.1.0-py3-none-any.whl` |
| wheel 安装与包资源检查 | 脱离源码目录导入，单目默认配置、Qt 许可和 CLI 均可读取 | `baseline/wheel-smoke.log`、`wheel-cli-help.txt` |
| WSLg 原生 Qt 窗口 | xcb 平台显示，120 帧回放加载，两个单目模式，正常关闭 | `outputs/wsl-validation/qt-wslg.png`、`qt-wslg.json` |

Qt 桌面第一次启动发现缺少 `libxcb-shape.so.0`，安装 `libxcb-shape0` 后通过。没有通过修改应用代码绕过图形依赖。截图使用合成场景，左侧明确显示没有真实摄像画面。

## 真实视频与预训练模型

输入为已校验的 SmartSpaces `Warehouse_026/Camera_0000.mp4`，解码确认 1920×1080、30 FPS、1800 帧。测试第 0、450、900 帧，阈值 confidence=0.4、IoU=0.45。两套权重均先核对发布方 SHA-256，再导出 640×640、batch=1、opset 12、无内置 NMS 的 ONNX，输出均为 `[1,6,8400]`。

| 模型 | OpenCV 与 PyTorch 输出对照 | 三帧检测数量 | 本次 CPU 检测耗时 |
|---|---|---|---|
| YOLO11s forklift/person | 数量和类别一致，框坐标最大差 0.000489 像素以内 | 1、0、0 | 约 272–309 ms/帧 |
| keremberke YOLOv8n | 数量和类别一致，框坐标最大差 0.000245 像素以内 | 1、2、1 | 约 115–118 ms/帧 |

耗时仅为三次 `OnnxDetector.detect` 调用，包含预处理/解码，不含采集、视频读取、模型加载、Qt 或预警流水线；不能据此宣称稳定实时帧率。采样可视化已看到漏检、合并人员框，以及 YOLOv8n 将大块货架区域误检为叉车。后端对照说明这些样例问题已出现在模型输出中，不能将模型下载和推理成功当作场景效果达标。

导出文件位于 `models/deployment/{yolo11s-forklift-person,keremberke-yolov8n}.onnx`，相邻 `.onnx.json` 保存映射后的类别顺序、权重/ONNX 哈希、固定版本来源 URL 和外部预训练标记。未执行本项目训练，也未伪造训练元数据。逐帧结果和检测图位于 `outputs/wsl-validation/models/`。

NVIDIA RT-DETR FP32 / FP16 通过 `onnx.checker` 结构检查。图中包含 `nvidia` 自定义域，FP16 另含 `trt.plugins`；没有在 OpenCV 或 CUDA/TensorRT 上执行推理，仍需对应后端和专用预处理/解码适配。

## 尚未覆盖

本轮没有验证 CUDA 推理、ROS 构建、USB/工业相机驱动、现场内外参、真实米制定位误差、独立危险事件指标或长时运行。三帧真实视频没有检测 GT，未计算准确率、召回率或 AP；32 组回归均为合成数据。程序运行验证已完成第一轮，实采效果与比赛验收仍需后续实验。

大文件、截图和原始运行输出保留本机，不入 Git。仓库保存复现工具、操作说明和小型证据 JSON。
