# 现成视觉模型与测试视频

核查日期 2026-11-04。已读取发布方模型卡、文件清单和视频目录，并准备本地资源；路径和下载核对方式见 [资源下载说明](RESOURCE_DOWNLOADS.md)。后续已在 WSL Ubuntu 验证两套 YOLO 的加载、ONNX 导出与三帧真实视频推理，详见 [验证记录](WSL_VALIDATION_2026-11-04.md)。未原生执行 Windows 推理。下述“现成”表示公开训练权重，场景效果仍需独立评测。

## 可免训练试用的模型

| 候选 | 已核实资源 | 当前工程接入 |
|---|---|---|
| [keremberke YOLOv8n forklift detection](https://huggingface.co/keremberke/yolov8n-forklift-detection) | 文件列表含 best.pt；模型卡类别 forklift/person；原说明基于较旧 Ultralytics 8.0.21 | WSL 已验证加载、ONNX 导出及当前检测器推理；样例存在明显误检 |
| [mustafahacioglu YOLO11s forklift/person](https://huggingface.co/mustafahacioglu/yolo11s-forklift-person-detection) | 文件列表含 best.pt；类别 0=forklift、1=person；模型卡标 AGPL-3.0 | WSL 已验证 ONNX 导出及当前检测器推理；样例仍有漏检 |
| [NVIDIA RT-DETR 2D Warehouse](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/rtdetr_2d_warehouse/deployable_rn50_v1.0.2/file-browser) | 官方目录列出 v1.0.2 ONNX/FP16 ONNX，覆盖 Person、Forklift 等仓储类，模型卡使用 NVIDIA Open Model License | 已通过 ONNX 结构检查；含 NVIDIA 自定义算子，推理及专用后端/解码仍待适配 |

Hugging Face API 核对到的权重 revision：

- keremberke/yolov8n-forklift-detection：d8b594f321dd4c64eda16e435601fd266089fc32。
- mustafahacioglu/yolo11s-forklift-person-detection：931258e2715f401f8a0535daa814c27d39ab4129。

keremberke 模型卡未给出明确的权重 license 字段，不能据此认定为 MIT 或无限制授权；重新分发前应核实。YOLO11s 的训练集含多个来源，模型卡的 AGPL 声明不能替代全部训练数据许可核查。发布者自报分数只作选型参考，不作为照微实测结果。

## 可直接下载的视频

### 首选真实仓储短片

[SmartSpaces Warehouse_026 Camera_0000 原视频下载](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces/resolve/main/MTMC_Tracking_2026/test/Warehouse_026/videos/Camera_0000.mp4)

[对应 calibration.json](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces/resolve/main/MTMC_Tracking_2026/test/Warehouse_026/calibration.json)

发布方 [数据卡](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces) 说明 Warehouse_026/027 为真实固定相机测试场景，每相机提供首 60 秒视频与标定，GT 和深度未公开。API 确认 Camera_0000.mp4 为 19,802,011 bytes，约 20 MB。适合快速测试检测、跟踪与 Qt 画面；缺少 GT 时不能据此计算正式定位误差或危险事件召回。视频许可为 CC BY 4.0。

若需要可量化指标，应另用 SmartSpaces train/val 中公开 GT 的场景，按现有单相机数据流程划分与评测，并标明合成来源。

### 真实工业监控视频集

[Video Dataset for Safe and Unsafe Behaviours](https://data.mendeley.com/datasets/xjmtb22pff/1)，作者 Oğuzhan Önal、Emre Dandıl。

发布页提供 691 段真实生产现场 MP4，1920×1080、24 FPS，包含安全通行、通道违规、叉车安全/超载搬运等行为，CC BY 4.0。可从文件列表选取含人员与叉车的片段；并非每段都有交汇风险，也没有据此核实米制标定和逐目标轨迹 GT。

### 未作为独立评测推荐的演示组合

[Forklift-pedestrian-proximity](https://github.com/sounak-sarkar-gwc/Forklift-pedestrian-proximity) 的 README 列出 best.pt 和 test_clip_nearmiss.mp4，但明确用该片段抽取的 24 帧补充训练。该组合可供理解演示流程，不能把同片段效果当作独立测试；本轮未确认其视频与权重的独立再分发许可。

## Linux 上最小检测试用

以下仅运行公开模型检测，不依赖本项目训练记录。先在 Linux 独立环境安装 ultralytics 和 huggingface_hub，再执行：

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

weights = hf_hub_download(
    "mustafahacioglu/yolo11s-forklift-person-detection", "best.pt",
    revision="931258e2715f401f8a0535daa814c27d39ab4129")
video = hf_hub_download(
    "nvidia/PhysicalAI-SmartSpaces",
    "MTMC_Tracking_2026/test/Warehouse_026/videos/Camera_0000.mp4",
    repo_type="dataset")
model = YOLO(weights)
for result in model.predict(source=video, stream=True, save=True, conf=0.25):
    pass
```

接入照微时，保持输出通道顺序，将 forklift 映射为 forklift_or_cart、person 映射为 person；不能把叉车映射为 COCO truck。当前 OnnxDetector 读取映射后的 class_names，要求 batch=1、原始 `[1,4+C,N]` 或 `[1,N,4+C]` 输出。现有 yudun.data export-onnx 入口绑定本项目训练元数据，外部预训练模型需要单独导出、检查输出并建立来源/哈希记录，不能直接伪造训练记录。

下载的 SmartSpaces 标定需通过其 homography 数据适配流程使用；不能直接把它改名为本项目 camera.local.json。仅有检测框尚不能得到可信的米制间距、速度或预警提前量。
