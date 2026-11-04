# 模型与视频本地资源

资源选择、发布者和许可见 [模型与视频筛选记录](READY_MODELS_AND_VIDEOS.md)。以下路径均相对项目根目录。原始权重、视频、下载回执和运行输出不入 Git；下载脚本与固定版本清单入 Git。

| 资源 | 本地路径 | 文件体积 |
|---|---|---|
| keremberke YOLOv8n | `models/pretrained/keremberke-yolov8n/best.pt` | 6,216,376 bytes |
| YOLO11s forklift/person | `models/pretrained/yolo11s-forklift-person/best.pt` | 19,190,682 bytes |
| RT-DETR Warehouse FP32 | `models/pretrained/rtdetr-warehouse/rtdetr_warehouse_v1.0.2.onnx` | 174,797,668 bytes |
| RT-DETR Warehouse FP16 | `models/pretrained/rtdetr-warehouse/rtdetr_warehouse_v1.0.2.fp16.onnx` | 87,890,438 bytes |
| SmartSpaces 仓库短片 | `datasets/raw/MTMC_Tracking_2026/test/Warehouse_026/videos/Camera_0000.mp4` | 19,802,011 bytes |
| 工业行为视频集 v1 | `datasets/raw/industrial_behaviours/Safe and Unsafe Behaviours Dataset/` | 691 段，9,999,951,582 bytes |

模型卡、模型配置、RT-DETR experiment.yaml，以及 SmartSpaces calibration.json 和数据卡随资源保存。工业数据保持发布方原始目录，train 566 段、test 125 段，共八类。哈希、大小、文件 URL 见 [模型/仓储短片清单](../configs/resource_downloads.json) 和 [全部工业视频清单](../configs/industrial_video_downloads.json)。

## 下载与核对

Windows PowerShell 下载命令如下；仅操作文件，不加载模型、不解码视频。网络需要代理时，为下载命令添加 `-Proxy http://127.0.0.1:端口`，无需修改系统或 Git 全局配置。

```powershell
./scripts/download_resources.ps1 -Manifest configs/resource_downloads.json
./scripts/download_resources.ps1 -Manifest configs/industrial_video_downloads.json
./scripts/resource_inventory.ps1
```

完整批量任务可使用 PowerShell 7 入口 `./scripts/download_all_resources.ps1 -Workers 8`，同样支持 `-Proxy`。它先核对模型和仓库短片，再分片下载全部工业视频；分片失败最多重试三次，完成后自动生成汇总。`outputs/resource-download-status.json` 保存进程 ID 与任务状态，`outputs/download-logs/` 保存分片日志。只有所有 702 个文件均有匹配回执时，最终状态才标记 `complete`。关闭后台进程后可重新执行相同命令恢复；不要同时启动多个批量任务。

下载使用 `.partial` 断点续传。已有最终文件会重新计算 SHA-256。清单含发布方哈希的文件必须匹配后才生成 `.receipt.json`；没有发布方哈希的说明文件记录本地 SHA-256。清单含大小的文件同时核对长度。

可按 `-Group industrial-0` 至 `industrial-3` 分组；每组可再配合 `-ShardCount 3 -ShardIndex 0` 至 `2` 分成互不重叠的任务。不同分片可并行，同一分片不要重复启动。发布方目录重新枚举入口为 `scripts/prepare_industrial_downloads.ps1`，枚举结果写入本地 metadata 目录，不覆盖 Git 固定清单。

汇总保存在 `outputs/resource-download-report.json`：完整资源应为 702/702 个文件（691 段工业视频及 11 个模型/短片/说明文件）。汇总检查下载时生成的哈希回执与当前大小，不重新读取所有视频计算哈希。需要重新校验时重跑下载命令。

## 工业数据的重复内容

按发布方 SHA-256 清单检查，691 个文件只有 658 组不同内容；32 组存在重复，其中 **9 组跨越原始 train/test**。重复清单见 [逐组哈希记录](INDUSTRIAL_DUPLICATE_HASHES.json)。下载保留全部原始文件；Linux 训练与评测前应按内容去重，并按采集序列重新分组，原始 train/test 不能直接用于声称独立评测。这项检查未解码视频，无法排除不同编码或相邻片段造成的进一步泄漏。

## 转移到 Linux

模型、`datasets/raw/` 和回执需连同项目另行复制，Git clone 不包含这些大文件。在 Linux 执行推理、模型格式适配和 Qt 验证，流程见 [筛选记录](READY_MODELS_AND_VIDEOS.md) 与 [单目主线说明](MONOCULAR.md)。本次文件完整性核对不代表已经通过推理或精度验证。
