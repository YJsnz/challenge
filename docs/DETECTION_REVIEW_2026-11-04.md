# 漏检与误检首轮改进：2026-11-04

全部 702 个资源文件校验完成后，进行了本轮检测改进。**人员漏检在所选仓库样例中明显减少；跨场景设备误报仍未解决。** 新选项默认不启用，未更改原有单次推理行为，也未更换原生 Qt 或启用 LiDAR。

## 实现和验证

- 支持按类别设置置信度：YOLO11 候选使用 person=0.15、forklift_or_cart=0.4，保留其他类别的原阈值。
- 可选全帧加 640 像素重叠分块，重叠率 0.5；对内部块边缘的残缺框进行过滤，再按类别全图 NMS 合并。真实图像外边缘的目标仍保留，全帧推理作为补充。
- 检测框改为普通 Python 浮点数，避免真实视频 `visual_detections` 写入 JSONL 时因 NumPy float32 失败。
- Qt 使用的视频/USB 入口和 SmartSpaces 推理已接入 `detector_options`。ROS 相机入口也转发相同选项，但未执行 ROS 运行验证。

WSL Ubuntu 完整测试 **69 passed**，包括阈值经过 NMS 后仍有效、切块坐标还原、重复框合并、内部边缘与真实图像边缘区分、JSON 序列化、相机/数据入口选项传递。sdist/wheel 构建及脱离源码的 wheel 导入检查通过。默认单次推理再次对照 PyTorch，两套模型在原三帧上的框坐标最大差仍小于 0.001 像素。

实现步骤保存在 `2e8f656`、`84d561a`、`70398d5`。逐帧检测框、源视频/模型哈希、参数与耗时见 [对照记录](validation/detection-review-2026-11-04.json)。未执行训练，没有伪造训练元数据。

## 固定仓库样例

视频为 SmartSpaces Warehouse_026 Camera_0000。开发帧为 0、450、900，另检查同视频 300、750、1350 帧；这些帧属于同一场景，不能当作跨场景独立测试。

| 源帧 | 原全帧 0.4 输出框数 | 全帧分类别阈值 | 重叠分块及边缘过滤 |
|---|---:|---:|---:|
| 0 | 1 | 2 | 4 |
| 450 | 0 | 1 | 2 |
| 900 | 0 | 0 | 2 |
| 300 | 1 | 1 | 3 |
| 750 | 1 | 1 | 3 |
| 1350 | 1 | 2 | 3 |
| 合计 | 4 | 7 | 17 |

**框数不等于正确检出数，更不等于准确率。** 代理目视检查发现：分块后能找回远处叉车、首帧的白衣人员和安全帽人员，以及近处叉车驾驶员；未经边缘处理的分块会把大块货架识别为目标，并产生截断人员重复框，本轮处理消除了所选仓库六帧中的这些明显伪框。第 450 帧弯腰人员、第 300 帧货架后人员仍有漏检，部分底边框也不够准确，不能据此声称米制定位精度提升。

这是 `agent_visual_review_pending_human` 的目视参考，尚待人工复核，没有真实检测框 GT，未计算 AP 或正式召回率。

## 跨场景失败案例与耗时

冻结上述参数后，使用已下载的工业视频 `test/0_safe_walkway_violation/0_te10.mp4` 第 0、86、172 帧检查。原方案输出 0、2、1 个框；分类别阈值输出 0、2、2 个框，目视可见新增前景人员检测；分块输出 3、5、3 个框，找回部分人员，同时在第 86 帧把墙面和蓝色设备误识别为叉车/人员，第 172 帧仍把蓝色设备误识别为人员。**因此没有将分块设为通用默认模式。** 这些负例与正例一起保留。

本机 CPU 所选仓库六帧的单次检测约 103–140 ms，分块约 1547–1939 ms；工厂三帧分块约 1417–2171 ms。包含预处理、推理和后处理，不含读帧与 Qt；首次仓库运行同时进行了后端对照，计时只供诊断，不是性能基准。1920×1080 输入下分块共 16 次网络前向，当前用于离线审查，不能宣称实时播放或实机预警达标。

## 如何启用与复现

较保守的 YOLO11 参数位于 `configs/detector.yolo11-thresholds.json`。分块候选位于 `configs/detector.warehouse026-review.json`，仅用于验证过的仓库样例及后续审查。把选定文件的 JSON 内容写入已核实相机标定的 `camera.detector_options`，或模型 sidecar 的 `detector_options` 字段。不要把参数预设当作相机标定，原有 `example_only` 校验继续生效。

相机仍需使用对应的真实模型和类别顺序，例如 YOLO11 导出文件 `models/deployment/yolo11s-forklift-person.onnx` 对应 `["forklift_or_cart", "person"]`。未填写 `detector_options` 时保持原全帧 0.4 阈值。

```bash
source "$HOME/.venvs/yudun/bin/activate"
cd /mnt/d/Code/lidar
QT_QPA_PLATFORM=offscreen python -m pytest -q
python scripts/compare_detector_profiles.py \
  --video datasets/raw/MTMC_Tracking_2026/test/Warehouse_026/videos/Camera_0000.mp4 \
  --model models/deployment/yolo11s-forklift-person.onnx \
  --metadata models/deployment/yolo11s-forklift-person.onnx.json \
  --output outputs/detector-review-new
```

工具比较原全帧、仅分类别阈值和给定分块参数，保存相同源帧的框、耗时、哈希及带框图片，输出目录必须为新目录。可用 `--video`、`--frames` 和 `--profile` 指定其他对照。大文件和截图不入 Git，本机本轮结果位于 `outputs/vision-review/final-warehouse/` 与 `final-control/`。

后续应先补充独立检测框标注，覆盖弯腰、遮挡、机器负例、不同叉车姿态和无目标片段，再决定模型微调、替换及 GPU 加速。不要按这九帧调参后声称跨场景泛化或比赛实测指标通过。
