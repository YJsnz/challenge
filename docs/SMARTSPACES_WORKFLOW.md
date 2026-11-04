# SmartSpaces 到 Qt 回放与独立评测

本文命令在 Linux/WSL 内执行。当前已完成首轮运行验证及资源下载，实际范围见 [WSL 验证记录](WSL_VALIDATION_2026-11-04.md)。后续地图集成核对发现并修正了旧版本单应矩阵方向错误，新增发布方内外参交叉回归；此前没有据旧投影结果宣称实采定位精度。

## 文件和接口依据

- [NVIDIA 官方数据说明](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces)：2025/2026 JSON 标注与标定格式。
- [Warehouse_000 文件目录](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces/tree/main/MTMC_Tracking_2026/train/Warehouse_000)：用于核对目录结构。
- [NVIDIA 标定说明](https://developer.nvidia.com/blog/simplifying-camera-calibration-to-enhance-ai-powered-multi-camera-tracking/)介绍图像和地面之间的变换。下载的 Warehouse_026 矩阵经其内外参交叉检查，确认为像素到地面；不能默认再取逆。
- [Ultralytics 训练](https://docs.ultralytics.com/modes/train/)与[导出](https://docs.ultralytics.com/modes/export/)：可选训练工具的调用依据。

实际 GT 文件字段含空格，例如 `object type`、`3d location`，转换器也兼容说明页中的下划线形式；遇到重复别名会拒绝。frame id 从 0 开始，区间为 `[start_frame,end_frame)`。大型 GT 使用 ijson 流式读取；缺失标签帧报错，不当作无人场景。

## 1 安装与选择数据

```bash
source .venv/bin/activate
python -m pip install -e '.[qt,test,data]'
```

先选训练场景 Warehouse_000 的一个相机，另选独立验证场景，例如官方验证集 Warehouse_020。手动从官方目录下载每个场景的 `calibration.json`、`ground_truth.json`、所选相机视频；深度图不属于本轮必需输入。按发行版、split、场景保存到 datasets/raw/，不提交数据文件到 Git。若文件名或相机 ID 不同，以实际 calibration/video 文件为准。

```text
datasets/raw/MTMC_Tracking_2026/
  train/Warehouse_000/
    calibration.json
    ground_truth.json
    videos/Camera_0000.mp4
  val/Warehouse_020/
    calibration.json
    ground_truth.json
    videos/<所选相机>.mp4
```

2026 的真实测试场景只公开视频和标定，未公开 GT，因此不能执行本流程中的有监督评测或标注导出。当前不支持 2024 MOT 文本格式、JRDB 或 LUMPI 格式，不能将这些文件直接交给 SmartSpaces 转换器。

## 2 配置单位和评测区域

```bash
cp configs/smartspaces.example.json configs/smartspaces.local.json
cp configs/smartspaces.validation.example.json configs/smartspaces.validation.local.json
```

两个示例的原点、ROI 和协方差仅展示字段，不代表相机实际有效区域或已校准精度。使用前修改：

| 字段 | 含义 |
|---|---|
| scene_directory | 相对于 spec 文件的本地原始场景目录 |
| release、scene_id、source_split、domain | 实际发行版、场景、官方 train/val/test 和 synthetic/real |
| camera_id、video | 与标定完全一致的 ID 和场景内视频路径；一条序列只使用一个相机 |
| world_unit_to_m | 原始世界单位到米的显式比例，不能把 floor-plan scaleFactor 当米制比例 |
| map_origin_source | 从原始 XY 减去的原点，随后乘 world_unit_to_m |
| roi_bounds_m | 转换到 map 后的固定评测区域；参考 GT 包括其中暂时不可见的目标 |
| obstacles_m | map 系货架矩形；留空时没有几何遮挡知识，不产生已知货架约束的效果 |
| measurement_covariance_m2 | 相机地面定位误差的经验协方差，需之后通过独立数据校准 |
| class_map | 所有模型类别对应的原始类名映射；未映射的原始类别明确排除 |

Person → person；Forklift/PalletTruck 可映射为 forklift_or_cart。机器人类别是否归入 AGV 应结合研究范围显式配置，避免把所有人形机器人都等同于 AGV。

世界坐标计算：`map_xy = (source_xy - map_origin_source) * world_unit_to_m`。模型框的底边中心通过像素到地面的 homography 映射到地面，再进行相同变换。其他数据若明确提供反方向矩阵，应显式声明 `homography_direction=ground_to_image`，不能猜测方向。采用近似平面，不使用 GT 位置生成观测，不从深度图伪造 LiDAR。

## 3 生成独立参考标签

```bash
python -m yudun.data prepare configs/smartspaces.local.json --output datasets/processed/warehouse000
```

输出：reference.jsonl、scene.json、manifest.json。manifest 记录 spec、标定、原始 GT 和参考文件的 SHA-256。

**reference.jsonl 只有参考标签，带 reference_only=true。跟踪器会拒绝直接处理它。** ground_truth 仅含 ROI 内目标；reference_boxes 保留整幅图像的可见目标，供检测器监督训练。这两类标签不进入定位观测生成过程。

## 4 导出分组训练集

```bash
python -m yudun.data manifest configs/smartspaces.local.json configs/smartspaces.validation.local.json --output datasets/processed/splits.json
python -m yudun.data export-yolo datasets/processed/splits.json --output datasets/yolo/warehouse-v1 --sample-every 10 --max-images-per-sequence 100
```

manifest 根据 spec 声明的官方 source_split 生成 train/validation/test。至少需要训练和验证场景，test 可以为空。原场景及其外观增强版本强制归为同一个 scene_group；不同相机、相邻帧不能跨分区。2026 版增强配对写在转换器中。

YOLO 导出保留来源索引 samples.jsonl、数据统计、类别顺序、源文件哈希和许可来源。负样本保存空标签文件，整图中的 ROI 外目标也保留标注，避免污染训练。输出采用 `.partial` 目录，成功后才发布为最终目录；失败时保留现场，重试请选择新目录。

## 5 训练和 ONNX 导出

此步骤需另行准备兼容 Linux/GPU 的 PyTorch 环境及实际本地 YOLOv8 检测初始化权重，不会将比赛装甲板模型转换成工业模型。

```bash
python -m pip install -e '.[train]'
python -m yudun.data train datasets/yolo/warehouse-v1 --weights /path/to/yolov8s.pt --output outputs/train/warehouse-v1 --device 0 --epochs 50
python -m yudun.data export-onnx outputs/train/warehouse-v1/weights/best.pt outputs/train/warehouse-v1/yudun_training.json --output models/warehouse-v1.onnx
```

训练元数据保留初始权重、数据集与最佳权重哈希，以及随机种子和参数。导出要求原始 detection head，拒绝 end-to-end head；检查 ONNX 输出维度。生成 `models/warehouse-v1.onnx.json`，其中包含部署类别顺序、输入尺寸与 ONNX 哈希。当前只支持 batch=1 的 `[1,4+C,N]` 或 `[1,N,4+C]` 原始检测输出。

训练工具、预训练权重和数据各有独立许可；项目 NOTICE 不覆盖这些第三方资产。

## 6 从视频推理生成实际观测

```bash
python -m yudun.data infer configs/smartspaces.local.json datasets/processed/warehouse000/reference.jsonl --model models/warehouse-v1.onnx --metadata models/warehouse-v1.onnx.json --output datasets/processed/warehouse000/observations.jsonl
python -m yudun --config datasets/processed/warehouse000/scene.json run datasets/processed/warehouse000/observations.jsonl --mode monocular --output outputs/warehouse000-visual
python -m yudun --config datasets/processed/warehouse000/scene.json gui --input datasets/processed/warehouse000/observations.jsonl
```

infer 只把视频图像交给模型；GT 在推理后作为参考字段附加，观测 ID 不使用 GT 身份。校验模型、参考数据与当前 spec/标定哈希，以及视频 FPS、尺寸、源帧号；出现解码失败或标签缺失时停止，不伪造空帧或插值轨迹。

Qt 中点击“打开参考视频”选择同一**未裁剪原视频**。即使转换从非零帧开始，回放也按源时间戳对齐，并叠加模型检测框。此数据只有相机，不会产生虚假的激光在线状态。使用输出的 scene.json，避免沿用默认模拟场景的坐标范围。

## 7 独立评测和基线比较

常规 run 已生成定位误差、IDF1、ID Switch、0.5/1/2 秒 ADE/FDE 和 95% 协方差覆盖率。ADE 需要完整的 10Hz 预测时域与对应 GT；没有 GT 的未来不补零、不做真值插值。位置误差和覆盖率只对空间/类别门控后的匹配样本计算，须结合召回率和缺失计数解读。

```bash
python scripts/evaluate_risk_probability.py outputs/warehouse000-visual/input.jsonl outputs/warehouse000-visual/results.jsonl --output outputs/warehouse000-visual/reliability.json
```

风险诊断包括所有符合条件目标对的概率，而不只报警对，计算 Brier score、ECE 与分箱可靠性。危险标签来自未来独立 GT 和声明的安全间距；未来标签不完整或身份不确定的样本单独排除并计数。该工具不拟合概率校准器；连续时间风险、正式危险事件和误报/小时仍需独立事件标注。

原 B3 需先在 Linux 实际运行上游并记录 CSV，然后配置列名、时间偏移与单位变换：

```bash
python -m yudun import-baseline outputs/warehouse000-visual/input.jsonl /path/to/upstream.csv configs/baseline_csv.example.json --output outputs/b3/results.jsonl
python -m yudun evaluate outputs/warehouse000-visual/input.jsonl outputs/b3/results.jsonl --output outputs/b3/metrics.json
```

示例映射不是自动正确的比赛坐标变换；必须按实际导出修改。上游没有提供的速度、协方差、预测和处理时延不生成相应指标。import-baseline 不运行原工程，也不证明外部文件确实来自 B3；须保留其提交版本和原始运行日志。

## Linux 验证顺序

先执行当前全部测试，包括新增数据格式、标签隔离、增强场景泄漏、预测指标、风险诊断与基线单位测试：

```bash
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

再用短视频/少量帧验证 prepare、export-yolo、infer 和 Qt 对齐。最后扩大训练数据，并在未参与训练的独立验证/测试场景上报告结果；不能把训练场景回放指标当成泛化结果。没有模型权重时可完成数据准备和标签导出，但不能宣称视频检测链路已验证。
