# 依赖与资产审计

| 项目 | 声明环境 | 当前发现 | 迁移处理 |
|---|---|---|---|
| 单目 | Ubuntu 22.04、Humble、Python 3.10、CUDA 11.8 | start.sh 包含用户绝对路径、Qt LD_PRELOAD；模型推理与比赛类别绑定 | 独立相机/观测适配器，工业模型单独训练 |
| 激光 | README：Ubuntu 24.04、Jazzy、TensorRT 10、OpenCV 4.6、PCL 1.14 | start.sh 加载 Humble；tdt_vision 强制 C++20、CUDA 路径；包内有 build/install/log 产物 | 仅选取源码包，在干净 Linux 工作区逐包重建 |
| 硬件 | README：Livox Avia、海康相机、NVIDIA GPU | 当前 launch 使用 Odin ROS 驱动，动态点云订阅 /odin1/cloud_raw | 现场确认型号、驱动、时钟和外参后接入 |

以上为源码声明，不是已验证兼容组合。新离线核心仅依赖 NumPy/SciPy；OpenCV 适配器为可选模块。Python 3.10+，本机实际测试版本在 VALIDATION.md 记录。ROS/GPU 不在本机可验证范围。

## 默认权重逐项核对

- `weights/car_v1.5.engine`：缺失。
- `weights/armor_v0.7.engine`：缺失。
- `weights/MobileNet_v3.1.pth`：缺失。

## 包内模型资产

模型存在不代表能在当前 TensorRT/GPU 上加载，更不代表支持工业类别。

- `upstream/RM2026-Radar(2026赛季单目相机方案）/training_workspace/training_20260220_200439/models/best_armor_classifier_MobileNetClassifier.pth` (27189835 bytes)
- `upstream/RM2026-Radar(2026赛季单目相机方案）/training_workspace/training_20260220_200625/models/best_armor_classifier_MobileNetClassifier.pth` (27189835 bytes)
- `upstream/RM2026-Radar(2026赛季单目相机方案）/training_workspace/training_20260221_084058/models/best_armor_classifier_MobileNetClassifier.pth` (27189835 bytes)
- `upstream/RM2026-Radar(2026赛季单目相机方案）/training_workspace/training_20260221_091153/models/best_armor_classifier_MobileNetClassifier.pth` (27189835 bytes)
- `upstream/RM2026-Radar(2026赛季单目相机方案）/yolo/Car-yolov8s.pt` (11552322 bytes)
- `upstream/RM2026-Radar(2026赛季单目相机方案）/yolo/praysky_coord_noe2e_0331_576x768.onnx` (5956277 bytes)
- `upstream/RM2026-Radar(2026赛季单目相机方案）/yolov8/ONNX/armor_yolo.onnx` (44615603 bytes)
- `upstream/RM2026-Radar(2026赛季单目相机方案）/yolov8/ONNX/classify.onnx` (8903716 bytes)
- `upstream/RM2026-Radar(2026赛季单目相机方案）/yolov8/ONNX/RM2025.onnx` (44615599 bytes)
- `upstream/RM2026-lightRadar(2026赛季激光雷达方案）/model/ONNX/armor_yolo.onnx` (44615603 bytes)
- `upstream/RM2026-lightRadar(2026赛季激光雷达方案）/model/ONNX/classify.onnx` (44710935 bytes)
- `upstream/RM2026-lightRadar(2026赛季激光雷达方案）/model/ONNX/RM2025.onnx` (44615599 bytes)
- `upstream/RM2026-lightRadar(2026赛季激光雷达方案）/model/TensorRT/armor_yolo.engine` (52364276 bytes)
- `upstream/RM2026-lightRadar(2026赛季激光雷达方案）/model/TensorRT/classify.engine` (51450324 bytes)
- `upstream/RM2026-lightRadar(2026赛季激光雷达方案）/model/TensorRT/yolo.engine` (64086212 bytes)

完整 ROS 包依赖见 ros_packages.json。
