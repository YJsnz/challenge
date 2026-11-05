# 单目相机与三维地图

固定相机的检测框底边中心经去畸变后形成一条射线，与标定的地面相交得到地面 XY。跟踪和风险模块继续在地面坐标系运行，配准后的轨迹、观测、速度、位置协方差和预测点同步写入三维地图坐标系。地图 XYZ 代表**人员或车辆的落地点**。地面倾斜时，地图 Z 随落地点变化。目标高度和离地物体的位置仍需其他测量。

## 使用仓库录像演示

内置图像生成以 SmartSpaces 的 Warehouse_026 / Camera_0000 第 450 帧为参考，制作了[仓库通道参考图](../yudun/assets/warehouse026-demo-plan.png)。根据[独立的示意布局](../yudun/assets/warehouse026-demo-layout.json)生成可旋转的米制点云。布局包含假定的 8 × 20 米场地、4.2 米货架、通道与托盘。参考图的[生成提示词](../yudun/assets/warehouse026-demo-plan.prompt.txt)和素材哈希随项目保留。点云和图片都标为示意，录像中的厂房尺寸没有实测。

在 WSL Ubuntu 中运行，输出目录需为新目录：

```bash
cd /mnt/d/Code/lidar
source "$HOME/.venvs/yudun/bin/activate"
scene=datasets/raw/MTMC_Tracking_2026/test/Warehouse_026
python -m yudun.map3d demo --output outputs/warehouse-demo \
  --source-calibration "$scene/calibration.json" \
  --model models/deployment/yolo11s-forklift-person.onnx
python -m yudun gui --video "$scene/videos/Camera_0000.mp4" \
  --calibration outputs/warehouse-demo/camera.example.json --allow-demo-calibration
```

录像推理采用发布方的相机内参和外参，将相机朝向对齐到示意通道。软件核对了发布方单应矩阵与内外参的方向。录像示例须显式启用，仅接受本地文件。CPU 推理可能慢于录像帧率，Qt 按已处理帧更新。

用原 Warehouse_026 标定生成的演示会标记疑似司机，并在证据满足时暂停司机与本车的风险配对。适用范围和误报对照见[叉车司机风险配对复核](FORKLIFT_OPERATOR_REVIEW_2026-11-05.md)。

中间面板可切换**地面 XY**、**三维地图**和**参考俯视图**。三维地图支持左键旋转、右键平移、滚轮缩放、双击复位。人员为绿色，车辆为蓝色，虚线为预测路径，空心标记表示当前目标处于预测状态。轨迹表提供地图 XYZ，点选一行可在两种地图视图高亮该目标。检测框与地图标记分别显示模型检测和地面落点。

没有下载录像和模型时，可以先验证坐标链路：

```bash
python -m yudun.map3d demo --output outputs/geometry-demo
python -m yudun --config outputs/geometry-demo/tracking_scene.json gui \
  --input outputs/geometry-demo/observations.jsonl
```

几何样例有 61 帧明确标注的模拟观测。录像运行将输入和结果写入 `outputs/live/<运行时间>/`。回放可用 Qt 的导出按钮写出 `map_tracks.csv`。也可从录像结果单独导出：

```bash
run_dir=outputs/live/20261105_151748_613090  # 换成面板本次运行的目录
python -m yudun.map3d project "$run_dir/results.jsonl" \
  --map-scene outputs/warehouse-demo/map.json --output outputs/map-export
```

`results.jsonl` 保留原来的地面跟踪字段。新增的 `map_3d` 包含地图坐标系名称、地图与几何哈希、示例标记、落地点 XYZ、速度 XYZ、观测、位置协方差和预测。协方差由地面误差传播到三维，地图配准误差应在独立验收中另行统计。

## 换成实测地图

1. **准备点云与控制点。** 支持 PCD 文本、二进制和压缩二进制，PLY 顶点点云，以及 NPY 的 XYZ 或 XYZ/RGB 数组。原始点云可直接通过 Qt 打开并选择单位，预览状态为未配准。用 `python -m yudun.map3d inspect path/to/cloud.pcd` 检查点数和抽样范围。
2. **地面与地图配准。** 在请求 JSON 里给出 `geometry`、`unit_to_m`、`frame_id`、`ground_points_m`、`map_points_m`、`check_points` 和 `max_registration_error_m`。拟合至少需要三个非共线的同名点，地面点为 `[x,y,0]`，地图点为米制 XYZ。实测请求用 `example_only: false` 并在 `check_points` 中提供另外至少三个同名点，结构同拟合点。用 `ground_bounds_m: [xmin,ymin,xmax,ymax]` 定义跟踪范围，`ground_obstacles_m` 可写入同结构的货架区域列表。配准求旋转和平移，源坐标缩放由用户指定。
3. **标定固定相机。** 请求 JSON 需提供 `map_scene`、`intrinsics`、`distortion`、`image_width`、`image_height`、`model`、`class_names`、`control_points`、`check_points` 和地面测量协方差 `measurement_covariance_m2`。拟合至少用四对控制点，另外至少三对地面检查点。两组均采用 `{"map_points_m": [[X,Y,Z], ...], "image_points_px": [[u,v], ...]}`。示意包里的 `calibration-request.example.json` 提供完整请求结构。填写实测内参和对应点后，用 `max_reprojection_error_px` 与 `max_position_error_m` 设置验收限值。软件生成相机文件、配套跟踪配置和检查报告。
4. **独立复查。** 使用 `{"calibration": "path/to/camera.json", "check_points": {...}}` 检查其他点。报告包含像素重投影误差、米制位置误差和检查点的地面偏移。相机移动、变焦、分辨率变化或地图更新后重新标定。

```bash
python -m yudun.map3d register register-request.json --output outputs/registered
python -m yudun.map3d calibrate camera-request.json --output outputs/calibrated
python -m yudun.map3d check recheck-request.json --output outputs/independent-check.json
python -m yudun gui --camera-index 0 --calibration outputs/calibrated/camera.json
```

相对路径以各自 JSON 所在目录为基准。旧版实机标定可继续使用原地面 XY 功能。三维接入要求地图与相机标定绑定，加载时核验文件哈希。离线导入旧观测和旧结果前，应核对原地面 XY 的原点、朝向和尺度。当前三维接入验证覆盖本地录像和 JSONL 回放，现场相机仍待实际标定与验收。

点云、标定、录像与打包的结果见 [WSL 验证记录](MAP3D_VALIDATION_2026-11-05.md)。
