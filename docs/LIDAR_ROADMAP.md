# 激光雷达预留接口与后续规划

2026-11-04：当前交付以固定单目相机为主线，LiDAR 保留为可选升级方向。默认场景、Qt、USB/视频入口和 ROS 相机节点不启动雷达，也不要求雷达在线。

## 已保留的边界

| 边界 | 约定 |
|---|---|
| 原始输入 | 规划话题 /sensors/lidar/points，sensor_msgs/PointCloud2，保留采集时间戳与原始 frame_id |
| 点云处理 | adapters.lidar_observations：外参转换、静态地图剔除、地面高度阈值、欧氏聚类 |
| 独立适配入口 | yudun.lidar.lidar_frame(points, timestamp, source_frame_id, calibration, static_map=None) |
| 统一输出 | schema_version=1、frame_id=map、sensor=lidar、米制位置、2×2 协方差、quality、observation_id |
| 后续 ROS 输出 | 规划话题 /perception/lidar_detections，复用 DetectionArray/Detection3D |
| 跟踪结果 | 保留统一状态 [x,y,vx,vy]、协方差和 sources；历史 ROS 消息结构继续兼容 |

点云聚类默认只输出 unknown_dynamic_object，不从几何中心自动推断 person、叉车或 AGV。空点云应输出空 observations 的有效 heartbeat，断流则停止 heartbeat。static_map 必须是 map 系下的米制 Nx3 点。

## 配置和启用边界

独立示例为 configs/lidar_extension.example.json，默认 enabled=false、example_only=true。实测标定后，显式启用适配函数才允许生成观测；它不包含硬件驱动，也不会自动向当前运行中的单目系统发布。

相机标定仍使用独立的 configs/calibration.example.json，不将 lidar 字段重新塞回单目相机配置。默认 Pipeline 只接受 camera，Qt 不显示雷达故障或点云区域。仅修改扩展配置的 enabled 不会打开融合功能。

## 后续实施顺序

1. 固定单目方案先完成 Linux、标定和院赛演示验证。
2. 根据实借设备选择 Odin/Livox 等驱动，将原始点云独立转换为预留观测帧，验证坐标、距离与空帧行为。
3. 新增显式的扩展运行入口与 /perception/lidar_detections 订阅；确定同步容差、传感器时钟、观测独立性、质量模型及退化策略，再连接历史融合算法。
4. 用同一份独立 GT 对比单目主线与扩展方案，检查遮挡、定位、身份切换和误报变化；需要时为 Qt 增加可选点云页。

当前没有启用自动融合，也没有完成 LiDAR 实机或 ROS 点云适配节点验证。新增接口回归代码保存在 tests/test_lidar_extension.py，按用户要求留到 Linux 执行。
