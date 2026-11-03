# 源码审计结果

静态检查结论；P1 表示阻碍可信运行或时间/状态正确性，P2 表示迁移与稳定性问题。未声称已复现原工程运行故障。原始快照未修改，后续修复应单独建补丁并做 Linux 回归。

## [P1] 默认权重不可解析

三个启用的默认模型路径均不存在；不能直接启动默认推理。

证据：`upstream/RM2026-Radar(2026赛季单目相机方案）/config/params.yaml:2` — `weights_path: "weights/car_v1.5.engine" # Path to the car detection model weights`

## [P1] 空点云使轨迹老化停止

空点云直接 return，跳过预测、超时清理和输出；cluster.cpp 也在空输入时 return。上游快照保留，新实现按空帧推进生命周期。

证据：`upstream/RM2026-lightRadar(2026赛季激光雷达方案）/src/fusion/kalman_filter/src/kalman_filter.cpp:94` — `if (cloud_xy->points.size() == 0)`

## [P1] 发布融合结果未赋 header

构造 detect_msg 后未设置 header 即发布，无法可靠做时空一致性或延迟评估。

证据：`upstream/RM2026-lightRadar(2026赛季激光雷达方案）/src/fusion/kalman_filter/src/kalman_filter.cpp:166` — `vision_interface::msg::DetectResult detect_msg;`

## [P1] 单目 dt 仅更新过程噪声

PositionKalmanFilter.predict(dt) 更新 Q，但不更新初始化时固定的 F；帧率变化时状态传播与噪声传播时间尺度不同。

证据：`upstream/RM2026-Radar(2026赛季单目相机方案）/tracker/CascadeMatchTracker/kalman.py:243` — `def predict(self, dt = None) -> tuple[List[float], List[float]]:`

## [P1] 激光滤波回放依赖墙钟且 F 固定

update_predict_point 计算墙钟 dt_，却没有重建 KF.transitionMatrix 与 Q；不同播放速度可能得到不同估计。

证据：`upstream/RM2026-lightRadar(2026赛季激光雷达方案）/src/fusion/kalman_filter/include/filter_plus.h:152` — `dt_ = get_time();`

## [P1] 滤波初始速度未初始化

随后只写 state[0] 与 state[2]，速度 state[1]/state[3] 未赋值就交给 statePost，可能引入未定义初始状态。

证据：`upstream/RM2026-lightRadar(2026赛季激光雷达方案）/src/fusion/kalman_filter/include/filter_plus.h:61` — `cv::Mat state(stateSize, 1, type);`

## [P2] 动态点云分块漏掉尾部

分块只遍历 thread_num*floor(N/thread_num) 点；不足 thread_num 时完全不处理。TransformCloud 中有相同分块模式。

证据：`upstream/RM2026-lightRadar(2026赛季激光雷达方案）/src/lidar/dynamic_cloud/src/dynamic_cloud.cpp:51` — `int  step = cloud_size / thread_num;`

## [P2] 聚类回调堆分配未释放

out_cloud 以裸指针 new，回调内无 delete 或所有权转移；持续运行会累计泄漏。

证据：`upstream/RM2026-lightRadar(2026赛季激光雷达方案）/src/lidar/cluster/src/cluster.cpp:26` — `new pcl::PointCloud<pcl::PointXYZ>);`

## [P2] 最新 TF 未按采集时刻查询

读取最新 TF；若外参是严格固定则影响有限，动态 TF 或离线回放需按消息时间查询。

证据：`upstream/RM2026-lightRadar(2026赛季激光雷达方案）/src/lidar/dynamic_cloud/src/dynamic_cloud.cpp:167` — `"rm_frame", msg->header.frame_id, tf2::TimePointZero);`

## [P2] 零坐标被当成无效测量

坐标轴上的合法目标被跳过。工业接口应使用有效标志或空列表。

证据：`upstream/RM2026-lightRadar(2026赛季激光雷达方案）/src/fusion/kalman_filter/src/kalman_filter.cpp:55` — `if (red_point.x == 0 || red_point.y == 0)`

## [P2] 比赛坐标和身份编码不可迁移

28x15 场地翻转、红蓝阵营与固定六槽位不是通用多目标身份。新接口使用独立 track_id。

证据：`upstream/RM2026-lightRadar(2026赛季激光雷达方案）/src/fusion/kalman_filter/src/kalman_filter.cpp:184` — `detect_msg.blue_x[i] = 28 - detect_msg.blue_x[i];`

## [P2] 获取时间戳未传入跟踪器

上方取得 time_stamp，但 track 调用未传入；不能假设速度估计与采集时间一致。

证据：`upstream/RM2026-Radar(2026赛季单目相机方案）/main_event_loop.py:325` — `self.tracker.track(frame, faction=self.faction)`

## [P2] 默认配置含比赛标定与推测点

场地网格、相机外参、装甲板身份和 guess_pts 不能直接用于仓储。

证据：`upstream/RM2026-Radar(2026赛季单目相机方案）/config/params.yaml:59` — `mesh_path: "field/RMUC2025_National.PLY"       ## 全国赛场地文件`

