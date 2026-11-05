# 照微公开数据集筛选

核查日期：2026-11-03。已核对发布方页面与文件目录；本轮未下载数据、未运行构建或测试。以下用途和接入顺序为本项目的选择建议，不是数据集原作者的性能保证。

## 优先级与适用范围

| 优先级 | 数据集 | 数据及标注 | 适合验证 | 主要边界 |
|---|---|---|---|---|
| P0 | NVIDIA PhysicalAI-SmartSpaces，MTMC_Tracking_2026 | 仓储多相机视频、相机标定、2D/3D 框、跨相机 ID、部分深度图 | 工业类别检测、地面投影、身份跟踪、Qt 视频与轨迹回放 | 标注训练集主要为合成数据；不是原始 LiDAR 数据集 |
| P1 | JRDB | 真实 RGB 与 LiDAR、2D/3D 人员框与跟踪 | 异构观测融合、人员遮挡跟踪 | 非叉车仓储类别；移动视角序列需额外处理自身运动 |
| P1 | LUMPI | 路口多视角视频、三维点云与交通参与者标注 | 固定基础设施多传感器融合、遮挡与跨视角关联 | 室外交通场景，不能替代仓储实测 |
| P2 | NVIDIA PhysicalAI SDG-Warehouse | 仓储叉车与人员险情等合成多视角视频、深度、2D/3D 框与相机参数 | 风险场景补充、受控回放 | 没有据此核实真实 LiDAR；场景名称不能代替逐时刻危险真值 |
| 申请型补充 | Forklift Collision Avoidance multispectral dataset | 真实叉车搭载 RGB/热成像、人员距离和遮挡标注 | 仓储行人检测、遮挡分层评测 | 车载视角；须向论文通讯作者申请，仅非营利用途 |

## 1 仓储视觉主线

[NVIDIA 官方数据页](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces) 提供人员、叉车、托盘搬运车等类别，许可标为 CC BY 4.0。2026 版不含深度图约 50 GB；训练/验证以合成为主，两个真实测试场景不公开 GT 与深度，不能据其视频直接计算有监督定位误差。

[已核对的 Warehouse_000 文件目录](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces/tree/main/MTMC_Tracking_2026/train/Warehouse_000) 包含 videos、calibration.json、ground_truth.json、map.png 和 depth_maps。首轮只选一个场景的少量相机视频，加标定、GT、地图；不下载深度图全集。

建议接入：Person → person；Forklift/PalletTruck 可按实验定义合并到 forklift_or_cart；机器人类别是否归入 AGV 应先核对实体定义并保留原始标签。GT 用于评测，不作为检测器输出；相机位置观测必须来自实际模型检测和独立标定。若先用 GT 加噪验证软件，必须标注为人工生成观测。

场景划分注意：官方说明 Warehouse_001/003/005/007/021/024 分别是 000/002/004/006/020/023 的外观增强版本。重划数据时应让原场景和增强版本属于同一 scene_group，避免相同轨迹跨训练/测试泄漏。以上目录、标注、拆分信息见官方数据卡。

## 2 真实相机与 LiDAR 融合

[JRDB 官方数据说明及下载入口](https://jrdb.erc.monash.edu/dataset/) 包含 54 个序列，RGB 与 LiDAR，人员检测/跟踪标注，许可 CC BY-NC-SA 3.0。建议先选静止传感器序列，核对各源时戳和标定；移动机器人视角不能直接当固定 map 观测使用。

[LUMPI 官方页面](https://data.uni-hannover.de/en/dataset/lumpi) 给出同步多视角视频、三维点云、标注以及下载目录和 Python SDK，许可 CC BY-NC 3.0。它适合验证路口融合与遮挡关联，但应用域为室外交通。官网目录页显示的几 MB 是门户附属文件大小，不是完整数据大小；下载前按实际序列文件核算。

JRDB 与 LUMPI 仍需要专用转换器，当前仓库尚未实现。SmartSpaces 已新增流式转换、训练数据导出与模型观测生成代码，使用方法见 [操作流程](SMARTSPACES_WORKFLOW.md)，尚待 Linux 验证。应保留原始时间戳、坐标变换、来源 ID 和参考标签；不能把标签本身当待测算法预测结果。

## 3 风险与仓储行人补充

[NVIDIA SDG-Warehouse 官方数据页](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes) 以合成数据提供叉车与人员近失事件等情节，许可标为 OpenMDW 1.1。RGB 和完整标注分开打包，完整标注为 TB 级；首轮只选匹配的少量 shards。深度反投影生成的点云不具有真实 LiDAR 的扫描和噪声性质，应作为合成深度实验独立标记。

[Forklift Collision Avoidance 原论文](https://www.mdpi.com/1424-8220/22/20/7953) 给出真实叉车采集的 RGB/热成像数据、人员距离及遮挡程度。论文数据可用性声明为向通讯作者申请、仅非营利用途；不能把论文开放许可误当成数据的无限制许可。

## 建议实施顺序

1. Linux 下载 SmartSpaces 单场景少量视频与必要标注，先实现格式转换、训练样本导出和 Qt 参考回放。
2. JRDB 单序列用于真实人员 camera/LiDAR 联合实验；必要时再加 LUMPI 验证固定路侧结构。
3. SDG-Warehouse 少量近失情节用于合成风险回归，独立定义危险窗口、最小间距和事件时间。
4. 采集本项目真实仓储交叉口数据，以独立参考定位获得 GT，验证 person、forklift_or_cart、AGV 和真实误报/漏报。

目前未确认有一套可直接覆盖“固定仓储视角 + 人员/叉车/AGV + 同步真实相机/LiDAR + 遮挡跟踪 ID + 危险事件真值”的完整公开数据。公开数据可分模块使用，最终完整闭环仍需受控实采。
