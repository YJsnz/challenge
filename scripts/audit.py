"""Read-only source audit; writes inventory and reports, never imports upstream."""
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/audit"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def evidence(base, relative, needle):
    path = base / relative
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    for index, line in enumerate(lines, 1):
        if needle in line:
            return f"`{path.relative_to(ROOT).as_posix()}:{index}` — `{line.strip()}`"
    raise RuntimeError(f"missing audit evidence: {relative}: {needle}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a = next((ROOT/"upstream").glob("RM2026-Radar*"))
    b = next((ROOT/"upstream").glob("RM2026-lightRadar*"))
    inventory = []
    for base in (a, b):
        for file in sorted(base.rglob("*")):
            if file.is_file():
                inventory.append({"path": file.relative_to(ROOT).as_posix(), "bytes": file.stat().st_size,
                                  "sha256": digest(file)})
    (OUT/"inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    archives = [{"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in ROOT.glob("*.zip")]
    licenses = [x for x in inventory if re.match(r"(?i)(license|copying|notice)", Path(x["path"]).name)]
    origin = """# 代码来源审计

审计日期：2026-11-03。方式：ZIP 安全路径检查、完整解压、源码静态阅读、文件清单及 SHA-256。未启动原工程，未验证原工程可编译性和 GPU/硬件行为。`upstream/` 原样保留，包含原始构建产物，不纳入新工程构建。

## 上游 A

RM2026-Radar 的根目录 LICENSE：MIT，Copyright 2025 HKUST ENTERPRIZE Team。README 使用 YSU System 名称，代码仍有 RM2025 与 HKUST 声明。外层文件名不能作为原作者或原创性证据。

## 上游 B

RM2026-lightRadar 的 README 标题为 T-DT 2025 Radar，声明东北大学 T-DT 来源。根目录 LICENSE：MIT，Copyright 2024 Shenxw、2025 mozihe。README 的 Livox 配置与当前 Odin 接入已有差异。

## 许可边界

根目录 MIT 不代表捆绑模型、训练数据、SDK 和所有第三方子目录均由同一许可覆盖。单目包包含 YOLO 系列第三方代码；Odin 驱动有独立 LICENSE。发布、商业部署或重新分发前，应逐项核查对应上游版本和权重来源。当前新模块未复制上游算法实现；NOTICE 保留来源，原目录许可不变。

## 压缩包校验

"""
    origin += "\n".join(f"- `{x['path']}`：{x['bytes']} bytes；SHA-256 `{x['sha256']}`" for x in archives)
    origin += "\n\n## 包内许可文件清单\n\n" + "\n".join(f"- `{x['path']}`" for x in licenses)
    (OUT/"CODE_ORIGIN.md").write_text(origin+"\n", encoding="utf-8")
    weights = [x for x in inventory if Path(x["path"]).suffix.lower() in (".engine", ".onnx", ".pt", ".pth", ".trt")]
    dependencies = """# 依赖与资产审计

| 项目 | 声明环境 | 当前发现 | 迁移处理 |
|---|---|---|---|
| 单目 | Ubuntu 22.04、Humble、Python 3.10、CUDA 11.8 | start.sh 包含用户绝对路径、Qt LD_PRELOAD；模型推理与比赛类别绑定 | 独立相机/观测适配器，工业模型单独训练 |
| 激光 | README：Ubuntu 24.04、Jazzy、TensorRT 10、OpenCV 4.6、PCL 1.14 | start.sh 加载 Humble；tdt_vision 强制 C++20、CUDA 路径；包内有 build/install/log 产物 | 仅选取源码包，在干净 Linux 工作区逐包重建 |
| 硬件 | README：Livox Avia、海康相机、NVIDIA GPU | 当前 launch 使用 Odin ROS 驱动，动态点云订阅 /odin1/cloud_raw | 现场确认型号、驱动、时钟和外参后接入 |

以上为源码声明，不是已验证兼容组合。新离线核心仅依赖 NumPy/SciPy；OpenCV 适配器为可选模块。Python 3.10+，本机实际测试版本在 VALIDATION.md 记录。ROS/GPU 不在本机可验证范围。

## 默认权重逐项核对

"""
    for name in ("weights/car_v1.5.engine", "weights/armor_v0.7.engine", "weights/MobileNet_v3.1.pth"):
        dependencies += f"- `{name}`：{'存在' if (a/name).exists() else '缺失'}。\n"
    dependencies += "\n## 包内模型资产\n\n模型存在不代表能在当前 TensorRT/GPU 上加载，更不代表支持工业类别。\n\n"
    dependencies += "\n".join(f"- `{x['path']}` ({x['bytes']} bytes)" for x in weights)
    packages = []
    for file in (b/"src").rglob("package.xml"):
        tree = ET.parse(file).getroot()
        packages.append({"name": tree.findtext("name"), "path": file.relative_to(ROOT).as_posix(),
                         "dependencies": sorted({n.text for n in tree if "depend" in n.tag and n.text})})
    (OUT/"ros_packages.json").write_text(json.dumps(packages, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT/"DEPENDENCIES.md").write_text(dependencies+"\n\n完整 ROS 包依赖见 ros_packages.json。\n", encoding="utf-8")
    findings = [
        ("P1", "默认权重不可解析", a, "config/params.yaml", 'weights/car_v1.5.engine', "三个启用的默认模型路径均不存在；不能直接启动默认推理。"),
        ("P1", "空点云使轨迹老化停止", b, "src/fusion/kalman_filter/src/kalman_filter.cpp", 'if (cloud_xy->points.size() == 0)', "空点云直接 return，跳过预测、超时清理和输出；cluster.cpp 也在空输入时 return。上游快照保留，新实现按空帧推进生命周期。"),
        ("P1", "发布融合结果未赋 header", b, "src/fusion/kalman_filter/src/kalman_filter.cpp", 'vision_interface::msg::DetectResult detect_msg;', "构造 detect_msg 后未设置 header 即发布，无法可靠做时空一致性或延迟评估。"),
        ("P1", "单目 dt 仅更新过程噪声", a, "tracker/CascadeMatchTracker/kalman.py", 'def predict(self, dt = None)', "PositionKalmanFilter.predict(dt) 更新 Q，但不更新初始化时固定的 F；帧率变化时状态传播与噪声传播时间尺度不同。"),
        ("P1", "激光滤波回放依赖墙钟且 F 固定", b, "src/fusion/kalman_filter/include/filter_plus.h", 'dt_ = get_time();', "update_predict_point 计算墙钟 dt_，却没有重建 KF.transitionMatrix 与 Q；不同播放速度可能得到不同估计。"),
        ("P1", "滤波初始速度未初始化", b, "src/fusion/kalman_filter/include/filter_plus.h", 'cv::Mat state(stateSize, 1, type);', "随后只写 state[0] 与 state[2]，速度 state[1]/state[3] 未赋值就交给 statePost，可能引入未定义初始状态。"),
        ("P2", "动态点云分块漏掉尾部", b, "src/lidar/dynamic_cloud/src/dynamic_cloud.cpp", 'int  step = cloud_size / thread_num;', "分块只遍历 thread_num*floor(N/thread_num) 点；不足 thread_num 时完全不处理。TransformCloud 中有相同分块模式。"),
        ("P2", "聚类回调堆分配未释放", b, "src/lidar/cluster/src/cluster.cpp", 'new pcl::PointCloud<pcl::PointXYZ>);', "out_cloud 以裸指针 new，回调内无 delete 或所有权转移；持续运行会累计泄漏。"),
        ("P2", "最新 TF 未按采集时刻查询", b, "src/lidar/dynamic_cloud/src/dynamic_cloud.cpp", '"rm_frame", msg->header.frame_id, tf2::TimePointZero);', "读取最新 TF；若外参是严格固定则影响有限，动态 TF 或离线回放需按消息时间查询。"),
        ("P2", "零坐标被当成无效测量", b, "src/fusion/kalman_filter/src/kalman_filter.cpp", 'if (red_point.x == 0 || red_point.y == 0)', "坐标轴上的合法目标被跳过。工业接口应使用有效标志或空列表。"),
        ("P2", "比赛坐标和身份编码不可迁移", b, "src/fusion/kalman_filter/src/kalman_filter.cpp", '28 - detect_msg.blue_x[i]', "28x15 场地翻转、红蓝阵营与固定六槽位不是通用多目标身份。新接口使用独立 track_id。"),
        ("P2", "获取时间戳未传入跟踪器", a, "main_event_loop.py", 'self.tracker.track(frame, faction=self.faction)', "上方取得 time_stamp，但 track 调用未传入；不能假设速度估计与采集时间一致。"),
        ("P2", "默认配置含比赛标定与推测点", a, "config/params.yaml", 'field/RMUC2025_National.PLY', "场地网格、相机外参、装甲板身份和 guess_pts 不能直接用于仓储。"),
    ]
    text = "# 源码审计结果\n\n静态检查结论；P1 表示阻碍可信运行或时间/状态正确性，P2 表示迁移与稳定性问题。未声称已复现原工程运行故障。原始快照未修改，后续修复应单独建补丁并做 Linux 回归。\n\n"
    for priority, title, base, file, needle, detail in findings:
        text += f"## [{priority}] {title}\n\n{detail}\n\n证据：{evidence(base, file, needle)}\n\n"
    (OUT/"FINDINGS.md").write_text(text, encoding="utf-8")
    (OUT/"REUSE_MATRIX.md").write_text("""# 复用边界与模块调用图

| 上游模块 | 可借鉴或复用的能力 | 新工程处理 |
|---|---|---|
| 单目 driver/hik_camera | 海康采集 | 保留 SDK 边界，尚需硬件测试 |
| 单目 model/yolov8、tracker | 推理、级联匹配思路 | 不加载比赛模型；提供工业 ONNX 适配器、新目标 ID 跟踪 |
| 单目 transform/ray_renderer | 基于场地网格的射线投影 | MVP 使用标定地面交点，非平地需后续网格模型 |
| 激光 localization | GICP 地图定位 | 作为可选上游独立节点；需重建仓储地图与编译验证 |
| dynamic_cloud、cluster | 静态背景剔除、欧氏聚类 | 新 Python 离线适配器，参数化地面和背景距离 |
| kalman_filter | 原后融合基线 | 保留原代码；新 fixed_fusion 不是原 B3 |
| referee、dart/fly、阵营映射 | 比赛通信与规则 | 不进入工业主线 |
| build/install/log | 原机生成产物 | 不复用为可移植发布物 |

```mermaid
flowchart LR
  A[上游 A 相机] --> B[比赛模型与级联跟踪]
  B --> C[像素到比赛场地]
  C --> D[裁判与哨兵通信]
  E[上游 B Odin 点云] --> F[GICP 场地定位]
  F --> G[静态背景剔除]
  G --> H[聚类中心]
  H --> I[原卡尔曼轨迹]
  J[上游 B 视觉检测与解算] --> I
  I --> K[比赛身份输出]
```

新工程：标定观测/JSONL → 时间与坐标校验 → 单源或质量融合 → 几何可见性与有限跟踪 → 概率轨迹风险 → 事件日志/CSV/本地看板。ROS 2 包为可选传输层。独立 GT 仅进入评测，不进入估计器。
""", encoding="utf-8")
    print(f"Audited {len(inventory)} files; {len(weights)} model assets; {len(findings)} findings -> {OUT}")


if __name__ == "__main__":
    main()
