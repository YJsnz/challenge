# 数据约定

`generated/` 是带 provenance=synthetic 的合成回归数据，不用于宣称实测性能。

实采数据建议每条完整序列一个 JSONL，逐帧保存 schema_version=1、frame_id=map、timestamp（秒）、sensors（本帧收到有效数据的传感器）、observations。独立参考定位保存在 ground_truth，字段 id、category、position=[x,y]、radius；这些字段不传入跟踪器。

分组清单形如：

```json
[
  {"sequence_id":"layout-a-day-1","path":"real/a.jsonl","scene_group":"layout-a","provenance":"reference_camera"},
  {"sequence_id":"layout-b-day-1","path":"real/b.jsonl","scene_group":"layout-b","provenance":"reference_camera"},
  {"sequence_id":"layout-c-day-1","path":"real/c.jsonl","scene_group":"layout-c","provenance":"reference_camera"}
]
```

用 `python scripts/split_dataset.py datasets/manifest.json` 生成按场景分组的训练/验证/测试划分。同一视频、同一天连续片段或相同采集布局应放入同一 scene_group，避免帧级泄漏。最少三个独立组。

危险事件独立标注文件：

```json
{"events":[{"id":"event-1","pair":["person-1","cart-1"],"start":3.0,"danger_time":5.0,"end":6.0}]}
```

start 为允许匹配预警的窗口起点，danger_time 为首次进入约定安全间距的真值时刻，end 为危险区间终点。必须覆盖无危险的完整时间，以便统计误报/小时；不能仅收录报警成功片段。pair 使用独立 GT 身份，不使用算法 track_id。评测器按 GT 位置关联身份。

检测器训练：标注 person、forklift_or_cart、AGV、unknown_dynamic_object；按序列分组生成 YOLO 数据集。工业 ONNX 权重应有类别清单、数据许可、训练配置和验证记录；当前不提供未经训练的虚构权重。部署适配器支持无 NMS 的 YOLOv8 detect ONNX `[1,4+C,N]` / `[1,N,4+C]` 输出，其他导出格式应单独适配。
