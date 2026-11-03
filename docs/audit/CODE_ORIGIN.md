# 代码来源审计

审计日期：2026-11-03。方式：ZIP 安全路径检查、完整解压、源码静态阅读、文件清单及 SHA-256。未启动原工程，未验证原工程可编译性和 GPU/硬件行为。`upstream/` 原样保留，包含原始构建产物，不纳入新工程构建。

## 上游 A

RM2026-Radar 的根目录 LICENSE：MIT，Copyright 2025 HKUST ENTERPRIZE Team。README 使用 YSU System 名称，代码仍有 RM2025 与 HKUST 声明。外层文件名不能作为原作者或原创性证据。

## 上游 B

RM2026-lightRadar 的 README 标题为 T-DT 2025 Radar，声明东北大学 T-DT 来源。根目录 LICENSE：MIT，Copyright 2024 Shenxw、2025 mozihe。README 的 Livox 配置与当前 Odin 接入已有差异。

## 许可边界

根目录 MIT 不代表捆绑模型、训练数据、SDK 和所有第三方子目录均由同一许可覆盖。单目包包含 YOLO 系列第三方代码；Odin 驱动有独立 LICENSE。发布、商业部署或重新分发前，应逐项核查对应上游版本和权重来源。当前新模块未复制上游算法实现；NOTICE 保留来源，原目录许可不变。

## 压缩包校验

- `RM2026-lightRadar(2026赛季激光雷达方案）.zip`：401384307 bytes；SHA-256 `0586702f427ac5362ce2017dd174c0ac42538925535bcc79ed875a4456f7c77e`
- `RM2026-Radar(2026赛季单目相机方案）.zip`：288009651 bytes；SHA-256 `075bbd3b69d5f1b9fa531b3139e7887404cc6da79f903d49324479f471a23bce`

## 包内许可文件清单

- `upstream/RM2026-Radar(2026赛季单目相机方案）/LICENSE`
- `upstream/RM2026-lightRadar(2026赛季激光雷达方案）/build/livox_sdk_vendor/livox-sdk-v2.3.0-prefix/src/livox-sdk-v2.3.0/LICENSE.txt`
- `upstream/RM2026-lightRadar(2026赛季激光雷达方案）/build/livox_sdk_vendor/livox-sdk-v2.3.0-prefix/src/livox-sdk-v2.3.0/sdk_core/include/third_party/spdlog/spdlog/fmt/bundled/LICENSE.rst`
- `upstream/RM2026-lightRadar(2026赛季激光雷达方案）/build/livox_sdk_vendor/livox-sdk-v2.3.0-prefix/src/livox-sdk-v2.3.0/sdk_core/src/third_party/FastCRC/LICENSE.md`
- `upstream/RM2026-lightRadar(2026赛季激光雷达方案）/catkin_ws/src/odin_ros_driver/LICENSE`
- `upstream/RM2026-lightRadar(2026赛季激光雷达方案）/LICENSE`
