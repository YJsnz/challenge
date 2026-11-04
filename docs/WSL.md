# Windows 上使用 WSL 验证

本机使用已有的 Ubuntu 24.04.4 / WSL 2、Python 3.12.3 和 WSLg。构建、测试和图形程序均在 Linux 进程内运行。源文件通过 `/mnt/d/Code/lidar` 共享；虚拟环境放在 Linux 文件系统 `$HOME/.venvs/yudun`，避免在 NTFS 上安装大量 Python 小文件。

## 初始化

先在 Windows PowerShell 运行 `wsl -d Ubuntu`，在 Ubuntu 中安装系统依赖：

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends python3-venv libgl1 libegl1 \
  libxcb-xinerama0 libxcb-shape0 libxkbcommon-x11-0 libxcb-cursor0 \
  libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 fonts-noto-cjk
cd /mnt/d/Code/lidar
python3 -m venv "$HOME/.venvs/yudun"
source "$HOME/.venvs/yudun/bin/activate"
python -m pip install --upgrade pip setuptools wheel build
python -m pip install -e '.[qt,test,data]'
```

本机 Ubuntu 的 VHDX 位于 `D:\WSL\Ubuntu`，Linux 环境文件也占用 D 盘空间。不要仅依据 Linux 虚拟磁盘声明的容量判断 Windows 宿主机剩余空间。

## 自动验证和桌面启动

```bash
source "$HOME/.venvs/yudun/bin/activate"
cd /mnt/d/Code/lidar
bash scripts/validate_linux.sh
# WSLg 显示原生 Qt 窗口；不需要网页服务。
QT_QPA_PLATFORM=xcb python -m yudun gui
# 自动打开回放窗口、保存截图并关闭（限时 15 秒）。
QT_QPA_PLATFORM=xcb python scripts/smoke_qt.py
```

自动验证默认输出到 `outputs/linux-validation/<UTC时间>/`：环境与版本、pytest XML/日志、相机断流合成回放、八个场景 × 两种模式 × 两个种子的 32 组回归、sdist 和 wheel，以及脱离源码目录的 wheel 导入检查。输出目录应为新目录，避免与旧报告混淆。

`QT_QPA_PLATFORM=offscreen` 仅适用于无窗口测试，不能据此宣称桌面已启动。若 xcb 加载失败，可用 `ldd "$HOME/.venvs/yudun/lib/python3.12/site-packages/PyQt5/Qt5/plugins/platforms/libqxcb.so"` 检查缺失库。本机首次启动缺少 `libxcb-shape.so.0`，安装上表中的 `libxcb-shape0` 后解决。

## 公开预训练模型

为了先验证可复现的 CPU 基线，可显式安装 CPU 版 PyTorch，再安装可选导出工具，避免首次安装同时拉取完整 CUDA 运行时：

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[train]'
YOLO_AUTOINSTALL=false python scripts/validate_pretrained.py
```

权重和视频无需复制进 Linux，直接使用项目 `models/` 与 `datasets/raw/`。资源来源和校验规则见 [资源下载说明](RESOURCE_DOWNLOADS.md)。`validate_pretrained.py` 核对两套 YOLO 的发布方哈希、导出 opset 12 原始输出 ONNX，并比较第 0、450、900 帧的 PyTorch/OpenCV 检测结果。输出位于 `models/deployment/` 与 `outputs/wsl-validation/models/`，每个模型有来源和哈希 sidecar；不执行训练，也不伪造训练记录。三帧冒烟测试不构成准确率评估。

本机 WSL 能看到 RTX 4060，但安装 CPU 版 PyTorch 不代表已验证 CUDA 推理。WSL 的 USB 相机还涉及设备接入；本地视频验证不能替代工业相机或实机标定。当前主线不需要 ROS；此 Ubuntu 24.04 环境未用于 ROS 2 Humble 构建。
