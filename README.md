# 🚗 Highway Emergency Lane Violation Detector

基于 YOLOv12 + HyperLPR3 + OpenVINO/CUDA GPU 的高速公路应急车道违章检测系统，自动识别占用应急车道的车辆并记录车牌号、车牌颜色、违章时间戳，按车牌生成一一对应的截图和 15 秒证据视频，并输出 HTML 检测报告。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![OpenVINO](https://img.shields.io/badge/OpenVINO-2026.1-purple)

## ✨ 功能特点

- 🔍 **自动违章检测**：基于 YOLOv12 目标检测 + 多目标跟踪，自动识别占用应急车道的车辆
- 🔟 **车牌识别**：HyperLPR3 多策略识别（多帧扫描 + CLAHE增强 + 2x放大），识别率极高
- 🎨 **车牌颜色识别**：HSV色彩空间分析，支持蓝牌/黄牌/绿牌/白牌/黑牌5种
- ⚡ **GPU全链路加速**：OpenVINO GPU 加速，YOLOv12s 与车牌识别全链路 GPU 推理
- 🎬 **15秒举报视频**：FFmpeg无损剪辑，保留违章前3秒
- 📸 **违章截图**：4K原图截取最佳车牌识别帧
- 📊 **HTML检测报告**：含违章详情、车牌信息、视频片段索引、内嵌截图
- 🖥️ **GUI界面**：tkinter图形界面，参数可调，实时进度
- 🌐 **Web界面**：Flask Web服务，内网多机访问，支持视频上传和结果下载

## 🏗️ 项目结构

```
highway-emergency-lane-detector/
├── emergency_lane/                    # 应急车道违章检测（核心模块）
│   ├── run_gpu_v8.py                  # ⭐ CLI 主程序（推荐验证链路）
│   ├── traffic_violation_gui.py       # GUI 图形界面 / 共享检测流程
│   ├── evidence_utils.py              # 证据视频剪辑工具
│   ├── report_utils.py                # HTML 报告 / manifest / 去重工具
│   ├── lpr3_openvino.py               # HyperLPR3 OpenVINO GPU 加速版
│   ├── lpr3_ort.py                    # HyperLPR3 ONNXRuntime CUDA 版
│   ├── gpu_backend.py                 # GPU 后端探测与设备选择
│   ├── utils.py                       # 共享工具函数
│   └── models/                        # 模型配置模块
│       ├── config.py                  # 统一模型路径（自动搜索多级目录）
│       └── __init__.py
├── models/                            # 模型文件目录（用户放置）
├── setup_models.py                    # 模型下载/检查脚本
├── requirements.txt                   # Python 依赖
├── 应急车道违章检测.pyw                # 双击启动 GUI
├── web_server.py                      # Web 服务
└── start_web.pyw                      # Web 服务控制台 / 启动 / 停止 / 重启 / 状态
```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- FFmpeg（视频剪辑需要）
- Intel 集成显卡（可选，用于 OpenVINO GPU 加速）

### 安装依赖

```bash
pip install -r requirements.txt

# HyperLPR3 OpenVINO版（需要从源码安装）
cd lpr3_openvino && pip install -e .
```

### 下载模型

```bash
# 一键检查/下载所有模型
python setup_models.py

# 只检查状态（不下载）
python setup_models.py --check

# 手动方式：YOLOv12s 模型（首次运行 ultralytics 会自动下载到缓存）
# 导出为 OpenVINO 格式（GPU 加速需要）
python -c "from ultralytics import YOLO; YOLO('yolo12s.pt').export(format='openvino')"
```

> 模型路径优先级：项目 `models/` 目录 → 用户缓存目录 → 自动下载  
> HyperLPR3 模型会在首次运行时自动下载到 `~/.hyperlpr3/` 目录，也可手动复制到 `models/hyperlpr3/`

### 运行检测

#### GUI 运行方式

```bash
# 方式一：双击启动（Windows 推荐）
应急车道违章检测.pyw

# 方式二：命令行启动 GUI
python emergency_lane/traffic_violation_gui.py
```

#### CLI 运行方式

```bash
python emergency_lane/run_gpu_v8.py "视频文件路径.mp4"
```

### Web 界面（内网多机访问）

#### 直接启动 Web 服务

```bash
python web_server.py --host 0.0.0.0 --port 8080
```

#### 使用启动器管理 Web 服务

```bash
# 启动
python start_web.pyw

# 停止
python start_web.pyw --stop

# 重启
python start_web.pyw --restart

# 查看状态
python start_web.pyw --status
```

也可以双击运行：
- `start_web.pyw`

启动后，局域网内其他电脑可通过浏览器访问：
- 本机访问：`http://localhost:8080`
- 局域网访问：`http://<本机IP>:8080`

**功能特性：**
- 📹 拖拽或点击上传视频
- ⚙️ 在线调整检测参数
- 📊 实时检测进度显示
- 📦 一键打包下载所有结果
- 📋 历史任务管理

### 输出结果

检测结果保存在视频同目录下的 `{视频名}_检测结果/` 文件夹，当前证据结构为“一个车牌一组证据”：

```
视频名_检测结果/
├── 视频名.mp4                              # 原视频副本
├── 视频名_violation_report.html            # HTML 报告
├── manifest.json                           # 证据索引（供 Web / 历史任务使用）
└── evidence/
    ├── 苏U7WV26/
    │   ├── 苏U7WV26.jpg                    # 该车牌最佳截图
    │   ├── 苏U7WV26.mp4                    # 该车牌 15 秒证据视频
    │   └── meta.json                       # 时间 / 置信度 / 颜色等元数据
    ├── 苏EP079A/
    │   ├── 苏EP079A.jpg
    │   ├── 苏EP079A.mp4
    │   └── meta.json
    └── ...
```

规则说明：
- 同一车牌在同一视频里只保留 **最佳一组证据**
- 未识别车牌默认 **不保留**
- 截图与视频文件名与车牌目录名保持一致，便于人工核查和上传到其他平台

## ⚙️ 参数配置

### GUI 参数（双击 `应急车道违章检测.pyw` 打开）

| GUI 控件 | 对应变量 | 默认值 | 范围 | 说明 |
|----------|----------|--------|------|------|
| 车道X位置 | `lane_x` | 0.84 | 0.0 ~ 1.0 | 应急车道左边界占画面宽度的比例。0.84 表示从画面 84% 处到右边缘为应急车道 |
| 车道宽度 | `lane_width` | 0.16 | 0.01 ~ 0.5 | 应急车道占画面宽度的比例。与车道X位置配合，0.84+0.16=1.0 即到画面右边缘 |
| 车道顶部Y | `lane_top` | 0.15 | 0.0 ~ 1.0 | 忽略画面上方的比例。0.15 表示忽略最上方 15%（远处目标太小容易误检） |
| GPU加速 | `use_gpu` | 开启 | - | 开启后进入严格 GPU 模式，任一步无法使用 GPU 都会直接报错，不会回退到 CPU |
| 置信度阈值 | `conf_threshold` | 0.5 | 0.1 ~ 1.0 | YOLOv12 车辆检测置信度，越高越严格（减少误检但可能漏检），越低越敏感 |
| 检测缩放 | `detection_scale` | 0.75 | 0.25 ~ 1.0 | 检测前将视频帧缩小的比例。22G 显存 GPU 建议优先使用 0.75；追求极致精度可用 1.0 |

### 脚本内部参数（`run_gpu_v8.py` 中的常量）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `LANE_START_X` | 0.84 | 应急车道左边界位置（画面宽度的比例） |
| `MIN_OVERLAP` | 0.25 | 车辆与应急车道最小重叠比，超过此值才算在应急车道内 |
| `CONF_THRESH` | 0.25 | YOLOv12 车辆检测置信度阈值 |
| `DET_SCALE` | 0.75 | 检测帧降采样比例 |
| `CONFIRM_FRAMES` | 3 | 需连续出现在应急车道 N 帧才确认违章 |
| `COOLDOWN_FRAMES` | 150 | 同一车辆重复记录的冷却帧数（150帧 ≈ 5秒@30fps） |
| `CLIP_DURATION` | 15 | 每段举报视频目标时长（秒） |
| `PRE_VIOLATION_PAD` | 3 | 违章前保留时间（秒） |
| `POST_VIOLATION_PAD` | 5 | 违章后保留时间（秒） |
| `MIN_GAP_TO_CUT` | 10 | 无违章超过此时长可裁切（秒） |

## 🔧 技术架构

```
输入视频 → YOLOv12s 检测(OpenVINO GPU) → 多目标跟踪 → 应急车道区域判定
                                                         ↓
                                              确认违章 → FFmpeg 15s剪辑
                                                         ↓
                                        多帧扫描(±30帧) → HyperLPR3(OpenVINO GPU)
                                                         ↓
                                              HSV颜色识别 → HTML报告
```

### 性能参考

| 环节 | CPU | Intel Arc 140V GPU | 加速比 |
|------|-----|---------------------|--------|
| YOLOv12s 推理 | 86.8ms/帧 | 4.8ms/帧 | 17.9x |
| 车牌识别(pipeline) | 435ms/次 | 27ms/次 | 16.1x |
| 3分钟4K视频全流程 | ~25min | ~8min | 3x |

> 测试环境：Intel Core Ultra 7 258V, 32GB RAM, Intel Arc 140V 16GB iGPU

## 📸 使用场景

本系统针对以下行车记录仪场景设计：

- 三车道高速公路
- 拍摄车辆在最右侧行车道
- 应急车道在拍摄车右侧
- 4K 宽屏视频（3840x1696）

如需适配其他场景，调整 `LANE_START_X` 等区域参数即可。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 开源许可

[MIT License](LICENSE)
