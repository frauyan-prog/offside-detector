# 越位识别系统 — 项目介绍

> 自动足球越位检测系统，基于本地 YOLOv8 模型，无需外部 API。

---

## 一、项目概述

本系统对足球比赛视频进行自动越位检测，输出带越位线标注的视频和分析数据（JSON）。

**核心功能：**
- 球场标定（手动点击 / AI 自动检测）
- 球员检测与跟踪（YOLOv8 + ByteTrack）
- 队服颜色分类（K-means HSV 聚类）
- 越位判定（IFAB Law 11）
- 可视化输出（越位线、球员框、小地图）

**技术栈：** Python 3.13 / OpenCV / NumPy / Ultralytics YOLOv8 / supervision / ByteTrack

---

## 二、运行入口与参数

### 入口文件

```
run.py
```

### 基本用法

```bash
# 交互式标定（手动点击4个点）
python run.py --video 输入视频.mp4

# 使用预保存的标定文件（跳过手动标定）
python run.py --video 输入视频.mp4 --calibration calib.json

# AI 自动标定（推荐，需要提前下载Pitch Detection模型到 offside_detector/models/ 目录）
python run.py --video 输入视频.mp4 --calib-mode auto

# 只处理前200帧，快速验证
python run.py --video 输入视频.mp4 --max-frames 200

# 缩放帧宽加快处理
python run.py --video 输入视频.mp4 --resize 960
```

### 完整参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--video` / `-v` | **输入视频路径（必填）** | — |
| `--calibration` / `-c` | 预保存的标定 JSON 文件路径（跳过交互标定） | `None` |
| `--calib-mode` | 标定模式：`auto`/`center_circle`/`penalty_area`/`four_corner`/`generic` | 无标定文件时为 `interactive` |
| `--attack-dir` | 进攻方向：`top_to_bottom`/`bottom_to_top`/`left_to_right`/`right_to_left` | 自动检测 |
| `--output` / `-o` | 输出目录 | `./output` |
| `--max-frames` / `-m` | 最多处理帧数 | 全部 |
| `--skip` / `-s` | 跳过开头 N 帧 | 0 |
| `--every` / `-e` | 每 N 帧处理一次 | 1 |
| `--resize` / `-r` | 缩放帧宽度（加快处理） | 不缩放 |
| `--yolo-model` | YOLO 模型大小：`football`(专用)/`n`/`s`/`m` | `football` |
| `--confidence` | 球员检测置信度阈值 | 0.35 |
| `--no-minimap` | 隐藏小地图叠加层 | 显示 |
| `--only-offside` | 只输出越位帧 | 全部输出 |
| `--mode` | 处理模式：`batch`(批处理)/`stream`(Web服务)/`interactive`(交互窗口) | `batch` |
| `--port` | Web 服务模式下的端口号 | 8080 |

---

## 三、使用的模型

### 1. 球员检测模型（必需）

| 模型 | 文件 | 大小 | 说明 |
|------|------|------|------|
| **Football YOLOv8**（推荐） | `football-player-detection.pt` | ~21MB | 直接检测4类：球、门将、球员、裁判 |
| YOLOv8n（备用） | `yolov8n.pt` | ~6MB | 只检测"人"，需后处理分类 |
| YOLOv8s（备用） | `yolov8s.pt` | ~22MB | 同上 |
| YOLOv8m（备用） | `yolov8m.pt` | ~52MB | 同上 |

- **首次运行**会自动下载模型到 `offside_detector/models/` 目录
- **Football 模型**首次使用会从 Ultralytics Hub 下载，需联网

### 2. 球场关键点检测模型（auto 模式可选）

| 模型 | 文件 | 大小 | 说明 |
|------|------|------|------|
| YOLOv8x-pose Football Pitch | `yolo-football-pitch-detection.pt` | ~134MB | 检测32个球场关键点 |

- 需手动从 HuggingFace 下载（[martinjolif/yolo-football-pitch-detection](https://huggingface.co/martinjolif/yolo-football-pitch-detection)），放入 `offside_detector/models/`
- 如在境外HuggingFace无法访问，需使用VPN下载

### 模型输入输出

```
输入：视频帧（图像像素 BGR 格式）
  ↓
[YOLOv8 球员检测] → 输出：每个球员的边界框 [x1,y1,x2,y2] + 类别 + 置信度
  ↓
[ByteTrack 跟踪] → 输出：每个球员分配稳定 track_id
  ↓
[世界坐标转换] → 输出：每个球员脚部位置的世界坐标 (world_x, world_y) 单位：米
  ↓
[越位分析] → 输出：越位线位置 + 越位球员列表 + 是否越位
  ↓
[可视化] → 输出：带标注的视频帧（BGR 图像）
```

---

## 四、项目文件结构

```
D:\workBuddy\越位识别\
├── run.py                          # CLI 入口
├── requirements.txt                # Python 依赖
├── project_plan.md                 # 项目计划与进度
│
├── offside_detector/               # 核心检测模块
│   ├── config.py                  # 球场配置（FIFA标准尺寸、32关键点坐标、可视化颜色）
│   ├── field_detector.py          # 球场标定模块（手动/自动标定，计算单应性矩阵）
│   ├── player_detector.py         # 球员检测与跟踪（YOLOv8 + ByteTrack + 队服颜色分类）
│   ├── pitch_keypoint_detector.py # 球场关键点检测（YOLOv8-pose，auto模式使用）
│   ├── view_transformer.py        # 坐标转换（像素↔世界坐标，基于单应性矩阵）
│   ├── offside_analyzer.py        # 越位判定逻辑（IFAB Law 11）
│   ├── visualizer.py              # 可视化绘制（越位线、球员框、小地图）
│   ├── video_processor.py         # 端到端视频处理流水线
│   └── models/                   # 模型文件存放目录
│       ├── yolov8n.pt
│       ├── football-player-detection.pt
│       └── yolo-football-pitch-detection.pt（需手动下载）
│
├── web/                           # Web 界面（可选）
│   ├── app.py                     # Flask 后端 API
│   └── templates/index.html       # 前端页面
│
├── output/                        # 输出目录
│   ├── *_offside.mp4             # 处理后的视频
│   ├── *_analysis.json            # 逐帧分析数据
│   └── debug_log.txt             # 调试日志
│
└── .workbuddy/                   # WorkBuddy 工作区记忆
    └── memory/                   # 项目记忆文件
```

---

## 五、各模块重要函数说明

### 1. `run.py` — CLI 入口

| 函数 | 说明 |
|------|------|
| `main()` | 解析命令行参数，构建 `ProcessingConfig`，启动对应处理模式 |
| `run_batch()` | 批处理模式：处理整个视频，保存输出 |
| `run_stream()` | 流媒体模式：启动 Flask Web 服务 |
| `run_interactive()` | 交互模式：逐帧显示结果，键盘控制 |

---

### 2. `offside_detector/config.py` — 球场配置

| 类/函数 | 说明 |
|---------|------|
| `SoccerFieldConfiguration` | FIFA 标准球场尺寸（宽68m × 长105m） |
| `.WIDTH` / `.LENGTH` | 球场宽（x轴）和长（y轴） |
| `.get_world_coordinates()` | 返回32个关键点的世界坐标（米） |
| `.KEYPOINT_LABELS` | 32个关键点的名称列表 |
| `YOLO_CONFIG` | YOLO 模型变体配置 |
| `VISUALIZATION` | 可视化颜色配置（越位线红色、球员框绿色等） |

**重要：** 世界坐标系 — `x ∈ [0, 68]`（宽度），`y ∈ [0, 105]`（长度）

---

### 3. `offside_detector/field_detector.py` — 球场标定

**核心类：**
- `FieldDetector` — 自动标定（AI检测关键点）
- `ManualFieldCalibrator` — 手动标定基类
- `CenterCircleCalibrator` — 中圈标定（2点）
- `PenaltyAreaCalibrator` — 禁区标定（4点）
- `FourCornerCalibrator` — 四角标定（4点）

| 重要函数 | 说明 |
|----------|------|
| `_build_from_2_points(pixel_pts, attack_dir)` | 从2个像素点计算中圈相似变换矩阵（**当前使用相似变换，不用虚拟点**） |
| `save_calibration(result, json_path)` | 保存标定结果到 JSON |
| `load_calibration(json_path)` | 从 JSON 加载标定结果 |
| `FieldDetectionResult` | 标定结果数据类（含 `homography`、`inverse_homography`、`attack_dir` 等字段） |

**标定模式说明：**
| 模式 | 需要标定点数 | 适用场景 |
|------|--------------|----------|
| `center_circle` | 2点（中圈直径两端） | 摄像头对准中圈，最简单 |
| `penalty_area` | 4点（禁区角点） | 摄像头对准球门附近 |
| `four_corner` | 4点（球场四角） | 全景摄像头 |
| `generic` | 4点（任意矩形） | 通用 |
| `auto` | 0点（AI自动） | 需下载134MB模型，完全自动 |

---

### 4. `offside_detector/player_detector.py` — 球员检测

**核心类：**
- `PlayerDetector` — 球员检测器
- `PlayerDetection` — 单帧单个球员检测结果数据类
- `FrameDetection` — 单帧所有检测结果数据类

| 重要函数 | 说明 |
|----------|------|
| `detect_and_track(frame, frame_idx)` | 对一帧进行球员检测 + ByteTrack 跟踪，返回 `FrameDetection` |
| `_classify_teams(frame, detections)` | 通过队服颜色（K-means HSV聚类）区分攻守双方 |
| `_filter_out_referees(detections)` | 过滤裁判（基于位置，站在两队之间的通常是裁判） |
| `_filter_out_field_staff(detections)` | 过滤场边人员（边界框超出球场区域） |

**检测类别（Football YOLOv8）：**
- `ball`（球）— 类别ID: 0
- `goalkeeper`（门将）— 类别ID: 1
- `player`（球员）— 类别ID: 2
- `referee`（裁判）— 类别ID: 3

---

### 5. `offside_detector/view_transformer.py` — 坐标转换

**核心类：**
- `ViewTransformer` — 坐标变换器

| 重要函数 | 说明 |
|----------|------|
| `set_homography(H, field_width, field_length)` | 设置单应性矩阵 H 及其逆矩阵 |
| `pixel_to_world(x, y)` | 像素坐标 → 世界坐标（米），返回 `[wx, wy]` |
| `world_to_pixel(wx, wy)` | 世界坐标 → 像素坐标，返回 `[px, py]` |
| `is_in_field(wx, wy)` | 判断世界坐标是否在球场范围内 |

**单应性矩阵 H：**
```
[wx]     [H[0,0]  H[0,1]  H[0,2]]   [px]
[wy]  =  [H[1,0]  H[1,1]  H[1,2]] * [py]
[ 1]     [H[2,0]  H[2,1]  H[2,2]]   [ 1]
```
（实际计算使用 `cv2.perspectiveTransform`）

---

### 6. `offside_detector/offside_analyzer.py` — 越位判定

**核心类：**
- `OffsideAnalyzer` — 越位分析器
- `OffsideResult` — 单帧越位分析结果数据类
- `AttackDirection` — 进攻方向枚举

| 重要函数 | 说明 |
|----------|------|
| `analyze(frame_idx, frame_det, ball_world_pos)` | 分析一帧的越位情况，返回 `OffsideResult` |
| `_determine_attack_direction(players)` | 根据门将世界坐标位置判断进攻方向 |
| `_separate_teams(players, attack_dir)` | 将球员分为进攻方和防守方 |
| `_find_second_last_defender(defenders, attack_dir)` | 找倒数第二名防守球员（**越位线参考**） |
| `_calculate_offside_pixel_line(offside_pos, attack_dir)` | 计算越位线的像素坐标（用于绘制） |
| `_debug_log(message)` | 写入调试日志到 `output/debug_log.txt` |

**进攻方向枚举：**
```python
class AttackDirection(Enum):
    TOP_TO_BOTTOM  = "top_to_bottom"   # 向 y=105 球门进攻（视频中球场竖向）
    BOTTOM_TO_TOP  = "bottom_to_top"   # 向 y=0 球门进攻（视频中球场竖向）
    LEFT_TO_RIGHT  = "left_to_right"    # 向 x=68 球门进攻（视频中球场横向）
    RIGHT_TO_LEFT  = "right_to_left"    # 向 x=0 球门进攻（视频中球场横向）
```

**越位判定逻辑（IFAB Law 11）：**
1. 找出**倒数第二名防守球员**（不含门将）—— 这就是越位线
2. 比较每个进攻球员：**是否比越位线更靠近对方球门？**
3. 同时比较**球的位置**：进攻球员必须比球更靠近对方球门
4. 满足以上条件 → 越位位置 → 再判断是否"参与比赛"（目前简化：只要在越位位置就标记）

---

### 7. `offside_detector/visualizer.py` — 可视化

**核心类：**
- `OffsideVisualizer` — 可视化绘制器

| 重要函数 | 说明 |
|----------|------|
| `draw(frame, offside_result, frame_det)` | 在主帧上绘制所有标注，返回标注帧 |
| `_draw_offside_line(frame, result)` | 绘制越位线（横跨整个球场宽度/长度） |
| `_draw_players(frame, frame_det, result)` | 绘制所有球员框（颜色区分：红=越位、绿=非越位、蓝=防守、橙=门将、黄=裁判） |
| `_draw_minimap(frame, result)` | 绘制小地图（俯视视角，显示球员位置） |
| `_draw_frame_info(frame, result)` | 绘制帧号、越位状态等文字信息 |

---

### 8. `offside_detector/video_processor.py` — 视频处理流水线

**核心类：**
- `VideoProcessor` — 视频处理器
- `ProcessingConfig` — 处理配置数据类

| 重要函数 | 说明 |
|----------|------|
| `process(video_path, config, ...)` | **主入口**：处理整个视频，返回处理统计 |
| `process_stream(video_path, ...)` | 流式处理：逐帧 yield 处理结果（用于 Web 模式） |
| `_setup_calibration(...)` | 设置标定（加载预标定文件或启动交互标定） |
| `_process_frame(frame, frame_idx, ...)` | 处理单帧：检测 → 坐标转换 → 越位分析 → 可视化 |
| `_init_models(config)` | 初始化 YOLO 模型、ByteTrack 跟踪器 |

**流水线步骤（`_process_frame`）：**
```
输入：原始视频帧
  ↓
[1] 球员检测] PlayerDetector.detect_and_track(frame, frame_idx)
  ↓ 输出：FrameDetection（含所有球员、门将、裁判的像素坐标和track_id）
[2] 坐标转换] ViewTransformer.pixel_to_world(foot_x, foot_y) 对每个球员
  ↓ 输出：每个球员的 world_x, world_y（米）
[3] 越位分析] OffsideAnalyzer.analyze(frame_idx, frame_det, ball_pos)
  ↓ 输出：OffsideResult（越位线位置、越位球员列表、是否越位）
[4] 可视化] OffsideVisualizer.draw(frame, result, frame_det)
  ↓ 输出：带标注的BGR图像帧
[5] 写入输出视频 + JSON数据
```

---

## 六、整体运作流程（串联方式）

```
                        ┌─────────────────────────────────────────────┐
                        │               run.py (CLI入口)                │
                        │  解析参数 → 构建ProcessingConfig           │
                        └──────────────────┬──────────────────────────┘
                                           │
                                           ▼
                        ┌─────────────────────────────────────────────┐
                        │        VideoProcessor.process()              │
                        │  (video_processor.py)                      │
                        └──────────────────┬──────────────────────────┘
                                           │
                                    ┌──────┴──────┐
                                    │  初始化阶段  │
                                    └──────┬──────┘
                                           │
                      ┌────────────────────┼────────────────────────┐
                      ▼                    ▼                        ▼
            [标定球场]             [初始化模型]              [打开视频]
        field_detector.py      player_detector.py            cv2.VideoCapture
        ↓                     ↓                             ↓
   计算单应性矩阵H       加载YOLOv8模型           读取视频元数据
   保存inverse_H          初始化ByteTrack             (宽、高、帧率)
                      └────────────────────┬────────────────────────┘
                                           │
                                    ┌──────┴──────┐
                                    │  处理循环    │ ← 对每一帧执行
                                    └──────┬──────┘
                                           │
            ┌──────────────────────────────┼──────────────────────────────┐
            │                              │                              │
            ▼                              ▼                              ▼
   [1] 球员检测                [2] 坐标转换                   [3] 越位分析
   PlayerDetector               ViewTransformer              OffsideAnalyzer
   .detect_and_track()         .pixel_to_world()           .analyze()
            │                              │                              │
            ▼                              ▼                              ▼
   FrameDetection             每个球员获得world_x,           OffsideResult
   (像素坐标 + track_id)      world_y (米)                 (越位线 + 越位球员)
            │                              │                              │
            └──────────────────────────────┼──────────────────────────────┘
                                           │
                                           ▼
                                    [4] 可视化绘制
                                    OffsideVisualizer
                                    .draw()
                                           │
                                           ▼
                                    [5] 输出
                                    ├── 写入视频文件 (.mp4)
                                    └── 写入JSON数据 (.json)
```

---

## 七、坐标系统重要说明

**世界坐标系（FIFA标准）：**
```
    y=0 (近门线)                     y=105 (远门线)
         ┌─────────────────────────┐
         │                         │
    x=0 │        球场             │ x=68
    (左) │      105m × 68m        │ (右)
         │                         │
         └─────────────────────────┘

    中圈中心: (x=34, y=52.5)
    左边球门: y=0      右边球门: y=105  (竖向球场)
    左边球门: x=0      右边球门: x=68   (横向球场)
```

**进攻方向与排序轴对应关系：**
| 进攻方向 | 球门在世界坐标 | 排序轴 | 排序方式 |
|----------|----------------|--------|----------|
| `top_to_bottom` | y=105 | world_y | 降序（大=靠近球门） |
| `bottom_to_top` | y=0 | world_y | 升序（小=靠近球门） |
| `left_to_right` | x=68 | world_x | 降序（大=靠近球门） |
| `right_to_left` | x=0 | world_x | 升序（小=靠近球门） |

---

## 八、标定模式详解

### `center_circle` 模式（2点，最简单）

用户在中圈直径两端各点一个点 → 系统：
1. 根据 `attack_dir` 判断球场方向
2. 使用**相似变换**（2点 → 精确4参数），不是透视变换
3. 计算单应性矩阵 H

**注意：** 只用2个点标定，远离中圈的区域坐标精度会下降。

### `auto` 模式（0点，全自动）

系统用 YOLOv8-pose 模型自动检测32个球场关键点 → 用最小二乘法计算 H。

**需要：** `offside_detector/models/yolo-football-pitch-detection.pt`（~134MB，需手动下载）

### 手动标定（交互模式）

运行时不加 `--calibration` 参数 → 弹出窗口 → 在首帧上按顺序点击标定点。

---

## 九、输出文件说明

### 1. 视频文件 `*_offside.mp4`

带标注的视频，包含：
- 越位线（红色实线）
- 球员边界框（颜色区分：红=越位、绿=非越位、蓝=防守、橙=门将）
- 小地图（右上角，俯视视角）
- 帧号和越位状态文字

### 2. JSON 文件 `*_analysis.json`

逐帧分析数据，结构：
```json
{
  "video_info": {"width": 1280, "height": 720, "fps": 30, "total_frames": 736},
  "calibration": {"mode": "center_circle", "attack_dir": "left_to_right"},
  "frames": [
    {
      "frame_idx": 100,
      "is_offside_situation": true,
      "offside_line_y": 50.0,
      "offside_players": [
        {"track_id": 7, "world_x": 45.2, "world_y": 78.3, "team": "attacking"}
      ],
      "all_players": [
        {"track_id": 1, "world_x": 60.1, "world_y": 5.2, "team": "defending", "class_name": "goalkeeper"}
      ]
    }
  ]
}
```

---

## 十、已知问题与工作日志

### 当前已知问题

1. **center_circle 标定精度有限** — 只用2个点，远离中圈的区域世界坐标会失真
   - **缓解方法：** 使用 `auto` 模式（需下载134MB模型）
   - **临时缓解：** 代码已实现像素位置回退机制（当世界坐标不可靠时自动切换）

2. **越位线轻微倾斜** — 手动标点不完美时，相似变换会有旋转误差（通常 < 10°）
   - **影响：** 视觉效果，不影响越位判定逻辑（判定用世界坐标，不是像素线）

3. **队服颜色分类偶尔错误** — K-means 聚类在球员重叠时可能分类错误
   - **影响：** 攻守双方可能颠倒 → 越位判定错误

4. **处理速度** — CPU 模式下约 8-12 fps（取决于视频分辨率和模型大小）
   - **加速方法：** `--resize 960` 或 `--every 2`（每2帧处理一次）

### 工作日志

详见 `.workbuddy/memory/2026-06-17.md` 和 `2026-06-19.md`

**最近修改（2026-06-19）：**
- 修复 `center_circle` 标定时水平球场垂直直径映射到错误的世界坐标轴的bug
- 添加 `attack_dir` 到标定 JSON 的保存/加载
- 实现像素位置回退机制：当世界坐标不可靠时，用像素位置排序防守球员

---

## 十一、快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备视频
# 将视频文件放到项目目录下，例如：13-有意处理球.mp4

# 3. 运行（交互式标定）
python run.py --video 13-有意处理球.mp4

# 4. 在弹出的窗口中按顺序点击标定点，然后按任意键确认

# 5. 查看输出
# output/13-有意处理球_offside.mp4  ← 带标注的视频
# output/13-有意处理球_analysis.json  ← 分析数据
# output/calibration.json  ← 标定数据（下次可直接用 --calibration 加载）
```

---

## 十二、项目状态

| 阶段 | 状态 |
|------|------|
| S1 球场标定验证 | ✅ 通过 |
| S2 Homography精度 | 🟡 基本可用（center_circle模式精度有限） |
| S3 球员检测 | ✅ 完成 |
| S4 越位判定 | ✅ 完成 |
| S5 可视化 | ✅ 基本可用 |
| S6 集成测试 | 🟡 能跑（8-12fps CPU） |

---

*最后更新：2026-06-19*
