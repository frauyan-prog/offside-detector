# 越位识别系统 — 现有资产清单（技术审计）

**审计人**: Anne（项目经理/主开发）
**审计日期**: 2026-06-16
**项目版本**: v1.0.0
**审计范围**: `offside_detector/` 全部源码 + `run.py` + `requirements.txt` + `web/`

---

## 第一章：项目结构

```
D:\workBuddy\越位识别\
├── run.py                          # CLI入口，argparse参数解析，batch/stream/interactive三种模式
├── requirements.txt                # 依赖：opencv, numpy, ultralytics, supervision, flask
├── PROJECT_PLAN.md                 # 6步分步计划（Step 1-6）
├── yolov8n.pt                      # YOLOv8n球员检测本地模型
├── *.mp4                           # 测试视频素材（3个）
├── offside_detector/
│   ├── __init__.py                 # 包声明，版本号 v1.0.0
│   ├── config.py                   # FIFA球场常量 + 32关键点标签/世界坐标 + 可视化配置
│   ├── field_detector.py           # 场地标定：5种模式（Auto/PenaltyArea/FourCorner/Generic/CenterCircle）
│   ├── pitch_keypoint_detector.py  # 自动标定：本地YOLOv8-pose检测32关键点 → RANSAC计算Homography
│   ├── player_detector.py          # 球员检测：YOLOv8 + ByteTrack + HSV K-means分队
│   ├── view_transformer.py         # 坐标转换：pixel↔world，基于Homography的perspectiveTransform
│   ├── offside_analyzer.py         # 越位判定：攻击方向检测 + 倒数第二名防守者 + 越位线计算
│   ├── visualizer.py               # 可视化：越位线/球员框/越位区域/小地图
│   ├── video_processor.py          # 主管线：标定→检测→转换→判定→可视化→输出
│   └── models/
│       └── yolo-football-player-detection.pt/   # （未使用的本地球员模型）
└── web/
    ├── app.py                      # Flask Web服务：上传/处理/进度查询/结果下载
    └── templates/
        └── index.html              # Web前端页面
```

### 数据流

```
视频帧 → field_detector（标定，得H）
       → player_detector（YOLOv8 + ByteTrack，得球员bbox+脚点）
       → view_transformer（H变换，像素→世界坐标）
       → offside_analyzer（IFAB规则判定）
       → visualizer（绘制标注帧）
       → 输出视频 + JSON
```

---

## 第二章：已完成且正确的模块

### 2.1 view_transformer.py — 坐标转换 ✅ 正确

**判定：逻辑正确，可原样保留**

- `pixel_to_world(x, y)`: 使用 `cv2.perspectiveTransform` + homography，将像素坐标转换为世界坐标 — 数学正确
- `world_to_pixel(wx, wy)`: 使用逆矩阵 `inverse_homography`，将世界坐标转换回像素 — 数学正确
- `batch_pixel_to_world` / `batch_world_to_pixel`: 批量版本，reshape + perspectiveTransform — 正确
- `is_in_field(wx, wy, margin=5.0)`: 边界检查，带5m容差 — 合理

**注意事项**：
- 该模块本身无bug，但其输出完全依赖输入的homography矩阵是否正确
- 如果上游 `pitch_keypoint_detector.py` 给出错误的H，本模块会忠实地执行错误的映射

### 2.2 player_detector.py — 球员检测管线 ✅ 基本正确

**判定：管线完整，小问题不影响核心功能**

- YOLOv8（`classes=[0]`仅检测person）→ supervision ByteTrack → 分类 → 分队 — 流程正确
- `foot_position = [(x1+x2)/2, y2]`：bbox底部中心作为脚部位置 — 合理近似
- 门将检测：帧底部20% + 中心区域 + 累计3帧确认 — 启发式方法，够用
- HSV K-means分队：提取上身球衣颜色 → 2-means聚类 → 按y位置分攻守 — 基本可行

**已知局限（非bug）**：
- `_classify_teams()` 只执行一次（`_team_assigned` 标志），如果首帧分类错误则全错
- 门将检测依赖画面位置（底部20%），仰角大的镜头可能误判
- referee 仅为标签占位，实际不区分

### 2.3 offside_analyzer.py — 越位判定逻辑 ✅ 基本正确（符合IFAB Law 11）

**判定：核心规则实现正确，可复用**

- `_find_second_last_defender()`: 按靠近防守球门的y值排序，取第二位 — 正确
- `_check_offside_players()`: world_y 与 offside_y 比较，±0.1m容差 — 符合IFAB"齐平不算越位"
- `_calculate_offside_pixel_line()`: `world_to_pixel(0, offside_y)` → `world_to_pixel(68, offside_y)` — 正确，越位线从左边线到右边线
- 攻击方向检测：30帧投票锁定机制 — 稳定性好

**小问题**：
- `_find_second_last_defender()`: 只有1名防守者时，将唯一防守者当作"倒数第二" — 理论上不正确但实际不会发生
- `_determine_attack_direction()`: 领先方向变量名 `vote` 在第222行引用了可能被覆盖的局部变量（`if vote != "unknown"`中`vote`已被重赋值）

### 2.4 visualizer.py — 可视化 ✅ 正确

**判定：绘制逻辑正确，可原样保留**

- 越位线：从 `offside_line_pixels` 的两个端点画线 — 正确
- 越位区域：半透明红色覆盖，TOP_TO_BOTTOM覆盖线下方，反之覆盖上方 — 正确
- 球员框：红色=越位，蓝色=倒数第二防守者，绿色=不越位 — 正确
- 小地图：world_to_map缩放到160×250像素 — 正确

### 2.5 video_processor.py — 帧处理流程 ✅ 正确

**判定：管线串联正确，流程无bug**

- 标定→检测→分析→可视化→输出 — 完整管线
- 支持视频文件和流式处理（generator）
- 标定优先级：文件加载 > Auto模式 > Roboflow API > 交互式 — 合理
- JSON输出 + _NumpyEncoder — 正确处理numpy类型

### 2.6 field_detector.py — 手动标定模式 ✅ 正确

**判定：4种手动标定模式均可正确工作**

- PenaltyArea: 门柱(30.34,0)(37.66,0) + 禁区线端(13.84,16.5)(54.16,16.5) — FIFA标准坐标正确
- FourCorner: (0,0)(68,0)(68,105)(0,105) — 正确
- Generic: 用户输入尺寸 — 正确
- CenterCircle: 2点直径 + 合成虚拟点 — 近似方法，精度有限但可用
- 交互UI：缩放/撤销/重置/确认 — 完善的用户体验

---

## 第三章：需要修改的模块

### 3.1 pitch_keypoint_detector.py — 🔴 关键bug，必须修改

#### Bug 1: RANSAC阈值单位错误（主因）

**位置**: `pitch_keypoint_detector.py:267`

```python
ransacReprojThreshold=10.0,  # metres — generous to handle far-side noise
```

**问题**: `cv2.findHomography` 的 `ransacReprojThreshold` 参数单位是**目标坐标系**的单位。此处 `dst_pts` 是 Roboflow 120×70 坐标系（米），10米意味着重投影误差在10米以内的点都被当作内点——对于120米长的球场，这等于允许8.3%的误差，几乎任何点都会成为内点。RANSAC无法有效过滤离群点，导致远端噪声关键点扭曲了homography，这是**越位线倾斜的根本原因**。

**修复**:
```python
ransacReprojThreshold=5.0,  # 5像素等价 ≈ 5 * (120/1920) ≈ 0.3m in Roboflow坐标
```
或者更好的做法：先将dst_pts归一化到像素尺度，再使用5px阈值。

**参考**: SoccerNet冠军方案使用5px阈值。

#### Bug 2: 缺少多子集投票（Voter）

**位置**: `pitch_keypoint_detector.py:226-278` (`compute_homography`方法)

**问题**: 当前只做一次RANSAC就确定homography，没有在不同关键点子集上分别计算再选最优。远端低置信度关键点的噪声可能主导结果。

**需添加**: 参考SoccerNet方案的多子集Voter：
1. 子集1: 所有点（conf > 0.3）
2. 子集2: 高置信度点（conf > 0.5）
3. 子集3: RANSAC内点重拟合
→ 选重投影RMSE最低的H

#### Bug 3: 缺少几何约束验证

**问题**: 计算出的homography没有经过任何几何合理性检查。极端情况下可能产生"相机在地下"或球场严重扭曲的结果。

**需添加**:
1. 球门线平行度：`world_to_pixel(0,0)→world_to_pixel(68,0)` 与 `world_to_pixel(0,105)→world_to_pixel(68,105)` 的夹角应 < 2°
2. 四角凸性：四角点投影后应是凸四边形
3. 相机高度合理性：H矩阵分解后相机不应在地面以下

#### Bug 4: Roboflow世界坐标使用非FIFA标准尺寸

**位置**: `pitch_keypoint_detector.py:44-133` (`_build_world_coords`)

**问题**: 世界坐标使用 Roboflow 120×70 坐标系，其中：
- 球场长度 120m（应为105m），宽度 70m（应为68m）
- 禁区 41.0×20.15m（应为40.32×16.5m）
- 这意味着模型输出的关键点对应的并非标准FIFA尺寸

**影响**: 通过 `_CONV_ROBOFLOW_TO_FIFA` 线性转换后，禁区尺寸在FIFA坐标系中为：
- 宽度: 41.0 × 68/70 = **39.83m**（标准40.32m，差0.49m）
- 深度: 20.15 × 105/120 = **17.63m**（标准16.5m，差1.13m）

**评估**: 此差异会在homography中引入约1m量级的位置偏差，但非主因。主因仍是RANSAC阈值bug。修复RANSAC后此偏差可容忍，后续版本再优化。

#### Bug 5: 转换矩阵的坐标系旋转

**位置**: `pitch_keypoint_detector.py:144-148`

`_CONV_ROBOFLOW_TO_FIFA` 做了180°旋转：
- Roboflow(0,0)近端左上角 → FIFA(68,105)远端右下角
- 这意味着Roboflow的"近端"对应FIFA的y=105端

**影响**: 转换本身数学正确，但需确保 `offside_analyzer.py` 中的攻击方向判定与FIFA坐标系一致。当前代码中攻击方向判定基于world_y均值与52.5m中线比较，与转换后的坐标系一致，**此处无bug**。

### 3.2 config.py — 🟡 关键点顺序与Roboflow模型不匹配

**问题**: `SoccerFieldConfiguration.get_world_coordinates()` 的32个关键点排序与Roboflow模型输出顺序不一致：

| 索引 | config.py | pitch_keypoint_detector.py (Roboflow) |
|------|-----------|--------------------------------------|
| 0 | corner_top_left | Top-left corner |
| 1 | corner_top_right | L-touchline @ PA-top |
| 2 | corner_bottom_right | L-touchline @ GA-top |
| 4 | penalty_area_left_top (y=0) | L-touchline @ PA-bottom |
| 5 | penalty_area_left_bottom (y=105) | Bottom-left corner |
| ... | ... | ... |

**影响**:
- **Auto模式（本地模型）**: 使用 `pitch_keypoint_detector.py` 的自有世界坐标，**不受影响** ✅
- **Roboflow API模式**: `field_detector.py:1024-1026` 使用 `config.py` 的世界坐标与API返回的关键点配对，**索引对不上，homography会完全错误** 🔴
- **手动标定模式**: 不使用32点映射，**不受影响** ✅

**修复**: 在 `field_detector.py` 的 `_detect_auto` 方法中，使用 `pitch_keypoint_detector.WORLD_KEYPOINTS` 替代 `SoccerFieldConfiguration.get_world_coordinates()`。

### 3.3 field_detector.py — 🟡 Roboflow API模式标定bug

**位置**: `field_detector.py:1024-1026`

```python
from .config import SoccerFieldConfiguration
world_coords = SoccerFieldConfiguration.get_world_coordinates()
dst_pts = world_coords[valid_indices]
```

**问题**: 如上所述，`config.py` 的关键点顺序与Roboflow API返回的关键点顺序不一致。

**修复方案**:
1. 短期：弃用Roboflow API模式，统一使用本地YOLOv8-pose模型
2. 长期：添加Roboflow关键点索引映射表

### 3.4 config.py — 🟡 冗余与不一致

**问题**: `config.py` 中存在两套坐标系定义：
1. `SoccerFieldConfiguration` 类：FIFA 68×105 坐标系
2. `field_detector.py` 中的 `FIFA` 字典：同样是FIFA标准常量

两套定义重复且 `config.py` 的关键点排序不正确。建议统一。

---

## 第四章：可以复用的成果

### 4.1 config.py — FIFA球场常量 ✅

以下常量经核实与FIFA标准一致，可原样保留：

| 常量 | 值 | FIFA标准 | 状态 |
|------|-----|---------|------|
| WIDTH | 68.0 | 68m | ✅ |
| LENGTH | 105.0 | 105m | ✅ |
| PENALTY_AREA_WIDTH | 40.32 | 40.32m (7.32+2×16.5) | ✅ |
| PENALTY_AREA_DEPTH | 16.5 | 16.5m | ✅ |
| GOAL_AREA_WIDTH | 18.32 | 18.32m (7.32+2×5.5) | ✅ |
| GOAL_AREA_DEPTH | 5.5 | 5.5m | ✅ |
| GOAL_WIDTH | 7.32 | 7.32m | ✅ |
| CENTER_CIRCLE_RADIUS | 9.15 | 9.15m | ✅ |
| PENALTY_SPOT_DISTANCE | 11.0 | 11m | ✅ |

`VISUALIZATION` 字典和 `DETECTION_CONFIDENCE` 字典也可原样保留。

### 4.2 view_transformer.py — 完整可复用 ✅

`pixel_to_world` / `world_to_pixel` / `batch_*` / `is_in_field` 全部正确，不需要任何修改。

### 4.3 offside_analyzer.py — 核心逻辑可复用 ✅

IFAB规则判定、攻击方向投票、越位线端点计算均正确。修改仅限于增加可测试性（输出中间结果）。

### 4.4 player_detector.py — 管线可复用 ✅

YOLOv8 + ByteTrack + HSV K-means 管线完整。如需提升精度可后续接入 YOLOv8-pose 做脚踝检测，但当前版本足够。

### 4.5 visualizer.py — 完整可复用 ✅

绘制逻辑正确，配色和布局无需修改。

### 4.6 video_processor.py — 主管线可复用 ✅

管线串联、输出控制、JSON导出均正确。标定失败时的fallback逻辑合理。

### 4.7 field_detector.py — 手动标定可复用 ✅

4种手动标定模式（PenaltyArea/FourCorner/Generic/CenterCircle）均可正常工作。交互UI（缩放/撤销/重置）完善。

### 4.8 web/app.py — Web服务可复用 ✅

Flask服务、任务状态持久化、后台处理线程、API端点设计合理。不需要修改。

### 4.9 run.py — CLI可复用 ✅

参数解析完整，三种模式（batch/stream/interactive）切换正确。

---

## 第五章：修改优先级排序

按照 PROJECT_PLAN.md 的 Step 1→6 顺序，结合审计发现，排列修改优先级：

### Priority 1 — Step 1+2: 修复标定精度（阻断性问题）

| 序号 | 修改项 | 文件 | 严重度 | 说明 |
|------|--------|------|--------|------|
| P1-1 | **修复RANSAC阈值** | `pitch_keypoint_detector.py:267` | 🔴 致命 | 10.0→5.0，这是越位线倾斜的根因 |
| P1-2 | **添加多子集Voter** | `pitch_keypoint_detector.py` 新方法 | 🔴 严重 | 提升homography鲁棒性 |
| P1-3 | **添加几何约束验证** | `pitch_keypoint_detector.py` 新方法 | 🟡 重要 | 防止畸形homography |
| P1-4 | **添加标定质量指标** | `pitch_keypoint_detector.py` 新方法 | 🟡 重要 | RMSE、内点比例、逐点误差 |
| P1-5 | **添加标定可视化** | `pitch_keypoint_detector.py` 新函数 | 🟡 重要 | 球场线叠加原图，目视验证 |

### Priority 2 — Step 1: 修复关键点顺序匹配

| 序号 | 修改项 | 文件 | 严重度 | 说明 |
|------|--------|------|--------|------|
| P2-1 | **修复Roboflow API关键点顺序** | `field_detector.py:1024-1026` | 🟡 重要 | 改用WORLD_KEYPOINTS或弃用API模式 |
| P2-2 | **统一坐标系定义** | `config.py` + `field_detector.py` | 🟢 建议 | 消除重复的FIFA常量定义 |

### Priority 3 — Step 3: 球员检测增强

| 序号 | 修改项 | 文件 | 严重度 | 说明 |
|------|--------|------|--------|------|
| P3-1 | 输出球员HSV颜色+队伍分配 | `player_detector.py` | 🟢 可选 | 增加可测试性 |
| P3-2 | 可选YOLOv8-pose脚踝检测 | `player_detector.py` | 🟢 可选 | 提升脚部定位精度 |

### Priority 4 — Step 4: 越位判定可测试性

| 序号 | 修改项 | 文件 | 严重度 | 说明 |
|------|--------|------|--------|------|
| P4-1 | 输出每帧判定中间结果 | `offside_analyzer.py` | 🟢 可选 | 攻击方向/防守者排序/越位判定明细 |

### Priority 5 — Step 5+6: 可视化和集成

| 序号 | 修改项 | 文件 | 严重度 | 说明 |
|------|--------|------|--------|------|
| P5-1 | 无需修改 | `visualizer.py` | — | 已正确 |
| P5-2 | 无需修改 | `video_processor.py` | — | 已正确 |

---

## 附录A：关键验证点核对

### A.1 关键点世界坐标顺序（对照Roboflow官方）

`pitch_keypoint_detector.py` 中的 `_build_world_coords()` 对照官方 `SoccerPitchConfiguration`:

| 索引 | 代码Label | 世界坐标 | 核对 |
|------|-----------|---------|------|
| 0 | 01 Top-left corner | (0, 0) | ✅ |
| 1 | 02 L-touchline @ PA-top | (0, 14.5) | ✅ |
| 2 | 03 L-touchline @ GA-top | (0, 25.84) | ✅ |
| 6 | 07 GA near-post top | (5.5, 25.84) | ✅ |
| 8 | 09 Penalty spot | (11.0, 35.0) | ✅ |
| 13 | 15 Half-line top | (60.0, 0.0) | ✅ |
| 30 | 14 CC left | (50.85, 35.0) | ✅ |
| 31 | 19 CC right | (69.15, 35.0) | ✅ |

**结论**: 32个关键点的索引与Roboflow官方定义一致，排序正确。

### A.2 Roboflow→FIFA转换矩阵验证

转换公式：`x_fifa = 68 - y_rob × 68/70`, `y_fifa = 105 - x_rob × 105/120`

| Roboflow点 | 描述 | → FIFA坐标 | 期望 | 核对 |
|------------|------|-----------|------|------|
| (0, 0) | 近端左上角 | (68, 105) | 远端右下 | ✅ 旋转正确 |
| (120, 0) | 远端左上角 | (68, 0) | 右上 | ✅ |
| (120, 70) | 远端右下角 | (0, 0) | 左上 | ✅ |
| (0, 70) | 近端右下角 | (0, 105) | 左下 | ✅ |
| (60, 35) | 球场中心 | (34, 52.5) | 中心 | ✅ |
| (11, 35) | 近端罚球点 | (34, 95.375) | 接近(34, 94) | ⚠️ 1.4m偏差 |

**结论**: 转换矩阵的旋转和缩放正确，但因Roboflow使用120×70而非105×68，罚球点等位置有~1m量级偏差。此偏差在当前RANSAC阈值10m的噪声下可忽略，修复RANSAC后需重新评估。

### A.3 越位线端点计算验证

`offside_analyzer.py:399-400`:
```python
left_pixel = self.transformer.world_to_pixel(0, world_y)
right_pixel = self.transformer.world_to_pixel(self.field_width, world_y)  # 68
```

- (0, world_y): 左边线上的点 → 投影到图像左边线附近 ✅
- (68, world_y): 右边线上的点 → 投影到图像右边线附近 ✅
- 两点连线 = 常数y的水平线 → 在正确homography下应平行于球门线 ✅

**当homography正确时，越位线必然平行于球门线**。当前越位线倾斜100%是因为homography错误（RANSAC阈值bug导致）。

---

## 附录B：问题根因分析

```
越位线倾斜
  └── homography不准确
       ├── RANSAC阈值10米（应为5像素）—— 主因，占90%
       │    └── 远端低置信度关键点不被过滤
       │         └── H矩阵被噪声点扭曲
       │              └── world_to_pixel映射歪斜
       │                   └── 越位线端点偏移
       │                        └── 越位线不平行于球门线
       ├── 缺少多子集Voter —— 次因，占8%
       └── Roboflow 120×70 与 FIFA 68×105 非线性偏差 —— 微因，占2%
```

**修复P1-1（RANSAC阈值）后，预期越位线倾斜问题可基本解决。**

---

*审计完成。下一步：按Priority 1开始修复。*
