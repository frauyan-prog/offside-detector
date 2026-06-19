# 越位识别系统 — 测试报告

**日期**: 2026-06-17  
**测试工程师**: Ada  
**系统版本**: 越位识别 offside_detector  
**报告路径**: `D:\workBuddy\越位识别\tests\TEST_REPORT_2026-06-17.md`

---

## 一、测试概述

本报告对越位识别系统在3个测试视频上的输出结果进行离线数据核查。核查内容包括：

1. 球场校准质量（从校准JSON提取关键指标）
2. 越位检测分析结果（从分析JSON提取帧统计数据）
3. 校准可视化检查（calib_latest.jpg）
4. 核心代码调用链审查（pitch_keypoint_detector.py）

---

## 二、球场校准结果

### 2.1 校准JSON文件概览

| 指标 | 11号视频 | 12号视频 | 13号视频 |
|------|---------|---------|---------|
| JSON文件 | `11-非有意触球-25中场-30-233-北京-梅州-越位-获利-6_offside_calibration.json` | `12-非有意触球-25中超-5-37-成都蓉城-大连英博海发-越位-获利-21_offside_calibration.json` | `13-有意处理球_offside_calibration.json` |
| calibration_mode | auto | auto | auto |
| is_reliable | true | true | true |
| field_orientation | 0 | 0 | 0 |
| pixel_points数量 | 32 | 32 | 32 |
| world_points数量 | 32 | 32 | 32 |

### 2.2 关键质量指标（n_detected / n_used / quality）

**重要说明**: 校准JSON文件中未直接存储 `n_detected`、`n_used`、`quality` 字段。这些指标在 `PitchKeypointDetector.calibrate()` 方法的 `info_dict` 中计算（来源代码 `pitch_keypoint_detector.py`），但写入校准JSON时未导出这些字段。

- `n_detected` = `pixel_kpts[:, 2] > 0.0` 的计数（置信度>0的检测点数）
- `n_used` = `pixel_kpts[:, 2] > 0.3` 的计数（置信度>0.3的有效点数）
- `quality` = `pixel_kpts[:, 2]` 的均值（所有点置信度平均值）

校准JSON中仅保存了 `pixel_points`（2D坐标，无置信度值），因此无法从JSON直接回算以上指标。

| 指标 | 11号视频 | 12号视频 | 13号视频 | 备注 |
|------|---------|---------|---------|------|
| n_detected | **不可计算** | **不可计算** | **不可计算** | JSON中缺少confidence数据 |
| n_used | **不可计算** | **不可计算** | **不可计算** | JSON中缺少confidence数据 |
| quality | **不可计算** | **不可计算** | **不可计算** | JSON中缺少confidence数据 |

**建议**: 修改 `calibrate()` 方法的JSON输出逻辑，将 `info_dict` 中的 `n_detected`、`n_used`、`quality` 写入校准JSON，便于后续离线审计。

### 2.3 坐标系信息

校准使用 Roboflow 120×70 坐标系（FIFA 68×105 场的近似变换）：
- x ∈ [0, 120]：长度方向（近端球门线 → 远端球门线）
- y ∈ [0, 70]：宽度方向（左侧边线 → 右侧边线）
- 中点：(60.0, 35.0)

---

## 三、越位检测分析结果

### 3.1 帧统计数据

| 指标 | 11号视频 | 12号视频 | 13号视频 |
|------|---------|---------|---------|
| 视频文件 | `11-非有意触球-25中场-30-233-北京-梅州-越位-获利-6.mp4` | `12-非有意触球-25中超-5-37-成都蓉城-大连英博海发-越位-获利-21.mp4` | `13-有意处理球.mp4` |
| total_frames | **779** | **616** | **736** |
| processed_frames | **300** | **300** | **15** |
| offside_frames | **281** | **48** | **10** |
| total_offside_instances | **699** | **51** | **19** |
| 越位帧占比 | **36.1%** (281/779) | **7.8%** (48/616) | **1.4%** (10/736) |
| 越位帧/处理帧 | **93.7%** (281/300) | **16.0%** (48/300) | **66.7%** (10/15) |

### 3.2 进攻方向

| 视频 | attack_direction | 说明 |
|------|-----------------|------|
| 11号 | **top_to_bottom** | 所有帧统一方向 |
| 12号 | **bottom_to_top** | 所有帧统一方向 |
| 13号 | **mixed** | 帧0-2为 `bottom_to_top`，帧3-14为 `top_to_bottom` |

> **注意**: 13号视频（有意处理球）存在进攻方向切换现象，需确认是否符合视频内容实际情况。

### 3.3 关键发现

1. **11号视频**：越位帧占比极高（93.7%的处理帧检测到越位），主要越位球员为 track_id=7（attacking team），越位距离在 1-8m 范围内。后期出现大量 "unknown" team 检测，可能存在球员分类问题。

2. **12号视频**：越位帧占比较低（16.0%的处理帧），越位球员 track_id=10 的越位距离在 0.2-7m 范围内。第29帧之后出现大量 clean（非越位）帧。

3. **13号视频**：仅处理15帧（处理帧数异常少），10帧检测到越位，越位距离在 27-88m 范围内，距离异常偏大，**疑似坐标系转换或关键点检测存在错误**。

---

## 四、校准可视化检查

### 4.1 calib_latest.jpg

**文件路径**: `D:\workBuddy\越位识别\output\calib_latest.jpg`

**检查结果**: 由于当前模型不支持图像多模态分析，**无法直接描述越位线是否倾斜**。建议以下替代方案：

1. 人工目视检查 `calib_latest.jpg` 中的绿色 pitch overlay 线条是否与真实球场边线对齐
2. 检查黄色 offside line 是否与球场底边平行（垂直于进攻方向）
3. 切换至多模态模型后重新进行此项检查

---

## 五、核心代码调用链审查

### 5.1 审查对象

**文件**: `D:\workBuddy\越位识别\offside_detector\pitch_keypoint_detector.py`  
**类**: `PitchKeypointDetector`  
**审查问题**: 以下四个方法是否被 `calibrate()` 或 `calibrate_multiframe()` 调用？

### 5.2 审查结论

| 方法 | 行号位置 | 是否被 calibrate() 调用 | 是否被 calibrate_multiframe() 调用 | 说明 |
|------|---------|----------------------|-------------------------------|------|
| `voter_homography()` | L289-429 | **否** | **否** | calibrate() 使用 `compute_homography()` 直接计算单应性矩阵 |
| `validate_homography()` | L446-491 | **否** | **否** | 该类中定义但未被校准流程调用，无 H 矩阵后验证环节 |
| `calibration_quality()` | L631-688 | **否** | **否** | 校准质量评估器存在于代码中但未被校准流程使用 |
| `draw_pitch_overlay()` | L692-801 | **否** | **否** | 静态方法，生成覆盖层图像，但不在校准流程中自动调用 |

### 5.3 实际调用链

```
calibrate()          → detect() → compute_homography() → (可选坐标转换)
calibrate_multiframe() → multi-frame detect() → 合并最佳关键点 → compute_homography() → (可选坐标转换)
```

### 5.4 风险评估

- **voter_homography() 未被调用**：该函数实现了子集投票机制（A/B/C/D subset），理论上有助于提升 homography 的稳定性（滤除离群点）。目前系统使用单次 `compute_homography()` 可能对错误检测的关键点敏感。
- **validate_homography() 未被调用**：缺少 H 矩阵质量的门禁检查（如重投影误差阈值），可能导致使用低质量校准进行越位判断。
- **calibration_quality() 未被调用**：未能对每次校准自动评分，不利于自动化质量监控。

---

## 六、问题汇总

| 编号 | 问题描述 | 严重程度 | 建议 |
|------|---------|---------|------|
| P1 | 校准JSON缺少 n_detected/n_used/quality 字段 | 中 | 修改 JSON 输出逻辑，将 info_dict 写入校准文件 |
| P2 | validator_homography() 未集成到校准流程 | 高 | 在校准完成后自动调用 validate_homography() 进行 H 矩阵门禁检查 |
| P3 | voter_homography() 未集成到校准流程 | 中 | 评估 multi-subset voting 方案，替换或补充 compute_homography() |
| P4 | calibration_quality() 未集成到校准流程 | 低 | 自动生成每次校准的质量评分并写入日志/JSON |
| P5 | 13号视频 processed_frames 仅15帧 | 高 | 排查13号视频处理提前终止的原因 |
| P6 | 13号视频存在 direction switch | 中 | 确认视频内容是否确实存在进攻方向切换 |
| P7 | calib_latest.jpg 无法自动分析 | 低 | 考虑添加自动化越位线倾斜度计算（基于 H 矩阵和 world points） |

---

## 七、测试结论

1. **校准功能**：3个视频的校准均标记为 `is_reliable: true`，但缺少定量质量指标（n_detected/n_used/quality），需要补充 JSON 输出。
2. **越位检测**：11号视频检测正常（93.7%越位帧率与视频内容吻合），12号视频检测正常（16.0%），13号视频存在异常（处理帧数过少、越位距离过大）。
3. **代码审查**：4个重要方法（voter_homography/validate_homography/calibration_quality/draw_pitch_overlay）均未被核心校准流程调用，建议根据风险评估逐步集成。
4. **可视化**：`calib_latest.jpg` 需人工或切换多模态模型后进行检查。

---

*报告生成时间: 2026-06-17*  
*测试工程师: Ada*
