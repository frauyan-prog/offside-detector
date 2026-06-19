"""
Field Detector Module — Multi-Mode Calibration

Supports 3 calibration modes for different camera views:

  MODE A — 禁区标定 (Penalty Area)
    For tight shots near the goal. Visible: goal + penalty area line.
    Clicks: ① left post bottom  ② right post bottom
            ③ PA line left end  ④ PA line right end
    World coords automatically computed (FIFA standard).

  MODE B — 全场标定 (Four Corner)
    For wide shots showing the entire pitch.
    Clicks: ① TL  ② TR  ③ BR  ④ BL

  MODE C — 通用标定 (Generic Rectangle)
    For any visible rectangular field marking.
    Clicks: 4 corners of a visible rectangle → enter real dimensions.

Requires: opencv-python, numpy
"""

import json
import os
import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum


class CalibrationMode(Enum):
    AUTO = "auto"                        # local YOLOv8-pose auto detection
    PENALTY_AREA = "penalty_area"       # near-goal shots
    FOUR_CORNER = "four_corner"          # wide full-pitch shots
    GENERIC = "generic"                   # any visible rectangle
    CENTER_CIRCLE = "center_circle"      # midfield, 2-point circle diameter


# ─────────────────────────────────────────────────────────────
# FIFA Standard Field Coordinates (meters)
# Origin: goal line at y=0, sideline at x=0
# Field: 68m wide × 105m long
# ─────────────────────────────────────────────────────────────

FIFA = {
    "field_width": 68.0,
    "field_length": 105.0,
    "half_width": 34.0,
    "half_length": 52.5,
    # Goal
    "goal_width": 7.32,       # 2.44m × 7.32m
    "goal_left_x": (68.0 - 7.32) / 2,   # 30.34
    "goal_right_x": (68.0 + 7.32) / 2,  # 37.66
    # Goal area (小禁区 / 6-yard box): 5.5m from goal line, 18.32m wide
    "ga_depth": 5.5,
    "ga_half_width": 9.16,   # 18.32 / 2
    # Penalty area (大禁区 / 18-yard box): 16.5m from goal line, 40.32m wide
    "pa_depth": 16.5,
    "pa_half_width": 20.16,  # 40.32 / 2
    # Penalty spot: 11m from goal line
    "penalty_spot_y": 11.0,
    # Center circle: radius 9.15m
    "center_radius": 9.15,
    # Corner arc: radius 1m
    "corner_radius": 1.0,
}


@dataclass
class FieldDetectionResult:
    """Result of field calibration."""
    keypoints: np.ndarray = None           # (N, 2) pixel coordinates used
    confidences: np.ndarray = None         # (N,) always 1.0 for manual
    homography: Optional[np.ndarray] = None
    inverse_homography: Optional[np.ndarray] = None
    is_reliable: bool = False
    num_valid_keypoints: int = 0
    calibration_mode: str = "none"
    # Metadata
    world_keypoints: Optional[np.ndarray] = None  # corresponding world coords
    field_orientation: int = 0  # 0=goal at y=0 is visible, 1=goal at y=105
    # Quality metrics (auto calibration only)
    mean_confidence: float = 0.0
    n_detected: int = 0
    n_used: int = 0
    calibration_rmse: float = 0.0
    calibration_rmse_pixels: float = 0.0
    n_inliers_calib: int = 0
    voter_rmse: float = 0.0
    subset: str = ""

    def __post_init__(self):
        if self.keypoints is None:
            self.keypoints = np.zeros((0, 2), dtype=np.float32)
        if self.confidences is None:
            self.confidences = np.zeros(0, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# Calibration Coordinate Builders
# ─────────────────────────────────────────────────────────────

def build_penalty_area_world_coords(facing_goal_y0: bool = True) -> np.ndarray:
    """
    Build world coordinates for penalty area calibration.

    Points (in click order):
      1. Left goalpost base     → (goal_left_x, 0) or (goal_left_x, 105)
      2. Right goalpost base    → (goal_right_x, 0) or (goal_right_x, 105)
      3. PA line left end       → (pa_left_x, 16.5) or (pa_left_x, 88.5)
      4. PA line right end      → (pa_right_x, 16.5) or (pa_right_x, 88.5)

    Where:
      pa_left_x  = (68 - 40.32)/2 = 13.84
      pa_right_x = (68 + 40.32)/2 = 54.16
    """
    gx_l = FIFA["goal_left_x"]    # 30.34
    gx_r = FIFA["goal_right_x"]   # 37.66
    pa_l = (68.0 - 40.32) / 2     # 13.84
    pa_r = (68.0 + 40.32) / 2     # 54.16

    if facing_goal_y0:
        # Goal line at y=0, PA line at y=16.5
        return np.array([
            [gx_l, 0.0],           # left post
            [gx_r, 0.0],           # right post
            [pa_l, 16.5],          # PA left
            [pa_r, 16.5],          # PA right
        ], dtype=np.float32)
    else:
        # Goal line at y=105, PA line at y=88.5
        return np.array([
            [gx_l, 105.0],
            [gx_r, 105.0],
            [pa_l, 88.5],
            [pa_r, 88.5],
        ], dtype=np.float32)


def build_four_corner_world_coords() -> np.ndarray:
    """Build world coordinates for full-field 4-corner calibration."""
    return np.array([
        [0, 0],          # TL
        [68, 0],         # TR
        [68, 105],       # BR
        [0, 105],        # BL
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# Base Calibrator (shared UI logic)
# ─────────────────────────────────────────────────────────────

class BaseCalibrator:
    """Shared interactive calibration UI logic."""

    def __init__(self):
        self._click_points: List[Tuple[float, float]] = []
        self._display_frame: Optional[np.ndarray] = None
        self._original_frame: Optional[np.ndarray] = None
        self._scale: float = 1.0
        self._num_required: int = 4
        self._window_name: str = "Calibration"

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize to fit screen, store scale factor. Keeps original if it fits."""
        self._original_frame = frame.copy()
        h, w = frame.shape[:2]
        # Target: fit within 90% of a 1920x1080 screen, but never scale UP
        max_display_w = 1700
        max_display_h = 900
        if w <= max_display_w and h <= max_display_h:
            self._scale = 1.0
            return frame.copy()
        scale_w = max_display_w / w
        scale_h = max_display_h / h
        self._scale = min(scale_w, scale_h)
        return cv2.resize(frame, (int(w * self._scale), int(h * self._scale)))

    def _unscale_point(self, x: float, y: float) -> Tuple[float, float]:
        """Convert display coordinates back to original image coordinates."""
        return (x / self._scale, y / self._scale)

    def _on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self._click_points) < self._num_required:
                self._click_points.append((float(x), float(y)))

    def _run_ui(self, frame: np.ndarray, title: str, instructions: List[str],
                click_labels: List[str], num_points: int = 4) -> Optional[List[Tuple[float, float]]]:
        """
        Run interactive calibration UI.

        Returns:
            List of (px, py) in ORIGINAL image coordinates, or None if cancelled.
        """
        self._click_points = []
        self._num_required = num_points
        self._display_frame = self._prepare_frame(frame)
        self._window_title_text = title

        # Use a short ASCII window name to avoid Win32 multi-window bugs
        self._window_name = "OffsideCalibration"
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        # Resize window to match the display frame dimensions
        dh, dw = self._display_frame.shape[:2]
        cv2.resizeWindow(self._window_name, dw, dh)
        cv2.setMouseCallback(self._window_name, self._on_click)

        print("\n" + "=" * 60)
        print(f"  {title}")
        print("=" * 60)
        for line in instructions:
            print(f"  {line}")
        print()
        print("  Controls: 鼠标左键=点击 | Z=撤销 | R=重置 | ENTER=确认 | ESC=取消")
        print("=" * 60 + "\n")

        while True:
            display = self._display_frame.copy()
            self._draw_overlay(display, instructions, click_labels)
            cv2.imshow(self._window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                print("[Calibration] Cancelled.")
                cv2.destroyWindow(self._window_name)
                cv2.waitKey(1)  # let the OS process window close
                return None
            if key == ord('z'):
                if self._click_points:
                    removed = self._click_points.pop()
                    print(f"[Calibration] Undo point {len(self._click_points)+1}: {removed}")
            if key == ord('r'):
                self._click_points.clear()
                print("[Calibration] All points reset.")
            if key == 13:  # Enter
                if len(self._click_points) == num_points:
                    print(f"[Calibration] {num_points} points collected. Computing...")
                    break
                else:
                    print(f"[Calibration] Need {num_points} points, have {len(self._click_points)}.")

        cv2.destroyWindow(self._window_name)
        cv2.waitKey(1)

        # Unscale points back to original image coordinates
        return [self._unscale_point(x, y) for x, y in self._click_points]

    def _draw_overlay(self, display: np.ndarray, instructions: List[str],
                      click_labels: List[str]):
        h, w = display.shape[:2]
        n = len(self._click_points)
        total = self._num_required

        # Semi-transparent top bar for title
        overlay = display.copy()
        bar_height = 45
        cv2.rectangle(overlay, (0, 0), (w, bar_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

        # Title text (Chinese)
        title_text = getattr(self, '_window_title_text', '')
        cv2.putText(display, title_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Instruction text below title bar
        if n < total:
            text = click_labels[n] if n < len(click_labels) else f"Click point {n+1}/{total}"
        else:
            text = "Press ENTER to confirm"
        cv2.putText(display, text, (20, 58), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 255), 2)

        # Draw points
        for i, pt in enumerate(self._click_points):
            px, py = int(pt[0]), int(pt[1])
            color = (0, 255, 0) if i < total else (0, 0, 255)
            cv2.circle(display, (px, py), 7, color, -1)
            cv2.circle(display, (px, py), 9, (255, 255, 255), 2)
            label = str(i + 1)
            cv2.putText(display, label, (px + 15, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # Draw connecting lines
        pts = np.array(self._click_points[:total], dtype=np.int32)
        if n >= 2:
            for i in range(min(n, total) - 1):
                cv2.line(display, tuple(pts[i]), tuple(pts[i+1]), (0, 255, 255), 2)
        if n == total:
            cv2.line(display, tuple(pts[-1]), tuple(pts[0]), (0, 255, 255), 2)

        # Progress bar
        bar_w, bar_h = 300, 6
        bar_x, bar_y = (w - bar_w) // 2, h - 40
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (80, 80, 80), -1)
        fill = int(bar_w * n / total)
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h),
                      (0, 255, 0), -1)

        # Point count
        cv2.putText(display, f"{n}/{total}", (bar_x + bar_w + 10, bar_y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def _compute_homography(self, pixel_pts: np.ndarray,
                            world_pts: np.ndarray) -> FieldDetectionResult:
        """Compute homography from pixel↔world point pairs."""
        try:
            H, mask = cv2.findHomography(
                pixel_pts, world_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
            )
            if H is not None:
                return FieldDetectionResult(
                    keypoints=pixel_pts,
                    confidences=np.ones(len(pixel_pts), dtype=np.float32),
                    homography=H,
                    inverse_homography=np.linalg.inv(H),
                    is_reliable=True,
                    num_valid_keypoints=len(pixel_pts),
                    world_keypoints=world_pts,
                )
        except Exception as e:
            print(f"[Calibration] Homography failed: {e}")

        return FieldDetectionResult(
            keypoints=pixel_pts,
            confidences=np.ones(len(pixel_pts), dtype=np.float32),
            is_reliable=False,
            world_keypoints=world_pts,
        )


# ─────────────────────────────────────────────────────────────
# MODE A: Penalty Area Calibration
# ─────────────────────────────────────────────────────────────

class PenaltyAreaCalibrator(BaseCalibrator):
    """
    Calibrate using the penalty area + goalposts.

    Works for: tight camera shots near the goal where you can see
               the goal and the penalty area (16.5m) line.

    Click order (4 points):
      1. Left goalpost — bottom of the left post
      2. Right goalpost — bottom of the right post
      3. Penalty area line LEFT end — where the 16.5m line meets the left side
      4. Penalty area line RIGHT end — where the 16.5m line meets the right side

    World coordinates (y=0 at near goal line):
      Point 1: (30.34, 0.0)    — left post
      Point 2: (37.66, 0.0)    — right post
      Point 3: (13.84, 16.5)   — PA line left
      Point 4: (54.16, 16.5)   — PA line right
    """

    MODE = CalibrationMode.PENALTY_AREA

    TITLE = "禁区标定 — 点击球门柱 + 禁区线"

    INSTRUCTIONS = [
        "请按顺序点击以下 4 个点（在画面中可见的）：",
        "",
        "  ① 左门柱底部      — 球门左边门柱与地面的接触点",
        "  ② 右门柱底部      — 球门右边门柱与地面的接触点",
        "  ③ 禁区线左端点    — 16.5m线（大禁区线）的左端",
        "  ④ 禁区线右端点    — 16.5m线（大禁区线）的右端",
        "",
        "提示：不需要精确到像素，大致位置即可。",
        "      如果禁区线两端不完全可见，请点击画面边缘的可见位置。",
    ]

    CLICK_LABELS = [
        "① 点击：左门柱底部",
        "② 点击：右门柱底部",
        "③ 点击：禁区线左端",
        "④ 点击：禁区线右端",
    ]

    def calibrate(self, frame: np.ndarray) -> Optional[FieldDetectionResult]:
        """
        Run penalty area calibration.

        Returns:
            FieldDetectionResult with homography, or None if cancelled.
        """
        result = self._run_ui(
            frame,
            title=self.TITLE,
            instructions=self.INSTRUCTIONS,
            click_labels=self.CLICK_LABELS,
            num_points=4,
        )
        if result is None:
            return None

        pixel_pts = np.array(result, dtype=np.float32)
        world_pts = build_penalty_area_world_coords(facing_goal_y0=True)

        calib_result = self._compute_homography(pixel_pts, world_pts)
        calib_result.calibration_mode = "penalty_area"
        calib_result.field_orientation = 0
        print("[PenaltyArea] Calibration done.")
        return calib_result


# ─────────────────────────────────────────────────────────────
# MODE B: Four-Corner Calibration (full pitch)
# ─────────────────────────────────────────────────────────────

class FourCornerCalibrator(BaseCalibrator):
    """
    Calibrate using the 4 corners of the football pitch.

    Works for: wide shots showing the entire field.

    Click order:
      1. Top-Left corner
      2. Top-Right corner
      3. Bottom-Right corner
      4. Bottom-Left corner

    World coordinates: TL=(0,0), TR=(68,0), BR=(68,105), BL=(0,105)
    """

    MODE = CalibrationMode.FOUR_CORNER

    TITLE = "全场标定 — 点击球场四个角"

    INSTRUCTIONS = [
        "请按顺序点击球场的 4 个角：",
        "",
        "  ① 左上角 (TL)",
        "  ② 右上角 (TR)",
        "  ③ 右下角 (BR)",
        "  ④ 左下角 (BL)",
        "",
        "提示：角点 = 边线与底线/球门线的交点。",
        "      如果角旗可见，点击角旗底部。",
    ]

    CLICK_LABELS = [
        "① 点击：左上角",
        "② 点击：右上角",
        "③ 点击：右下角",
        "④ 点击：左下角",
    ]

    def calibrate(self, frame: np.ndarray) -> Optional[FieldDetectionResult]:
        result = self._run_ui(
            frame,
            title=self.TITLE,
            instructions=self.INSTRUCTIONS,
            click_labels=self.CLICK_LABELS,
            num_points=4,
        )
        if result is None:
            return None

        pixel_pts = np.array(result, dtype=np.float32)
        world_pts = build_four_corner_world_coords()

        calib_result = self._compute_homography(pixel_pts, world_pts)
        calib_result.calibration_mode = "four_corner"
        print("[FourCorner] Calibration done.")
        return calib_result


# ─────────────────────────────────────────────────────────────
# MODE C: Generic Rectangle Calibration
# ─────────────────────────────────────────────────────────────

class GenericRectangleCalibrator(BaseCalibrator):
    """
    Calibrate using any visible rectangle on the field.

    User clicks 4 corners of a visible rectangular area, then enters
    the real-world dimensions (width × height in meters).

    Common references:
      - Penalty area:    40.32m × 16.5m
      - Goal area:       18.32m × 5.5m
      - Center circle box: 18.3m × 18.3m (diameter of center circle)
      - Half of field:   68m × 52.5m
    """

    MODE = CalibrationMode.GENERIC

    TITLE = "通用标定 — 点击任意矩形区域的四个角"

    INSTRUCTIONS = [
        "请点击画面中可见的任意矩形场地区域的 4 个角：",
        "",
        "  ① 左上角",
        "  ② 右上角",
        "  ③ 右下角",
        "  ④ 左下角",
        "",
        "常用参照：",
        "  · 大禁区整体:  40.32m宽 × 16.5m深（2个禁区线端点 + 2个门柱）",
        "  · 大禁区线本身: 40.32m宽 × 0m（仅线两端，系统自动处理）",
        "  · 中圈矩形:     18.3m × 18.3m",
        "  · 小禁区:       18.32m × 5.5m",
        "",
        "标定完成后会要求输入实际尺寸。",
    ]

    CLICK_LABELS = [
        "① 点击：左上角",
        "② 点击：右上角",
        "③ 点击：右下角",
        "④ 点击：左下角",
    ]

    # Preset dimensions for common field areas
    PRESETS = {
        "1": {"name": "大禁区 (40.32m × 16.5m)", "w": 40.32, "h": 16.5},
        "2": {"name": "小禁区 (18.32m × 5.5m)", "w": 18.32, "h": 5.5},
        "3": {"name": "中圈 (18.3m × 18.3m)", "w": 18.3, "h": 18.3},
        "4": {"name": "半场 (68m × 52.5m)", "w": 68.0, "h": 52.5},
    }

    def calibrate(self, frame: np.ndarray) -> Optional[FieldDetectionResult]:
        result = self._run_ui(
            frame,
            title=self.TITLE,
            instructions=self.INSTRUCTIONS,
            click_labels=self.CLICK_LABELS,
            num_points=4,
        )
        if result is None:
            return None

        # Ask for dimensions
        real_w, real_h = self._ask_dimensions()

        pixel_pts = np.array(result, dtype=np.float32)
        # Assume the rectangle is aligned with field axes
        # Top-left = (0, 0), Top-right = (w, 0),
        # Bottom-right = (w, h), Bottom-left = (0, h)
        world_pts = np.array([
            [0, 0],
            [real_w, 0],
            [real_w, real_h],
            [0, real_h],
        ], dtype=np.float32)

        calib_result = self._compute_homography(pixel_pts, world_pts)
        calib_result.calibration_mode = "generic"
        print(f"[Generic] Calibration done with {real_w:.1f}m × {real_h:.1f}m")
        return calib_result

    def _ask_dimensions(self) -> Tuple[float, float]:
        """Ask user for real-world dimensions of the rectangle."""
        print("\n" + "-" * 40)
        print("  选择标定区域的类型（输入编号）：")
        for key, preset in self.PRESETS.items():
            print(f"    {key}. {preset['name']}")
        print("    5. 自定义尺寸")
        print("-" * 40)

        while True:
            try:
                choice = input("  请输入 (1-5): ").strip()
                if choice in self.PRESETS:
                    p = self.PRESETS[choice]
                    print(f"  已选择: {p['name']}")
                    return p["w"], p["h"]
                elif choice == "5":
                    w = float(input("  宽度 (米): ").strip())
                    h = float(input("  高度/深度 (米): ").strip())
                    print(f"  自定义: {w:.1f}m × {h:.1f}m")
                    return w, h
                else:
                    print("  无效输入，请输入 1-5")
            except (ValueError, EOFError):
                print("  输入无效，使用默认：大禁区 40.32m × 16.5m")
                return 40.32, 16.5


# ─────────────────────────────────────────────────────────────
# MODE D: Center Circle Calibration (2 points only)
# ─────────────────────────────────────────────────────────────

class CenterCircleCalibrator(BaseCalibrator):
    """
    Calibrate using only 2 points on the center circle's horizontal diameter.

    Works for: midfield shots where only the center circle is visible.
    System synthesizes 2 additional virtual points (above and below the
    clicked line) to create a 4-point homography.

    Click order (2 points):
      Point 1: LEFT endpoint of center circle's horizontal diameter
      Point 2: RIGHT endpoint of center circle's horizontal diameter

    World coordinates (center of field at (34, 52.5)):
      Point 1: (25.85, 52.5)   — left  of center (34 - 18.3/2)
      Point 2: (43.15, 52.5)   — right of center (34 + 18.3/2)

    Virtual points added automatically:
      Point 3: (25.85, 42.5)   — same x, -10m in y
      Point 4: (43.15, 62.5)   — same x, +10m in y
    """

    MODE = CalibrationMode.CENTER_CIRCLE
    CENTER_CIRCLE_DIAMETER = 18.3  # meters (radius 9.15m)
    CENTER_X = 34.0                # center of field
    CENTER_Y = 52.5                # halfway line

    TITLE = "中圈标定 — 点击中圈直径两端（仅需2点）"

    INSTRUCTIONS = [
        "您的画面只显示中圈区域，请点击中圈水平直径的两个端点：",
        "",
        "  ① 左端点  — 中圈水平直径的左侧端点",
        "  ② 右端点  — 中圈水平直径的右侧端点",
        "",
        "提示：点击中圈白线的最左端和最右端。",
        "      系统会自动根据 FIFA 标准尺寸补全其余参考点。",
        "",
        "中圈直径 = 18.3米（半径9.15米），位于球场正中心。",
    ]

    CLICK_LABELS = [
        "① 点击：中圈直径左端",
        "② 点击：中圈直径右端",
    ]

    def calibrate(self, frame: np.ndarray, attack_dir: str = None) -> Optional[FieldDetectionResult]:
        """Run 2-point center circle calibration."""
        result = self._run_ui(
            frame,
            title=self.TITLE,
            instructions=self.INSTRUCTIONS,
            click_labels=self.CLICK_LABELS,
            num_points=2,
        )
        if result is None:
            return None

        return self._build_from_2_points(result, attack_dir=attack_dir)

    def _build_from_2_points(
        self, pixel_points: List[Tuple[float, float]],
        attack_dir: str = None,
    ) -> FieldDetectionResult:
        """
        Build a 4-point homography from 2 user-clicked points on the
        center circle's diameter.

        Approach:
          1. World coords for the 2 clicked points (real + virtual).
          2. Synthesize 2 virtual pixel points perpendicular to the
             clicked line at its midpoint, offset by a fraction of
             the line length (to estimate depth in image space).
          3. Compute homography from 4 point pairs.

        attack_dir: Optional override for field orientation.
          - "left_to_right" / "right_to_left": Field length is horizontal
            in image; vertical diameter maps to WIDTH axis (world_x).
          - None / "top_to_bottom" / "bottom_to_top": Original behavior.
            Vertical diameter maps to LENGTH axis (world_y).
        """
        p1_px, p1_py = pixel_points[0]
        p2_px, p2_py = pixel_points[1]

        # Pixel midpoint + line vector
        mid_px = (p1_px + p2_px) / 2.0
        mid_py = (p1_py + p2_py) / 2.0
        dx = p2_px - p1_px
        dy = p2_py - p1_py
        line_len = np.sqrt(dx * dx + dy * dy)

        if line_len < 10:
            print("[CenterCircle] Points too close together — need wider span")
            return FieldDetectionResult(
                keypoints=np.array(pixel_points, dtype=np.float32),
                is_reliable=False,
                calibration_mode="center_circle",
            )

        # Perpendicular direction (rotate 90° CCW)
        # Normalized perpendicular vector
        perp_x = -dy / line_len
        perp_y = dx / line_len

        # Estimate pixel depth for ~5m vertical offset
        # Known: line_len pixels = 18.3m horizontally
        # Expected vertical pixel-per-meter ratio is similar but may be
        # reduced by perspective — use a conservative factor of 0.6
        ppm = line_len / 18.3  # pixels per meter (horizontal)
        perp_offset = ppm * 5.0 * 0.6  # ~5m vertical × conservative factor

        # Virtual pixel points: above and below the line
        v1_px = mid_px + perp_x * perp_offset
        v1_py = mid_py + perp_y * perp_offset
        v2_px = mid_px - perp_x * perp_offset
        v2_py = mid_py - perp_y * perp_offset

        # Pixel points (4 total): left, right, virtual-below, virtual-above
        pixel_pts = np.array([
            [p1_px, p1_py],    # 0: left endpoint
            [p2_px, p2_py],    # 1: right endpoint
            [v1_px, v1_py],    # 2: virtual point (+perp direction)
            [v2_px, v2_py],    # 3: virtual point (-perp direction)
        ], dtype=np.float32)

        # World coordinates (center of field at (34, 52.5))
        r = self.CENTER_CIRCLE_DIAMETER / 2  # 9.15
        cx, cy = self.CENTER_X, self.CENTER_Y
        vert = 5.0  # vertical offset in meters

        # Determine orientation of the clicked diameter in the image.
        # If |dy| > |dx|: diameter appears vertical  (along y-axis / center line)
        # If |dx| > |dy|: diameter appears horizontal (along x-axis / width)
        is_vertical = abs(dy) > abs(dx)

        # Check if field is horizontally oriented (goals left-right)
        # attack_dir may be a string or AttackDirection enum
        if attack_dir:
            attack_dir_str = attack_dir.value if hasattr(attack_dir, 'value') else str(attack_dir)
            is_horizontal_field = attack_dir_str in ("left_to_right", "right_to_left")
        else:
            is_horizontal_field = False

        if is_vertical and is_horizontal_field:
            # ── HORIZONTAL FIELD ORIENTATION ──
            # Field length runs left-right in image; vertical diameter in
            # image = WIDTH direction (touchline-to-touchline), NOT length.
            # Both clicked points are at world_y=52.5 (center line), with
            # different world_x values.

            # Determine which clicked point maps to which side of width.
            # Upper point in image -> depends on camera angle, but we use
            # consistent mapping: upper->left(x=cx-r), lower->right(x=cx+r).
            # The attack_dir determines which direction "toward goal" means.
            if p1_py < p2_py:
                # p1 is above p2 in image
                world_pts = np.array([
                    [cx - r, cy],         # 0: p1 (upper) → left on center line
                    [cx + r, cy],         # 1: p2 (lower) → right on center line
                    [cx + r, cy - vert],  # 2: virtual: right + toward goal (y-)
                    [cx - r, cy + vert],  # 3: virtual: left  + away from goal (y+)
                ], dtype=np.float32)
            else:
                world_pts = np.array([
                    [cx + r, cy],         # 0: p1 (lower) → right on center line
                    [cx - r, cy],         # 1: p2 (upper) → left on center line
                    [cx - r, cy - vert],  # 2: virtual: left  + toward goal (y-)
                    [cx + r, cy + vert],  # 3: virtual: right + away from goal (y+)
                ], dtype=np.float32)

            print(f"[CenterCircle] HORIZONTAL FIELD mode (attack={attack_dir}): "
                  f"vertical diameter → width axis (world_x)")
        elif is_vertical:
            # Diameter along y-axis (center line direction). Standard vertical field.
            if p1_py > p2_py:
                # p1 is lower in image (near goal) -> world y = cy - r
                world_pts = np.array([
                    [cx,      cy - r],    # 0: p1 -> near goal side
                    [cx,      cy + r],    # 1: p2 -> far goal side
                    [cx + vert, cy - r],    # 2: virtual +x, near side
                    [cx - vert, cy + r],    # 3: virtual -x, far side
                ], dtype=np.float32)
            else:
                world_pts = np.array([
                    [cx,      cy + r],    # 0: p1 -> far goal side
                    [cx,      cy - r],    # 1: p2 -> near goal side
                    [cx - vert, cy + r],    # 2: virtual -x, far side
                    [cx + vert, cy - r],    # 3: virtual +x, near side
                ], dtype=np.float32)
        else:
            # Diameter along x-axis (width direction) - original logic.
            world_pts = np.array([
                [cx - r, cy],         # 0: left of center
                [cx + r, cy],         # 1: right of center
                [cx - r, cy + vert],  # 2: left,   +5m toward far goal
                [cx + r, cy - vert],  # 3: right,  -5m toward near goal
            ], dtype=np.float32)

        calib_result = self._compute_homography(pixel_pts, world_pts)
        calib_result.calibration_mode = "center_circle"
        calib_result.world_keypoints = world_pts

        # ── DEBUG LOG ──
        import os
        log_path = os.path.join(os.path.dirname(__file__), "..", "output", "debug_log.txt")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n" + "="*60 + "\n")
            f.write("[CALIB] CenterCircleCalibrator._build_from_2_points\n")
            f.write(f"  is_vertical={is_vertical}, is_horizontal_field={is_horizontal_field}\n")
            f.write(f"  attack_dir={attack_dir}\n")
            f.write(f"  clicked points: p1=({p1_px:.1f},{p1_py:.1f}) p2=({p2_px:.1f},{p2_py:.1f})\n")
            f.write(f"  dx={dx:.1f}, dy={dy:.1f}, line_len={line_len:.1f}px = 18.3m\n")
            f.write("  PIXEL pts (4):\n")
            for i, (px, py) in enumerate(pixel_pts):
                f.write(f"    [{i}] ({px:.1f}, {py:.1f})\n")
            f.write("  WORLD pts (4):\n")
            for i, (wx, wy) in enumerate(world_pts):
                f.write(f"    [{i}] ({wx:.1f}, {wy:.1f})\n")
            f.write(f"  homography H (3x3):\n")
            H = calib_result.homography_matrix
            for row in H:
                f.write(f"    [{row[0]:+.6e}  {row[1]:+.6e}  {row[2]:+.6e}]\n")
            # Test: pixel->world->pixel roundtrip
            import cv2
            for i in range(4):
                px, py = pixel_pts[i]
                wx, wy = world_pts[i]
                # Forward: pixel -> world (using H)
                pt = np.array([[[px, py]]], dtype=np.float32)
                wpt = cv2.perspectiveTransform(pt, H)
                f.write(f"  Test[{i}]: pixel=({px:.0f},{py:.0f}) -> world=({wpt[0][0][0]:.1f},{wpt[0][0][1]:.1f}) expected=({wx:.1f},{wy:.1f})\n")
            f.write("  NOTE: If world != expected, homography is inaccurate!\n")

        print(f"[CenterCircle] 2-point calibration: line={line_len:.0f}px → "
              f"18.3m, synthesised 4 points for homography")
        return calib_result


# ─────────────────────────────────────────────────────────────
# Unified Field Detector
# ─────────────────────────────────────────────────────────────

class ManualFieldCalibrator:
    """
    Unified entry point for all manual calibration modes.

    Automatically selects the best mode or lets user choose:

    Usage:
        cal = ManualFieldCalibrator()

        # Auto-select based on frame content (recommended)
        result = cal.calibrate_interactive(frame)

        # Or specify mode explicitly
        result = cal.calibrate_interactive(frame, mode="penalty_area")
        result = cal.calibrate_interactive(frame, mode="four_corner")
        result = cal.calibrate_interactive(frame, mode="generic")

        # Save / load
        cal.save_calibration(result, "calibration.json")
        result = cal.load_calibration("calibration.json")
    """

    def __init__(self):
        self._penalty = PenaltyAreaCalibrator()
        self._four_corner = FourCornerCalibrator()
        self._generic = GenericRectangleCalibrator()
        self._center_circle = CenterCircleCalibrator()

    def calibrate_interactive(
        self, frame: np.ndarray,
        mode: Optional[str] = None,
        attack_dir: Optional[str] = None,
    ) -> Optional[FieldDetectionResult]:
        """
        Run interactive calibration.

        Args:
            frame: First frame of the video (BGR numpy array)
            mode: "penalty_area", "four_corner", "generic", "center_circle", or None for auto-prompt
            attack_dir: Attack direction override ("left_to_right", etc.) for center_circle mode

        Returns:
            FieldDetectionResult or None if cancelled
        """
        if mode is None:
            mode = self._prompt_mode()

        if mode == "auto":
            return self.calibrate_auto(frame)
        elif mode == "penalty_area":
            return self._penalty.calibrate(frame)
        elif mode == "four_corner":
            return self._four_corner.calibrate(frame)
        elif mode == "generic":
            return self._generic.calibrate(frame)
        elif mode == "center_circle":
            return self._center_circle.calibrate(frame, attack_dir=attack_dir)
        else:
            print(f"[Calibration] Unknown mode: {mode}")
            return None

    def calibrate_from_points(
        self, pixel_points: List[Tuple[float, float]],
        world_points: Optional[List[Tuple[float, float]]] = None,
    ) -> FieldDetectionResult:
        """
        Calibrate from pre-defined point pairs.

        Args:
            pixel_points: 4+ (x, y) pixel coordinates
            world_points: 4+ (wx, wy) world coords. If None, uses penalty area preset.
        """
        src = np.array(pixel_points, dtype=np.float32)
        if world_points is None:
            dst = build_penalty_area_world_coords()
        else:
            dst = np.array(world_points, dtype=np.float32)

        calib = BaseCalibrator()
        result = calib._compute_homography(src, dst)
        result.calibration_mode = "manual_points"
        return result

    def load_calibration(self, json_path: str) -> Optional[FieldDetectionResult]:
        """Load calibration from a JSON file."""
        with open(json_path, "r") as f:
            data = json.load(f)

        pixel_pts = data.get("pixel_points", [])
        world_pts = data.get("world_points", None)
        mode = data.get("calibration_mode", "loaded")

        if len(pixel_pts) < 2:
            print(f"[Calibration] Need at least 2 pixel_points, got {len(pixel_pts)}")
            return None

        src = np.array(pixel_pts, dtype=np.float32)

        # Handle center_circle mode (2 points → synthesize 4)
        if mode == "center_circle" and len(pixel_pts) == 2:
            cc = CenterCircleCalibrator()
            return cc._build_from_2_points(pixel_pts)

        if len(pixel_pts) < 4:
            print(f"[Calibration] Need at least 4 pixel_points for mode '{mode}', got {len(pixel_pts)}")
            return None

        src = np.array(pixel_pts, dtype=np.float32)

        if world_pts and len(world_pts) >= 4:
            dst = np.array(world_pts, dtype=np.float32)
        else:
            # Default to penalty area coords
            dst = build_penalty_area_world_coords()

        calib = BaseCalibrator()
        result = calib._compute_homography(src, dst)
        result.calibration_mode = mode
        return result

    def save_calibration(self, result: FieldDetectionResult, json_path: str):
        """Save calibration result for reuse."""
        # For center_circle mode, save only the 2 user-clicked points (not the 4 synthesized ones)
        if result.calibration_mode == "center_circle":
            pixel_pts = result.keypoints[:2].tolist() if len(result.keypoints) >= 2 else result.keypoints.tolist()
            world_pts = None  # will be re-synthesized on load
        else:
            pixel_pts = result.keypoints.tolist()
            world_pts = result.world_keypoints.tolist() if result.world_keypoints is not None else None

        data = {
            "pixel_points": pixel_pts,
            "world_points": world_pts,
            "calibration_mode": result.calibration_mode,
            "is_reliable": result.is_reliable,
            "field_orientation": result.field_orientation,
        }
        # Include quality metrics for auto calibration
        if result.calibration_mode == "auto":
            data["quality"] = {
                "mean_confidence": round(result.mean_confidence, 4),
                "n_detected": result.n_detected,
                "n_used": result.n_used,
                "n_inliers": result.n_inliers_calib,
                "calibration_rmse_m": round(result.calibration_rmse, 3),
                "calibration_rmse_px": round(result.calibration_rmse_pixels, 1),
                "voter_rmse_m": round(result.voter_rmse, 3),
                "subset": result.subset,
            }
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[Calibration] Saved to {json_path}")

    def calibrate_auto(self, frame: np.ndarray, video_path: Optional[str] = None) -> Optional[FieldDetectionResult]:
        """
        Fully automatic calibration using local YOLOv8-pose model.
        No user clicks needed — the model detects 32 field keypoints automatically.

        Args:
            frame: First frame (BGR numpy array).
            video_path: If provided, uses multi-frame sampling for better coverage.

        Returns:
            FieldDetectionResult with homography, or None if auto-detection failed.
        """
        try:
            from .pitch_keypoint_detector import PitchKeypointDetector
        except ImportError as e:
            print(f"[AutoCalibrator] Cannot import detector: {e}")
            return None

        print("\n" + "=" * 60)
        print("  AUTO CALIBRATION — 正在自动检测球场关键点...")
        print("=" * 60)

        detector = PitchKeypointDetector()

        # Use multi-frame if video path is provided
        if video_path and os.path.exists(video_path):
            print("  Sampling 5 frames for better coverage...")
            H, pixel_kpts, info = detector.calibrate_multiframe(
                video_path, n_frames=5, min_confidence=0.2, output_fifa=True,
            )
        else:
            H, pixel_kpts, info = detector.calibrate(frame, min_confidence=0.25, output_fifa=True)

        print(f"  Detected: {info['n_detected']}/32 keypoints")
        print(f"  Mean confidence: {info['quality']:.3f}")
        print(f"  Homography {'OK' if info['success'] else 'FAILED'}")
        print("=" * 60)

        if H is None:
            return None

        # Build FieldDetectionResult
        result = FieldDetectionResult(
            keypoints=pixel_kpts[:, :2] if pixel_kpts is not None else None,
            confidences=pixel_kpts[:, 2] if pixel_kpts is not None else None,
            homography=H,
            inverse_homography=np.linalg.inv(H),
            is_reliable=info["success"],
            num_valid_keypoints=info["n_detected"],
            calibration_mode="auto",
            world_keypoints=detector.get_world_keypoints()[:, :2],
            # Quality metrics
            mean_confidence=info.get("quality", 0.0),
            n_detected=info.get("n_detected", 0),
            n_used=info.get("n_used", 0),
            calibration_rmse=info.get("calibration_rmse", 0.0),
            calibration_rmse_pixels=info.get("calibration_rmse_pixels", 0.0),
            n_inliers_calib=info.get("n_inliers_calib", 0),
            voter_rmse=info.get("voter_rmse", 0.0),
            subset=info.get("subset", ""),
        )
        return result

    def _prompt_mode(self) -> str:
        """Ask user which calibration mode to use."""
        print("\n" + "=" * 60)
        print("  请选择标定模式（根据画面内容）：")
        print("=" * 60)
        print("  [0] 自动标定   — AI自动检测球场关键点，无需手动点击（推荐！）")
        print("  [1] 禁区标定   — 画面能看到球门和禁区线（4点）")
        print("  [2] 中圈标定   — 画面只显示中圈附近（仅需2点！）")
        print("  [3] 全场标定   — 画面能看到整个球场（4点）")
        print("  [4] 通用标定   — 画面中能看到任意矩形场地区域（4点）")
        print("=" * 60)

        while True:
            try:
                choice = input("  请输入 (0/1/2/3/4，默认=0): ").strip()
                if not choice or choice == "0":
                    return "auto"
                elif choice == "1":
                    return "penalty_area"
                elif choice == "2":
                    return "center_circle"
                elif choice == "3":
                    return "four_corner"
                elif choice == "4":
                    return "generic"
                else:
                    print("  无效输入，请输入 0、1、2、3 或 4")
            except EOFError:
                return "auto"


# ─────────────────────────────────────────────────────────────
# Field Detector (auto mode, kept for future Roboflow integration)
# ─────────────────────────────────────────────────────────────

class FieldDetector:
    """
    Field detector with optional Roboflow auto-detection.

    Usage (auto):
        detector = FieldDetector(api_key="your_key")
        result = detector.detect(frame)

    Usage (manual — prefer ManualFieldCalibrator directly):
        calibrator = ManualFieldCalibrator()
        result = calibrator.calibrate_interactive(frame, mode="penalty_area")
    """

    def __init__(self, api_key: Optional[str] = None, confidence: float = 0.5):
        self.api_key = api_key
        self.confidence = confidence
        self._last_result: Optional[FieldDetectionResult] = None

    def detect(self, frame: np.ndarray) -> FieldDetectionResult:
        """Auto-detect or return empty (caller handles manual fallback)."""
        if self.api_key:
            return self._detect_auto(frame)
        return FieldDetectionResult(is_reliable=False, calibration_mode="none")

    def _detect_auto(self, frame: np.ndarray) -> FieldDetectionResult:
        """Roboflow API field detection."""
        import base64
        import urllib.request
        import urllib.error

        try:
            _, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")

            model_id = "football-field-detection-f07vi/14"
            url = f"https://detect.roboflow.com/{model_id}?api_key={self.api_key}"
            payload = json.dumps({"image": {"type": "base64", "value": image_b64}}).encode()

            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode())

            keypoints = np.zeros((32, 2), dtype=np.float32)
            confidences = np.zeros(32, dtype=np.float32)

            if "predictions" in result and result["predictions"]:
                pred = result["predictions"][0]
                if "keypoints" in pred:
                    for i, kp in enumerate(pred["keypoints"][:32]):
                        if isinstance(kp, dict):
                            keypoints[i] = [kp.get("x", 0), kp.get("y", 0)]
                            confidences[i] = kp.get("confidence", 0)
                        elif isinstance(kp, (list, tuple)) and len(kp) >= 2:
                            keypoints[i] = [kp[0], kp[1]]
                            confidences[i] = kp[2] if len(kp) > 2 else 1.0

            valid_mask = confidences > self.confidence
            num_valid = int(valid_mask.sum())

            if num_valid < 4:
                return FieldDetectionResult(
                    keypoints=keypoints, confidences=confidences,
                    is_reliable=False, num_valid_keypoints=num_valid,
                    calibration_mode="auto",
                )

            src_pts = keypoints[valid_mask]
            valid_indices = np.where(valid_mask)[0]

            # Use world coords from config
            from .config import SoccerFieldConfiguration
            world_coords = SoccerFieldConfiguration.get_world_coordinates()
            dst_pts = world_coords[valid_indices]

            H, _ = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC,
                                      ransacReprojThreshold=3.0, maxIters=2000)

            return FieldDetectionResult(
                keypoints=keypoints, confidences=confidences,
                homography=H,
                inverse_homography=np.linalg.inv(H) if H is not None else None,
                is_reliable=H is not None and num_valid >= 8,
                num_valid_keypoints=num_valid,
                calibration_mode="auto",
            )

        except Exception as e:
            print(f"[FieldDetector] Auto detection failed: {e}")
            return FieldDetectionResult(is_reliable=False, calibration_mode="auto")

    def pixel_to_world(self, px: float, py: float) -> Optional[np.ndarray]:
        if self._last_result is None or self._last_result.homography is None:
            return None
        pt = np.array([[[px, py]]], dtype=np.float32)
        world = cv2.perspectiveTransform(pt, self._last_result.homography)
        return world[0, 0]

    def world_to_pixel(self, wx: float, wy: float) -> Optional[np.ndarray]:
        if self._last_result is None or self._last_result.inverse_homography is None:
            return None
        pt = np.array([[[wx, wy]]], dtype=np.float32)
        pixel = cv2.perspectiveTransform(pt, self._last_result.inverse_homography)
        return pixel[0, 0]
