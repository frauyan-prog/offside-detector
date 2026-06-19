#!/usr/bin/env python3
"""
Pitch Keypoint Detector — Local YOLOv8-pose model for automatic field calibration.

Uses martinjolif/yolo-football-pitch-detection model to detect 32 keypoints
on a football pitch, then computes the homography matrix for pixel↔world coordinate
transformation.

Keypoint mapping follows the official Roboflow SoccerPitchConfiguration (32 vertices):
  Labels 01-06:  Left touchline (near end)
  Labels 07-13:  Near penalty/goal area inner
  Labels 15-18:  Centre line + CC top/bottom
  Labels 20-26:  Far penalty/goal area inner
  Labels 27-32:  Right touchline (far end)
  Labels 14,19:  Centre circle left/right

World coordinates: x ∈ [0, 120] (length), y ∈ [0, 70] (width), centre at (60, 35)
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
from ultralytics import YOLO

from .config import SoccerFieldConfiguration

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Model path (download from: https://huggingface.co/martinjolif/yolo-football-pitch-detection)
# ──────────────────────────────────────────────────────────────────────────────
_MODEL_DIR = Path(__file__).parent / "models"
_MODEL_FILENAME = "yolo-football-pitch-detection.pt"


# ──────────────────────────────────────────────────────────────────────────────
# World coordinates for the 32 keypoints (matches official model definition)
# Based on FIFA standard pitch: width=68m, length=105m
# ──────────────────────────────────────────────────────────────────────────────
def _build_world_coords():
    """
    Build the 32-keypoint world coordinate array (N×3 with confidence dummy).

    Coordinate system matches Roboflow SoccerPitchConfiguration (the training data):
      - x = length direction, 0→120 m (near goal line to far goal line)
      - y = width direction,  0→70 m  (left touchline to right touchline)
      - origin (0,0) at TOP-LEFT touchline corner (Roboflow convention)
      - centre point at (60.0, 35.0)

    Pitch dimensions (Roboflow 120×70):
      Length: 120 m, Width: 70 m
      Penalty box: 41.0×20.15 m (width×depth)
      Goal box:    18.32×5.5 m
      Centre circle radius: 9.15 m
      Penalty spot: 11.0 m from goal line

    CRITICAL — Official Roboflow vertex order (from sports/configs/soccer.py):
      index 0-12:  labels 01-13  (near left touchline + near PA/GA inner)
      index 13-16: labels 15-18  (halfway line: top, CC-top, CC-bottom, bottom)
                              ^^ NOTE: label 14 is NOT here!
      index 17-23: labels 20-26  (far PA/GA inner)
      index 24-29: labels 27-32  (far right touchline)
      index 30-31: labels 14, 19 (centre circle LEFT/RIGHT — placed at END!)
                             ^^^^ These two are intentionally placed after all others
    """
    L, W = 120.0, 70.0  # length, width in metres
    mid_x, mid_y = L / 2, W / 2  # (60.0, 35.0)

    # Half-widths
    hw_pa = 41.0 / 2    # 20.5
    hw_ga = 18.32 / 2   # 9.16
    pa_depth = 20.15
    ga_depth = 5.5
    pen_dist = 11.0
    cc_r = 9.15

    # Y-axis helpers
    pa_top    = mid_y - hw_pa   # 14.50
    pa_bottom = mid_y + hw_pa   # 55.50
    ga_top    = mid_y - hw_ga   # 25.84
    ga_bottom = mid_y + hw_ga   # 44.16

    pts = np.zeros((32, 3), dtype=np.float32)

    # ── Near left touchline (x=0) — indices 0-5; labels 01-06 ──
    pts[0]  = [0.0,      0.0,     1.0]  # 01  Top-left corner
    pts[1]  = [0.0,      pa_top,  1.0]  # 02  L-touchline @ PA-top
    pts[2]  = [0.0,      ga_top,  1.0]  # 03  L-touchline @ GA-top
    pts[3]  = [0.0,      ga_bottom,1.0]  # 04  L-touchline @ GA-bottom
    pts[4]  = [0.0,      pa_bottom,1.0]  # 05  L-touchline @ PA-bottom
    pts[5]  = [0.0,      W,       1.0]  # 06  Bottom-left corner

    # ── Near penalty/goal area inner — indices 6-12; labels 07-13 ──
    pts[6]  = [ga_depth, ga_top,    1.0]  # 07  GA near-post top
    pts[7]  = [ga_depth, ga_bottom, 1.0]  # 08  GA near-post bottom
    pts[8]  = [pen_dist, mid_y,     1.0]  # 09  Penalty spot (near)
    pts[9]  = [pa_depth, pa_top,    1.0]  # 10  PA inner-top
    pts[10] = [pa_depth, ga_top,    1.0]  # 11  PA inner @ GA-top
    pts[11] = [pa_depth, ga_bottom, 1.0]  # 12  PA inner @ GA-bottom
    pts[12] = [pa_depth, pa_bottom, 1.0]  # 13  PA inner-bottom

    # ── Halfway line (x=60) — indices 13-16; labels 15-18 ──
    pts[13] = [mid_x,     0.0,         1.0]  # 15  Half-line top (touchline)
    pts[14] = [mid_x,     mid_y - cc_r, 1.0]  # 16  CC top
    pts[15] = [mid_x,     mid_y + cc_r, 1.0]  # 17  CC bottom
    pts[16] = [mid_x,     W,            1.0]  # 18  Half-line bottom (touchline)

    # ── Far penalty/goal area inner — indices 17-23; labels 20-26 ──
    pts[17] = [L - pa_depth, pa_top,    1.0]  # 20  Far PA inner-top
    pts[18] = [L - pa_depth, ga_top,    1.0]  # 21  Far PA inner @ GA-top
    pts[19] = [L - pa_depth, ga_bottom, 1.0]  # 22  Far PA inner @ GA-bottom
    pts[20] = [L - pa_depth, pa_bottom, 1.0]  # 23  Far PA inner-bottom
    pts[21] = [L - pen_dist, mid_y,     1.0]  # 24  Far penalty spot
    pts[22] = [L - ga_depth, ga_top,    1.0]  # 25  Far GA near-post top
    pts[23] = [L - ga_depth, ga_bottom, 1.0]  # 26  Far GA near-post bottom

    # ── Far right touchline (x=120) — indices 24-29; labels 27-32 ──
    pts[24] = [L,         0.0,     1.0]  # 27  Top-right corner
    pts[25] = [L,         pa_top,  1.0]  # 28  R-touchline @ far-PA top
    pts[26] = [L,         ga_top,  1.0]  # 29  R-touchline @ far-GA top
    pts[27] = [L,         ga_bottom,1.0]  # 30  R-touchline @ far-GA bottom
    pts[28] = [L,         pa_bottom,1.0]  # 31  R-touchline @ far-PA bottom
    pts[29] = [L,         W,       1.0]  # 32  Bottom-right corner

    # ── Centre circle left & right — indices 30-31; labels 14, 19 ──
    pts[30] = [mid_x - cc_r, mid_y, 1.0]  # 14  CC left
    pts[31] = [mid_x + cc_r, mid_y, 1.0]  # 19  CC right

    return pts


# Pre-computed world coordinates (x, y, confidence)
WORLD_KEYPOINTS = _build_world_coords()

# Conversion matrix: Roboflow 120×70 → FIFA 68×105 (x=width, y=length)
#   Roboflow: x_rob ∈ [0,120] along length, y_rob ∈ [0,70] along width
#   FIFA:     x_fifa ∈ [0,68]  along width,  y_fifa ∈ [0,105] along length
#   x_fifa = (70 - y_rob) * 68/70     (Roboflow top  → FIFA right)
#   y_fifa = (120 - x_rob) * 105/120   (Roboflow far  → FIFA top)
_CONV_ROBOFLOW_TO_FIFA = np.array([
    [0.0,       -68.0/70.0,  68.0],      # x_fifa = 68 - y_rob * 68/70
    [-105.0/120.0, 0.0,      105.0],      # y_fifa = 105 - x_rob * 105/120
    [0.0,        0.0,        1.0],
], dtype=np.float64)


class PitchKeypointDetector:
    """
    Local YOLOv8-pose based football pitch keypoint detector.

    Detects 32 field keypoints in a single frame, then computes the homography
    for pixel↔world coordinate mapping. No external API required.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = 0.3,
        image_size: int = 640,
    ):
        """
        Args:
            model_path: Path to the .pt model file. Defaults to local models dir.
            confidence: Minimum keypoint confidence threshold.
            image_size: Input image size for YOLO inference.
        """
        if model_path is None:
            model_path = str(_MODEL_DIR / _MODEL_FILENAME)

        self.model_path = model_path
        self.confidence = confidence
        self.image_size = image_size

        logger.info(f"Loading pitch keypoint model: {model_path}")
        self.model = YOLO(model_path, task="pose")
        logger.info("Model loaded successfully (YOLOv8-pose, 32 keypoints)")

    # ── Detection ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Run keypoint detection on a single frame.

        Args:
            frame: BGR image (H×W×3).

        Returns:
            np.ndarray of shape (32, 3) with (x, y, confidence) pixel coordinates,
            or None if no keypoints detected.
        """
        results = self.model(frame, imgsz=self.image_size, verbose=False)

        if len(results) == 0 or results[0].keypoints is None:
            return None

        kpts = results[0].keypoints
        if len(kpts) == 0 or kpts.data.shape[0] == 0:
            return None

        return kpts.data.cpu().numpy()[0]  # First (usually only) instance

    def detect_best(self, frame: np.ndarray, frame_id: int = 0) -> Tuple[Optional[np.ndarray], float]:
        """
        Detect with a quality score to help pick the best calibration frame.

        Returns:
            (keypoints_array, quality_score) — quality is mean confidence of visible points.
        """
        kpts = self.detect(frame)
        if kpts is None:
            return None, 0.0

        high_conf = kpts[:, 2] > 0.3
        if not np.any(high_conf):
            return kpts, 0.0

        quality = float(np.mean(kpts[high_conf, 2]))
        return kpts, quality

    # ── Homography Computation ───────────────────────────────────────────────

    def compute_homography(
        self,
        pixel_kpts: np.ndarray,
        min_points: int = 4,
        min_confidence: float = 0.3,
        method: str = "ransac",
    ) -> Optional[np.ndarray]:
        """
        Compute homography matrix from pixel keypoints to world coordinates.

        Args:
            pixel_kpts: (32, 3) array of (x, y, confidence) in pixel space.
            min_points: Minimum number of good points required.
            min_confidence: Minimum confidence to use a point.
            method: "ransac" (robust) or "all" (use all points).

        Returns:
            3×3 homography matrix (pixel → world), or None if failed.
        """
        # Filter valid points
        valid = pixel_kpts[:, 2] > min_confidence
        n_valid = int(np.sum(valid))

        logger.info(
            f"Keypoints with confidence > {min_confidence}: {n_valid}/32"
        )

        if n_valid < min_points:
            logger.warning(
                f"Only {n_valid} valid keypoints (need ≥ {min_points})"
            )
            return None

        # Extract valid pixel and world points
        src_pts = pixel_kpts[valid, :2].reshape(-1, 1, 2).astype(np.float32)
        dst_pts = WORLD_KEYPOINTS[valid, :2].reshape(-1, 1, 2).astype(np.float32)

        # NOTE: ransacReprojThreshold is in dst_pts units (metres here).
        # 5.0m is calibrated for the Roboflow YOLOv8-pose model's typical
        # keypoint accuracy (~1-3m per point). SoccerNet's 5px threshold
        # is used when src_pts/dst_pts are both in pixel space.
        _RANSAC_THRESHOLD = 10.0  # metres — generous, empirically necessary for this model

        if method == "ransac":
            H, mask = cv2.findHomography(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=_RANSAC_THRESHOLD,
                maxIters=2000,
            )
            if H is not None:
                inliers = int(np.sum(mask))
                logger.info(
                    f"Homography computed: {inliers}/{n_valid} inliers "
                    f"(RANSAC threshold={_RANSAC_THRESHOLD}m)"
                )
        else:
            H, mask = cv2.findHomography(src_pts, dst_pts, method=0)

        return H

    # ── Multi-Subset Voter (SoccerNet #1 inspired) ──────────────────────────

    def voter_homography(
        self,
        pixel_kpts: np.ndarray,
        min_confidence: float = 0.2,
        min_points: int = 4,
        ransac_threshold: float = 10.0,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], dict]:
        """
        Multi-subset voting for robust homography estimation.

        Inspired by SoccerNet Camera Calibration 2023 #1 solution.
        Computes homography on 4 subsets and picks the one with lowest
        reprojection RMSE on ALL valid points.

        Subsets:
          A — all points with conf > min_confidence
          B — high-confidence subset (conf > 0.5)
          C — iterative confidence threshold sweep
          D — RANSAC inliers re-fit with least squares (then tested on all)

        Args:
            pixel_kpts: (32, 3) keypoints (x, y, confidence).
            min_confidence: lowest confidence to consider (default 0.2).
            min_points: minimum points per subset.
            ransac_threshold: RANSAC reproj threshold in metres (dst_pts unit).
            min_points: minimum points per subset.

        Returns:
            (best_H, best_mask, metrics_dict)
            metrics: {rmse, n_inliers, n_total, subset_name, all_subsets_results}
            Returns (None, None, metrics) if all subsets fail.
        """
        # Gather valid keypoints (pixel ↔ world)
        if min_confidence is None:
            valid_mask = np.ones(len(pixel_kpts), dtype=bool)
        else:
            valid_mask = pixel_kpts[:, 2] > min_confidence
        confidences = pixel_kpts[:, 2]

        n_valid = int(np.sum(valid_mask))
        if n_valid < min_points:
            return None, None, {
                "rmse": float("inf"),
                "n_inliers": 0,
                "n_total": n_valid,
                "subset_name": "none",
                "error": "not_enough_points",
            }

        src_all = pixel_kpts[valid_mask, :2].reshape(-1, 1, 2).astype(np.float32)
        dst_all = WORLD_KEYPOINTS[valid_mask, :2].reshape(-1, 1, 2).astype(np.float32)

        candidates = []

        # ── Subset A: all valid points, RANSAC ──
        if len(src_all) >= min_points:
            H_a, mask_a = cv2.findHomography(
                src_all, dst_all,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_threshold,
                maxIters=2000,
            )
            if H_a is not None and mask_a.sum() >= min_points:
                inlier_src = src_all[mask_a.ravel() == 1]
                inlier_dst = dst_all[mask_a.ravel() == 1]
                # Re-fit with least squares on inliers
                H_a_ls, _ = cv2.findHomography(inlier_src, inlier_dst, method=0)
                rmse_a = self._compute_rmse(H_a_ls, src_all, dst_all)
                candidates.append((H_a_ls, mask_a, "subset_all", rmse_a, int(mask_a.sum())))

        # ── Subset B: high-confidence (conf > 0.5) ──
        hi_mask = (valid_mask) & (confidences > 0.5)
        n_hi = int(np.sum(hi_mask))
        if n_hi >= min_points:
            src_hi = pixel_kpts[hi_mask, :2].reshape(-1, 1, 2).astype(np.float32)
            dst_hi = WORLD_KEYPOINTS[hi_mask, :2].reshape(-1, 1, 2).astype(np.float32)
            H_b, mask_b = cv2.findHomography(
                src_hi, dst_hi,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_threshold,
                maxIters=2000,
            )
            if H_b is not None and mask_b.sum() >= min_points:
                inlier_src = src_hi[mask_b.ravel() == 1]
                inlier_dst = dst_hi[mask_b.ravel() == 1]
                H_b_ls, _ = cv2.findHomography(inlier_src, inlier_dst, method=0)
                rmse_b = self._compute_rmse(H_b_ls, src_all, dst_all)
                candidates.append((H_b_ls, mask_b, "subset_high_conf", rmse_b, int(mask_b.sum())))

        # ── Subset B2: medium-confidence (conf > 0.2) ──
        med_mask = (valid_mask) & (confidences > 0.2)
        n_med = int(np.sum(med_mask))
        if n_med >= min_points:
            src_med = pixel_kpts[med_mask, :2].reshape(-1, 1, 2).astype(np.float32)
            dst_med = WORLD_KEYPOINTS[med_mask, :2].reshape(-1, 1, 2).astype(np.float32)
            H_m, mask_m = cv2.findHomography(
                src_med, dst_med,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_threshold,
                maxIters=2000,
            )
            if H_m is not None and mask_m.sum() >= min_points:
                inlier_src = src_med[mask_m.ravel() == 1]
                inlier_dst = dst_med[mask_m.ravel() == 1]
                H_m_ls, _ = cv2.findHomography(inlier_src, inlier_dst, method=0)
                rmse_m = self._compute_rmse(H_m_ls, src_all, dst_all)
                candidates.append((H_m_ls, mask_m, "subset_med_conf", rmse_m, int(mask_m.sum())))

        # ── Subset C: iterative confidence thresholding ──
        # Start at min_confidence, try progressively higher thresholds
        for conf_th in [0.2, 0.3, 0.4]:
            iter_mask = (valid_mask) & (confidences > conf_th)
            n_iter = int(np.sum(iter_mask))
            if n_iter < 8:  # need enough points
                continue
            src_iter = pixel_kpts[iter_mask, :2].reshape(-1, 1, 2).astype(np.float32)
            dst_iter = WORLD_KEYPOINTS[iter_mask, :2].reshape(-1, 1, 2).astype(np.float32)
            H_c, mask_c = cv2.findHomography(
                src_iter, dst_iter,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_threshold,
                maxIters=2000,
            )
            if H_c is not None and mask_c.sum() >= min_points:
                inlier_src = src_iter[mask_c.ravel() == 1]
                inlier_dst = dst_iter[mask_c.ravel() == 1]
                H_c_ls, _ = cv2.findHomography(inlier_src, inlier_dst, method=0)
                rmse_c = self._compute_rmse(H_c_ls, src_all, dst_all)
                candidates.append((H_c_ls, mask_c, f"subset_conf_gt_{conf_th}", rmse_c, int(mask_c.sum())))
                break  # use the tightest that succeeds

        if not candidates:
            logger.debug(
                "voter_homography: ALL subsets failed. "
                "n_valid=%d, conf_range=[%.3f, %.3f]",
                n_valid, float(np.min(pixel_kpts[:, 2])), float(np.max(pixel_kpts[:, 2])),
            )
            return None, None, {
                "rmse": float("inf"),
                "n_inliers": 0,
                "n_total": n_valid,
                "subset_name": "none",
                "error": "all_subsets_failed",
            }

        # Pick candidate with lowest RMSE on all valid points
        candidates.sort(key=lambda c: c[3])
        best_H, best_mask, best_name, best_rmse, best_inliers = candidates[0]

        metrics = {
            "rmse": round(best_rmse, 3),
            "n_inliers": best_inliers,
            "n_total": n_valid,
            "subset_name": best_name,
            "all_candidates": [
                {"name": c[2], "rmse": round(c[3], 3), "inliers": c[4]}
                for c in candidates
            ],
        }

        logger.info(
            f"Voter: selected '{best_name}' — RMSE={best_rmse:.3f}m, "
            f"{best_inliers}/{n_valid} inliers "
            f"(vs {len(candidates)} candidates)"
        )

        return best_H, best_mask, metrics

    # ── Validation ──────────────────────────────────────────────────────────

    @staticmethod
    def _compute_rmse(
        H: np.ndarray,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
    ) -> float:
        """Compute reprojection RMSE (metres) for a homography."""
        if src_pts.shape[0] == 0:
            return float("inf")
        proj = cv2.perspectiveTransform(src_pts, H)
        errors = np.linalg.norm(proj - dst_pts, axis=2).ravel()
        return float(np.sqrt(np.mean(errors ** 2)))

    def validate_homography(self, H: np.ndarray) -> Tuple[bool, dict]:
        """
        Check geometric plausibility of homography.
        Returns (is_valid, details_dict).
        """
        details = {"valid": True, "warnings": [], "goal_line_angle_deg": 0.0, "is_convex": True}
        if H is None:
            details["valid"] = False
            details["warnings"].append("H is None")
            return False, details

        try:
            invH = np.linalg.inv(H)
            corners = np.array([[[0.0, 0.0]], [[68.0, 0.0]],
                                 [[68.0, 105.0]], [[0.0, 105.0]]], dtype=np.float32)
            pix = cv2.perspectiveTransform(corners, invH)
            tl, tr, br, bl = pix[0, 0], pix[1, 0], pix[2, 0], pix[3, 0]

            # Goal line parallelism
            v_top = tr - tl
            v_bot = br - bl
            ang_top = np.arctan2(v_top[1], v_top[0]) * 180 / np.pi
            ang_bot = np.arctan2(v_bot[1], v_bot[0]) * 180 / np.pi
            details["goal_line_angle_deg"] = round(abs(ang_top - ang_bot), 1)
            if abs(ang_top - ang_bot) > 5.0:
                details["warnings"].append(
                    f"Goal line angle={abs(ang_top - ang_bot):.1f} deg (threshold 5 deg)"
                )

            # Convexity
            quad = np.array([tl, tr, br, bl], dtype=np.float32)
            if not self._is_convex_quad(quad):
                details["is_convex"] = False
                details["warnings"].append("Projected corners not convex")

            # NaN check
            if np.any(np.isnan(pix)) or np.any(np.isinf(pix)):
                details["valid"] = False
                details["warnings"].append("NaN or Inf in projection")
        except Exception as e:
            details["valid"] = False
            details["warnings"].append(f"Validation error: {e}")
            return False, details

        details["valid"] = len(details["warnings"]) == 0
        return details["valid"], details

    @staticmethod
    def _is_convex_quad(pts: np.ndarray) -> bool:
        """Check if 4 points form a convex quadrilateral."""
        signs = []
        for i in range(4):
            p0, p1, p2 = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
            cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
            signs.append(1 if cross > 1e-6 else (-1 if cross < -1e-6 else 0))
        first = next((s for s in signs if s != 0), 0)
        if first == 0:
            return False
        return all(s == 0 or s == first for s in signs)

    # ── Vanishing-Point Constrained Homography ─────────────────────────────

    @staticmethod
    def _fit_line_ransac(
        pts: np.ndarray,
        max_iter: int = 200,
        inlier_thresh: float = 15.0,
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Fit a line to 2D points using RANSAC.

        Returns: (line_params, inlier_mask)
            line_params: (a, b, c) for ax + by + c = 0, or None if < 2 points
            inlier_mask: bool array marking inlier indices
        """
        n = len(pts)
        if n < 2:
            return None, np.zeros(n, dtype=bool)

        best_line = None
        best_inliers = n
        best_mask = np.zeros(n, dtype=bool)

        for _ in range(max_iter):
            i, j = np.random.choice(n, 2, replace=False)
            p1, p2 = pts[i], pts[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if abs(dx) < 0.5 and abs(dy) < 0.5:
                continue
            # Line: dy*x - dx*y + (dx*p1[1] - dy*p1[0]) = 0
            a = dy
            b = -dx
            c = dx * p1[1] - dy * p1[0]
            norm = np.sqrt(a * a + b * b)
            a /= norm
            b /= norm
            c /= norm

            dists = np.abs(a * pts[:, 0] + b * pts[:, 1] + c)
            mask = dists < inlier_thresh
            n_inliers = int(np.sum(mask))
            if n_inliers < best_inliers:
                best_inliers = n_inliers
                best_line = (a, b, c)
                best_mask = mask

        if best_line is None or best_inliers < 2:
            # Fallback: least-squares through all points
            if n >= 2:
                mean = np.mean(pts, axis=0)
                cov = np.cov(pts.T)
                if cov[0, 0] < 1e-6 and cov[1, 1] < 1e-6:
                    return None, np.zeros(n, dtype=bool)
                _, eigvecs = np.linalg.eigh(cov)
                a, b = eigvecs[0, 1], eigvecs[1, 1]
                c = -(a * mean[0] + b * mean[1])
                return (float(a), float(b), float(c)), np.ones(n, dtype=bool)
            return None, np.zeros(n, dtype=bool)

        return best_line, best_mask

    @staticmethod
    def _line_intersection(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
        """Compute intersection of two lines (a1,b1,c1) and (a2,b2,c2)."""
        a1, b1, c1 = float(l1[0]), float(l1[1]), float(l1[2])
        a2, b2, c2 = float(l2[0]), float(l2[1]), float(l2[2])
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-10:
            return np.array([np.nan, np.nan])
        x = (b1 * c2 - b2 * c1) / det
        y = (a2 * c1 - a1 * c2) / det
        return np.array([x, y])

    def compute_homography_vp(
        self,
        pixel_kpts: np.ndarray,
        min_confidence: float = 0.2,
    ) -> Optional[np.ndarray]:
        """
        Compute homography using vanishing-point constraint from touchlines.
        Uses ALL 32 keypoints (including low-confidence near-field) weighted by
        confidence, with the VP as a hard constraint.
        """
        # ── 1. VP from ALL touchline keypoints ──
        left_indices = np.array([0, 1, 2, 3, 4, 5])
        right_indices = np.array([24, 25, 26, 27, 28, 29])
        left_xy = pixel_kpts[left_indices, :2].astype(np.float64)
        right_xy = pixel_kpts[right_indices, :2].astype(np.float64)

        l_line, _ = self._fit_line_ransac(left_xy)
        r_line, _ = self._fit_line_ransac(right_xy)
        if l_line is None or r_line is None:
            print("__VP_DEBUG__: line fitting failed L=%s R=%s" % (l_line, r_line))
            return None
        vp = self._line_intersection(np.array(l_line), np.array(r_line))
        if np.any(np.isnan(vp)):
            print("__VP_DEBUG__: VP is NaN (lines nearly parallel)")
            return None
        vp_h = np.array([vp[0], vp[1], 1.0], dtype=np.float64)

        # ── 2. Weighted DLT with VP as soft constraint ──
        # Standard DLT: for each (wx,wy)→(px,py):
        #   [0, 0, 0, -wx, -wy, -1,  py*wx,  py*wy,  py] [h11..h33]ᵀ = 0
        #   [wx,wy, 1,  0,  0,  0, -px*wx, -px*wy, -px] [h11..h33]ᵀ = 0
        #
        # VP constraint: h2 ∝ VP → h2 × VP = 0 (2 independent equations)
        #   h22 - vpy*h32 = 0
        #   h12 - vpx*h32 = 0

        weights = np.maximum(pixel_kpts[:, 2], 0.05)  # cap at 0.05 for near-field
        src_xy = pixel_kpts[:, :2].astype(np.float64)
        dst_xy = WORLD_KEYPOINTS[:, :2].astype(np.float64)

        vpx, vpy = float(vp[0]), float(vp[1])
        A_rows = []
        w_rows = []

        for i in range(32):
            w = float(weights[i])
            if w < 0.01:
                continue
            wx, wy = dst_xy[i, 0], dst_xy[i, 1]
            px, py = src_xy[i, 0], src_xy[i, 1]

            # Eq1: -wx*h21 - wy*h22 - h23 + py*wx*h31 + py*wy*h32 + py*h33 = 0
            A_rows.append([0, 0, 0, -wx, -wy, -1,  py*wx,  py*wy,  py])
            w_rows.append(w)

            # Eq2: wx*h11 + wy*h12 + h13 - px*wx*h31 - px*wy*h32 - px*h33 = 0
            A_rows.append([wx, wy, 1,  0,   0,   0, -px*wx, -px*wy, -px])
            w_rows.append(w)

        # ── Add VP constraint (weighted strongly) ──
        vp_weight = 10.0  # strong soft constraint
        # h22 - vpy*h32 = 0  →  row: [0, 0, 0, 0, 1, 0, 0, -vpy, 0]
        A_rows.append([0, 0, 0, 0, 1, 0, 0, -vpy, 0])
        w_rows.append(vp_weight)
        # h12 - vpx*h32 = 0  →  row: [0, 1, 0, 0, 0, 0, -vpx, 0, 0]
        A_rows.append([0, 1, 0, 0, 0, 0, -vpx, 0, 0])
        w_rows.append(vp_weight)

        A = np.array(A_rows, dtype=np.float64)
        wv = np.array(w_rows, dtype=np.float64)
        Aw = A * wv[:, np.newaxis]

        # SVD: solve Aw * h = 0 subject to |h| = 1
        _, _, Vt = np.linalg.svd(Aw, full_matrices=False)
        h = Vt[-1, :]  # last row of Vt = smallest singular vector

        # ── 3. Build H from solution vector ──
        H = h.reshape(3, 3).astype(np.float64)
        if abs(H[2, 2]) > 1e-10:
            H /= H[2, 2]

        return H

    # ── Field Line Detection & VP from image ─────────────────────────────────

    def detect_field_lines(
        self,
        frame: np.ndarray,
        canny_low: int = 30,
        canny_high: int = 90,
        hough_threshold: int = 50,
        min_line_length: int = 40,
        max_line_gap: int = 20,
    ) -> List[np.ndarray]:
        """
        Detect horizontal-ish field lines from the broadcast image.

        Strategy: isolate the green field → find white lines on it using
        value channel edges, and also use saturation channel for grass
        striping edges. Apply Canny on both, combine, then HoughLinesP.

        This is inspired by PnLCalib's line detection branch and the
        single-camera offside paper (arxiv 2502.16030).
        """
        if frame is None or frame.size == 0:
            return []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create green mask to isolate the field
        lower_green = np.array([35, 30, 30])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Dilate the mask to include white lines (which sit ON green)
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.dilate(green_mask, kernel, iterations=2)

        # White line detection: lines are bright on green background
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]

        edges_s = cv2.Canny(s_channel, canny_low, canny_high)
        edges_v = cv2.Canny(v_channel, canny_low, canny_high)
        edges = cv2.bitwise_or(edges_s, edges_v)

        # Only keep edges within the field region
        edges = cv2.bitwise_and(edges, edges, mask=green_mask)

        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180, threshold=hough_threshold,
            minLineLength=min_line_length, maxLineGap=max_line_gap,
        )

        if lines is None:
            return []

        filtered = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < 1e-6:
                angle_deg = 90.0
            else:
                angle_deg = abs(np.degrees(np.arctan2(dy, dx)))
            # Accept wider angle range — field lines can be at various
            # apparent angles depending on camera position
            if 10 < angle_deg < 70:
                filtered.append(np.array([x1, y1, x2, y2], dtype=np.float64))

        return filtered

    def find_vp_from_lines(
        self,
        lines: List[np.ndarray],
        ransac_threshold: float = 100.0,
        ransac_iters: int = 1000,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Find vanishing point from detected field lines by clustering by angle
        then using RANSAC within the dominant cluster.

        Returns:
            (vp_xy, inlier_ratio) or (None, 0.0) on failure
        """
        if len(lines) < 2:
            return None, 0.0

        n = len(lines)

        # ── Group lines by angle ──
        angles = []
        for ln in lines:
            dx = ln[2] - ln[0]
            dy = ln[3] - ln[1]
            ang = abs(np.degrees(np.arctan2(dy, dx))) if abs(dx) > 1e-6 else 90.0
            angles.append(ang)
        angles = np.array(angles)

        # Simple clustering: find the largest group within ±8° band
        from collections import Counter
        angle_bins = np.round(angles / 5.0) * 5.0
        best_bin = Counter(angle_bins).most_common(1)[0][0]
        cluster_mask = np.abs(angles - best_bin) < 10.0
        cluster_lines = [lines[i] for i in range(n) if cluster_mask[i]]

        if len(cluster_lines) < 2:
            # Fallback: use all lines
            cluster_lines = lines
            if len(cluster_lines) < 2:
                return None, 0.0

        nc = len(cluster_lines)

        # If only 2 lines, just return their intersection
        if nc == 2:
            vp = self._line_intersection_from_segments(
                cluster_lines[0], cluster_lines[1],
            )
            if np.any(np.isnan(vp)):
                return None, 0.0
            return vp, 1.0

        # ── RANSAC on intersections ──
        intersections = []
        for i in range(nc):
            for j in range(i + 1, nc):
                pt = self._line_intersection_from_segments(
                    cluster_lines[i], cluster_lines[j],
                )
                if not np.any(np.isnan(pt)):
                    intersections.append(pt)

        if len(intersections) < 2:
            return None, 0.0

        inters = np.array(intersections)
        best_vp = None
        best_inliers = 0

        for _ in range(min(ransac_iters, len(intersections) * 3)):
            idx = np.random.randint(0, len(intersections))
            candidate = inters[idx]
            dists = np.linalg.norm(inters - candidate, axis=1)
            n_inliers = int(np.sum(dists < ransac_threshold))
            if n_inliers > best_inliers:
                best_inliers = n_inliers
                best_vp = candidate

        if best_vp is None:
            return None, 0.0

        # Refine: average all inlier intersections
        dists = np.linalg.norm(inters - best_vp, axis=1)
        inlier_mask = dists < ransac_threshold
        refined_vp = np.mean(inters[inlier_mask], axis=0)

        return refined_vp, float(np.sum(inlier_mask)) / len(intersections)

    @staticmethod
    def _line_intersection_from_segments(
        seg1: np.ndarray, seg2: np.ndarray,
    ) -> np.ndarray:
        """Compute intersection point of two line segments (extended to lines)."""
        x1, y1, x2, y2 = seg1
        x3, y3, x4, y4 = seg2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return np.array([np.nan, np.nan])
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return np.array([px, py])

    def compute_homography_line_vp(
        self,
        pixel_kpts: np.ndarray,
        frame: np.ndarray,
        min_confidence: float = 0.2,
    ) -> Optional[np.ndarray]:
        """
        Compute homography using vanishing point from actual field lines
        (Canny+Hough detected from image) combined with YOLO keypoints.

        This combines the best of both approaches:
        - YOLO keypoints: accurate positions for visible points
        - Image lines: provide VP constraint that compensates for missing
          near-field keypoints

        The VP constraint is:
          For any two world points at the same depth (same world-x) but
          different lateral positions (different world-y), the line through
          their image projections should pass through the VP.
        """
        # ── 1. Detect field lines and compute VP ──
        field_lines = self.detect_field_lines(frame)
        vp, vp_quality = self.find_vp_from_lines(field_lines)

        if vp is None:
            # Fallback: try the touchline-based VP
            left_indices = np.array([0, 1, 2, 3, 4, 5])
            right_indices = np.array([24, 25, 26, 27, 28, 29])
            left_xy = pixel_kpts[left_indices, :2].astype(np.float64)
            right_xy = pixel_kpts[right_indices, :2].astype(np.float64)
            l_line, _ = self._fit_line_ransac(left_xy)
            r_line, _ = self._fit_line_ransac(right_xy)
            if l_line is None or r_line is None:
                return None
            vp = self._line_intersection(np.array(l_line), np.array(r_line))
            if np.any(np.isnan(vp)):
                return None

        print(f"__LINE_VP__: VP=({vp[0]:.1f}, {vp[1]:.1f}) "
              f"quality={vp_quality:.2f} n_lines={len(field_lines)}")

        # ── 2. Generate VP-constrained virtual near-field keypoints ──
        # Strategy: for each successfully detected keypoint at world-x > 0,
        # use the VP to extrapolate where the same y-coordinate would project
        # at x=0 (the goal line / near edge).
        #
        # Given: world point (x_w, y_w) → image point (px, py)
        #        VP at (vpx, vpy)
        #        Desired: image point for world (0, y_w)
        #
        # The line CONNECTING these two image points passes through the VP.
        # So the target (px0, py0) lies on the ray from VP through (px, py).
        #
        # For the goal line (x_w=0), the point is farther from VP than (px, py).
        # The extrapolation factor f = (some func of depth) is approximated
        # using the known keypoint at the CENTER LINE (x_w=60) and its
        # expected distance ratio.

        vpx, vpy = float(vp[0]), float(vp[1])
        weights = np.maximum(pixel_kpts[:, 2], 0.05)
        src_xy = pixel_kpts[:, :2].astype(np.float64)
        dst_xy = WORLD_KEYPOINTS[:, :2].astype(np.float64)

        # Build standard DLT rows first
        A_rows = []
        w_rows = []
        for i in range(32):
            w = float(weights[i])
            if w < 0.01:
                continue
            wx, wy = dst_xy[i, 0], dst_xy[i, 1]
            px, py = src_xy[i, 0], src_xy[i, 1]

            A_rows.append([0, 0, 0, -wx, -wy, -1,  py * wx,  py * wy,  py])
            w_rows.append(w)
            A_rows.append([wx, wy, 1,  0,   0,   0, -px * wx, -px * wy, -px])
            w_rows.append(w)

        # ── 3. VP constraint: h2 × VP = 0 ──
        # The second column of H (horizontal direction in world) maps to VP.
        # h2 = [h12, h22, h32]^T must satisfy:
        #   h12 / h32 = vpx   →   h12 - vpx * h32 = 0
        #   h22 / h32 = vpy   →   h22 - vpy * h32 = 0
        vp_weight = max(10.0, 30.0 * vp_quality)  # weight by line quality
        A_rows.append([0, 1, 0, 0, 0, 0, -vpx, 0, 0])
        w_rows.append(vp_weight)
        A_rows.append([0, 0, 0, 0, 1, 0, 0, -vpy, 0])
        w_rows.append(vp_weight)

        # ── 4. Add virtual near-field keypoints from line extrapolation ──
        # For selected well-detected keypoints (high confidence, moderate depth),
        # extrapolate their position to the near-field goal line.
        #
        # These "virtual" points are generated by extending the ray from VP
        # through the keypoint's image position farther away from the VP.
        # The distance is proportional to the world x-coordinate ratio.
        #
        # Only use keypoints with x_w >= 60 (center line+) since they have
        # the most reliable detection and give the best extrapolation.

        n_virtual = 0
        for i in range(32):
            if weights[i] < 0.3:  # only high-confidence keypoints
                continue
            x_w = dst_xy[i, 0]
            if x_w < 50:  # center region or farther
                continue
            px, py = src_xy[i, 0], src_xy[i, 1]
            y_w = dst_xy[i, 1]

            # Extrapolate from (px,py) away from VP toward goal line
            # The VP is roughly in the center-top of the image for broadcast
            # view. Goal line is at the bottom. So we extend the ray DOWN.
            # Direction from VP to keypoint
            dx = px - vpx
            dy = py - vpy
            dist_kp_to_vp = np.sqrt(dx * dx + dy * dy)
            if dist_kp_to_vp < 10:
                continue

            # Extrapolation factor:
            # For a keypoint at world-x=60 (center line), the goal line at x=0
            # is "behind" it. In perspective, further world-x means closer to VP.
            # The ray direction from VP outward: VP→near_field→midfield→far_field
            # So goal line (x=0) is FARTHER from VP than center line (x=60).
            # Extrapolation factor ≈ 1 + (x_w / field_length) * scale
            # scale calibrated so center line maps correctly
            ext_factor = 1.0 + (x_w / 120.0) * 1.5

            virt_px = vpx + dx * ext_factor
            virt_py = vpy + dy * ext_factor

            # World coordinates for the virtual point: same y_w, but x_w = 0
            virt_wx = 0.0
            virt_wy = y_w

            # Add to DLT with moderate weight
            vw = 2.0
            A_rows.append([0, 0, 0, -virt_wx, -virt_wy, -1,
                           virt_py * virt_wx, virt_py * virt_wy, virt_py])
            w_rows.append(vw)
            A_rows.append([virt_wx, virt_wy, 1, 0, 0, 0,
                           -virt_px * virt_wx, -virt_px * virt_wy, -virt_px])
            w_rows.append(vw)
            n_virtual += 1

        print(f"__LINE_VP__: {n_virtual} virtual near-field points generated")

        # ── 5. Solve weighted DLT ──
        A = np.array(A_rows, dtype=np.float64)
        wv = np.array(w_rows, dtype=np.float64)
        Aw = A * wv[:, np.newaxis]

        _, _, Vt = np.linalg.svd(Aw, full_matrices=False)
        h = Vt[-1, :]
        H = h.reshape(3, 3).astype(np.float64)
        if abs(H[2, 2]) > 1e-10:
            H /= H[2, 2]

        return H

    def calibration_quality(
        self,
        pixel_kpts: np.ndarray,
        H: np.ndarray,
        min_confidence: float = 0.2,
    ) -> dict:
        """
        Compute detailed quality metrics for a calibration result.

        Returns:
            {rmse, rmse_pixels, n_inliers, n_total, n_detected,
             per_point_errors: [(index, error_m, error_px), ...]}
        """
        valid = pixel_kpts[:, 2] > min_confidence
        n_valid = int(np.sum(valid))

        if n_valid == 0 or H is None:
            return {"rmse": float("inf"), "rmse_pixels": float("inf"),
                    "n_inliers": 0, "n_total": 0, "n_detected": 0,
                    "per_point_errors": []}

        src_pts = pixel_kpts[valid, :2].reshape(-1, 1, 2).astype(np.float32)
        dst_pts = WORLD_KEYPOINTS[valid, :2].reshape(-1, 1, 2).astype(np.float32)

        # RMSE in world coords (metres)
        proj = cv2.perspectiveTransform(src_pts, H)
        errors_m = np.linalg.norm(proj - dst_pts, axis=2).ravel()
        rmse_m = float(np.sqrt(np.mean(errors_m ** 2)))

        # RMSE in pixel space (for human-friendly interpretation)
        invH = np.linalg.inv(H)
        projected_pix = cv2.perspectiveTransform(dst_pts, invH)
        errors_px = np.linalg.norm(projected_pix - src_pts, axis=2).ravel()
        rmse_px = float(np.sqrt(np.mean(errors_px ** 2)))

        # Inlier count (within RANSAC 3.0m threshold)
        n_inliers = int(np.sum(errors_m < 3.0))

        # Per-point errors
        valid_indices = np.where(valid)[0]
        per_point = []
        for i, idx in enumerate(valid_indices):
            per_point.append({
                "index": int(idx),
                "error_m": round(float(errors_m[i]), 2),
                "error_px": round(float(errors_px[i]), 1),
                "confidence": round(float(pixel_kpts[idx, 2]), 3),
            })
        per_point.sort(key=lambda x: x["error_m"], reverse=True)  # worst first

        return {
            "rmse": round(rmse_m, 3),
            "rmse_pixels": round(rmse_px, 1),
            "n_inliers": n_inliers,
            "n_total": n_valid,
            "n_detected": int(np.sum(pixel_kpts[:, 2] > 0.0)),
            "per_point_errors": per_point[:10],  # top 10 worst
        }

    # ── Visualisation ───────────────────────────────────────────────────────

    @staticmethod
    def draw_pitch_overlay(
        frame: np.ndarray,
        H: np.ndarray,
        alpha: float = 0.4,
        color: tuple = (0, 255, 0),
        line_width: int = 2,
    ) -> np.ndarray:
        """
        Draw FIFA pitch lines projected onto the frame via homography H.

        Useful for visually verifying calibration quality — if H is correct,
        the drawn lines should align with the actual pitch markings.

        Args:
            frame: BGR image.
            H: Homography (pixel → FIFA 68×105). Inverse will be used to project.
            alpha: Opacity of overlay (0–1).
            color: BGR line colour.
            line_width: Line thickness in pixels.

        Returns:
            Overlay image (same size as frame).
        """
        if H is None:
            return frame.copy()

        invH = np.linalg.inv(H)
        overlay = frame.copy()
        h, w = frame.shape[:2]

        def _proj(wx, wy):
            """Project a single FIFA world point to pixel coords."""
            pt = np.array([[[wx, wy]]], dtype=np.float32)
            pix = cv2.perspectiveTransform(pt, invH)
            return int(round(pix[0, 0, 0])), int(round(pix[0, 0, 1]))

        def _draw_line(x1, y1, x2, y2, c=None):
            cv2.line(overlay, (x1, y1), (x2, y2), c or color, line_width, cv2.LINE_AA)

        # Clamp to frame
        def _clamp(x, y):
            return max(0, min(x, w - 1)), max(0, min(y, h - 1))

        # ── Boundary ──
        tl = _clamp(*_proj(0, 0))
        tr = _clamp(*_proj(68, 0))
        br = _clamp(*_proj(68, 105))
        bl = _clamp(*_proj(0, 105))
        for a, b in [(tl, tr), (tr, br), (br, bl), (bl, tl)]:
            _draw_line(*a, *b, (0, 255, 0))

        # ── Halfway line ──
        hm_t = _clamp(*_proj(34, 0))
        hm_b = _clamp(*_proj(34, 105))
        _draw_line(*hm_t, *hm_b, (0, 200, 200))

        # ── Centre circle (approximated) ──
        cc_pts = []
        for angle_deg in np.linspace(0, 360, 24):
            rad = np.radians(angle_deg)
            wx = 34.0 + 9.15 * np.cos(rad)
            wy = 52.5 + 9.15 * np.sin(rad)
            cc_pts.append(_clamp(*_proj(wx, wy)))
        for i in range(len(cc_pts)):
            j = (i + 1) % len(cc_pts)
            _draw_line(*cc_pts[i], *cc_pts[j], (200, 200, 0))

        # ── Penalty areas (both ends) ──
        # Near end (y=0): x=[13.84, 54.16], y=[0, 16.5]
        pa_near = [
            _clamp(*_proj(13.84, 0)), _clamp(*_proj(54.16, 0)),
            _clamp(*_proj(54.16, 16.5)), _clamp(*_proj(13.84, 16.5)),
        ]
        for i in range(4):
            _draw_line(*pa_near[i], *pa_near[(i + 1) % 4], (0, 200, 200))

        # Far end (y=105): x=[13.84, 54.16], y=[88.5, 105]
        pa_far = [
            _clamp(*_proj(13.84, 88.5)), _clamp(*_proj(54.16, 88.5)),
            _clamp(*_proj(54.16, 105)), _clamp(*_proj(13.84, 105)),
        ]
        for i in range(4):
            _draw_line(*pa_far[i], *pa_far[(i + 1) % 4], (0, 200, 200))

        # ── Goal areas (both ends) ──
        # Near: x=[29.34, 38.66], y=[0, 5.5]
        ga_near = [
            _clamp(*_proj(29.34, 0)), _clamp(*_proj(38.66, 0)),
            _clamp(*_proj(38.66, 5.5)), _clamp(*_proj(29.34, 5.5)),
        ]
        for i in range(4):
            _draw_line(*ga_near[i], *ga_near[(i + 1) % 4], (200, 0, 200))

        # Far: x=[29.34, 38.66], y=[99.5, 105]
        ga_far = [
            _clamp(*_proj(29.34, 99.5)), _clamp(*_proj(38.66, 99.5)),
            _clamp(*_proj(38.66, 105)), _clamp(*_proj(29.34, 105)),
        ]
        for i in range(4):
            _draw_line(*ga_far[i], *ga_far[(i + 1) % 4], (200, 0, 200))

        # ── Penalty spots ──
        for py in [11.0, 94.0]:
            ps = _clamp(*_proj(34.0, py))
            cv2.circle(overlay, ps, 4, (0, 0, 255), -1, cv2.LINE_AA)

        # Blend
        blended = cv2.addWeighted(frame, 1.0 - alpha, overlay, alpha, 0)
        return blended

    # ── Full Auto-Calibration ────────────────────────────────────────────────

    def calibrate(
        self,
        frame: np.ndarray,
        min_confidence: float = 0.3,
        output_fifa: bool = True,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], dict]:
        """
        Full auto-calibration: detect keypoints + compute homography.

        Args:
            frame: BGR image (H×W×3).
            min_confidence: Minimum keypoint confidence to use.
            output_fifa: If True, convert homography output to FIFA 68×105 (x=width, y=length).
                        If False, output in Roboflow 120×70 (x=length, y=width).

        Returns:
            (H_matrix, pixel_keypoints, info_dict)
            info_dict contains: n_detected, n_used, inliers, quality
        """
        info = {
            "success": False,
            "n_detected": 0,
            "n_used": 0,
            "inliers": 0,
            "quality": 0.0,
            "coord_system": "fifa_68x105" if output_fifa else "roboflow_120x70",
        }

        # Step 1: Detect keypoints
        pixel_kpts = self.detect(frame)
        if pixel_kpts is None:
            return None, None, info

        info["n_detected"] = int(np.sum(pixel_kpts[:, 2] > 0.0))
        info["quality"] = float(np.mean(pixel_kpts[:, 2]))

        # Step 2: Compute homography (pixel → Roboflow 120×70)
        H = self.compute_homography(
            pixel_kpts,
            min_confidence=min_confidence,
            method="ransac",
        )

        if H is not None:
            info["success"] = True
            info["n_used"] = int(np.sum(pixel_kpts[:, 2] > min_confidence))
            info["inliers"] = info["n_used"]

            # Step 3: Convert to FIFA coordinates if requested
            if output_fifa:
                H = _CONV_ROBOFLOW_TO_FIFA @ H

            # Step 4: Validate homography quality
            valid_flag, valid_detail = self.validate_homography(H)
            info["validation"] = valid_detail
            if not valid_flag:
                logger.warning(
                    "Homography validation: %s", valid_detail.get("warnings", [])
                )

            # Step 5: Compute calibration quality metrics
            quality_metrics = self.calibration_quality(
                pixel_kpts, H, min_confidence
            )
            info["calibration_rmse"] = quality_metrics.get("rmse", float("inf"))
            info["calibration_rmse_pixels"] = quality_metrics.get("rmse_pixels", float("inf"))
            info["n_inliers_calib"] = quality_metrics.get("n_inliers", 0)

        return H, pixel_kpts, info

    def calibrate_multiframe(
        self,
        video_path: str,
        n_frames: int = 5,
        min_confidence: float = 0.3,
        output_fifa: bool = True,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], dict]:
        """
        Auto-calibration using multiple video frames for improved coverage.

        Each keypoint index picks the highest-confidence detection across frames,
        improving detection rate especially for partially visible pitch regions.

        Args:
            video_path: Path to video file.
            n_frames: Number of frames to sample (evenly spaced).
            min_confidence: Minimum keypoint confidence for homography.
            output_fifa: Output in FIFA coordinates.

        Returns:
            (H_matrix, fused_pixel_keypoints, info_dict)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None, {"success": False, "n_detected": 0}

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < n_frames:
            n_frames = max(1, total)

        # Sample frame indices evenly
        indices = np.linspace(0, total - 1, n_frames, dtype=int)

        # Accumulate best keypoints (index → (x, y, conf))
        best_kpts = np.zeros((32, 3), dtype=np.float32)

        for fidx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret:
                continue

            kpts = self.detect(frame)
            if kpts is None:
                continue

            # Merge: keep highest-confidence detection for each keypoint
            for idx in range(32):
                if kpts[idx, 2] > best_kpts[idx, 2]:
                    best_kpts[idx] = kpts[idx]

        cap.release()

        info = {
            "success": False,
            "n_detected": 0,
            "n_used": 0,
            "inliers": 0,
            "quality": 0.0,
            "n_frames_sampled": n_frames,
            "coord_system": "fifa_68x105" if output_fifa else "roboflow_120x70",
        }

        n_detected = int(np.sum(best_kpts[:, 2] > 0.0))
        info["n_detected"] = n_detected
        info["quality"] = float(np.mean(best_kpts[:, 2]))

        if n_detected < 4:
            return None, best_kpts, info

        # Compute homography — use multi-subset voter for robustness
        H, voter_mask, voter_metrics = self.voter_homography(
            best_kpts,
            min_confidence=min_confidence,
            ransac_threshold=10.0,
        )
        if H is not None:
            info["success"] = True
            info["n_used"] = voter_metrics.get("n_total", 0)
            info["inliers"] = voter_metrics.get("n_inliers", 0)
            info["subset"] = voter_metrics.get("subset_name", "voter")
            info["voter_rmse"] = voter_metrics.get("rmse", float("inf"))

            if output_fifa:
                H = _CONV_ROBOFLOW_TO_FIFA @ H

            # Step 3: Validate homography quality
            valid_flag, valid_detail = self.validate_homography(H)
            info["validation"] = valid_detail
            if not valid_flag:
                logger.warning(
                    "Homography validation: %s", valid_detail.get("warnings", [])
                )
                orig_angle = valid_detail.get("goal_line_angle_deg", 999)

                # ── Fallback 1: voter with ALL 32 points + loose RANSAC ──
                print("__FALLBACK1_CALLED__")
                H_fb1, fb1_mask, fb1_metrics = self.voter_homography(
                    best_kpts,
                    min_confidence=None,
                    ransac_threshold=10.0,
                )
                print("__FALLBACK1_RESULT__ H_fb1_is_None=", H_fb1 is None, "metrics=", fb1_metrics)
                if H_fb1 is not None:
                    if output_fifa:
                        H_fb1 = _CONV_ROBOFLOW_TO_FIFA @ H_fb1
                    fb1_valid, fb1_detail = self.validate_homography(H_fb1)
                    fb1_angle = fb1_detail.get("goal_line_angle_deg", 999)
                    if fb1_valid or fb1_angle < orig_angle:
                        H = H_fb1
                        info["validation"] = fb1_detail
                        info["subset"] = "voter_loose"
                        info["voter_rmse"] = fb1_metrics.get("rmse", float("inf"))
                        logger.info(
                            "Fallback1 voter(all+tightRANSAC): "
                            "%.1f°→%.1f°", orig_angle, fb1_angle,
                        )
                        orig_angle = fb1_angle

                # ── Fallback 2: VP-constrained homography ──
                if orig_angle > 5.0:
                    print("__FALLBACK2_CALLED__ orig_angle=%.1f" % orig_angle)
                    vp_H = self.compute_homography_vp(best_kpts, min_confidence)
                    print("__FALLBACK2_RESULT__ vp_H_is_None=%s" % (vp_H is None))
                    if vp_H is not None:
                        if output_fifa:
                            vp_H = _CONV_ROBOFLOW_TO_FIFA @ vp_H
                        vp_valid, vp_detail = self.validate_homography(vp_H)
                        vp_angle = vp_detail.get("goal_line_angle_deg", 999)
                        if vp_valid or vp_angle < orig_angle:
                            H = vp_H
                            info["validation"] = vp_detail
                            info["subset"] = "vp_constrained"
                            logger.info(
                                "Fallback2 VP-constrained: "
                                "%.1f°→%.1f°", orig_angle, vp_angle,
                            )
                            orig_angle = vp_angle

                # ── Fallback 3: Line-detected VP from image + virtual near-field ──
                if orig_angle > 3.0:
                    print("__FALLBACK3_CALLED__ orig_angle=%.1f" % orig_angle)
                    # Re-read first sampled frame for line detection
                    cap3 = cv2.VideoCapture(video_path)
                    cap3.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
                    ret3, frame3 = cap3.read()
                    cap3.release()
                    if ret3:
                        line_vp_H = self.compute_homography_line_vp(
                            best_kpts, frame3, min_confidence,
                        )
                        print("__FALLBACK3_RESULT__ line_vp_H_is_None=%s" % (line_vp_H is None))
                        if line_vp_H is not None:
                            if output_fifa:
                                line_vp_H = _CONV_ROBOFLOW_TO_FIFA @ line_vp_H
                            lv_valid, lv_detail = self.validate_homography(line_vp_H)
                            lv_angle = lv_detail.get("goal_line_angle_deg", 999)
                            if lv_valid or lv_angle < orig_angle:
                                H = line_vp_H
                                info["validation"] = lv_detail
                                info["subset"] = "line_vp"
                                logger.info(
                                    "Fallback3 Line-VP: %.1f°→%.1f°",
                                    orig_angle, lv_angle,
                                )

            # Step 4: Compute calibration quality metrics
            quality_metrics = self.calibration_quality(
                best_kpts, H, min_confidence
            )
            info["calibration_rmse"] = quality_metrics.get("rmse", float("inf"))
            info["calibration_rmse_pixels"] = quality_metrics.get("rmse_pixels", float("inf"))
            info["n_inliers_calib"] = quality_metrics.get("n_inliers", 0)

            # Step 5: Save calibration overlay
            try:
                # Re-read first sampled frame for overlay
                cap2 = cv2.VideoCapture(video_path)
                cap2.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
                ret_overlay, overlay_frame = cap2.read()
                cap2.release()
                if ret_overlay:
                    overlay = self.draw_pitch_overlay(overlay_frame, H)
                    overlay_path = Path(video_path).stem + "_calib_overlay.jpg"
                    output_dir = Path("output")
                    output_dir.mkdir(exist_ok=True)
                    cv2.imwrite(str(output_dir / overlay_path), overlay)
                    info["calib_overlay"] = str(output_dir / overlay_path)
            except Exception as e:
                logger.warning("Failed to save calibration overlay: %s", e)

        logger.info(
            "Multi-frame (%d frames): %d/32 keypoints, "
            "%d used, quality=%.3f, voter_rmse=%.3fm",
            n_frames, n_detected, info["n_used"],
            info["quality"], info.get("voter_rmse", float("inf")),
        )

        return H, best_kpts, info

    # ── Keypoint Access ─────────────────────────────────────────────────────

    def get_keypoints_by_category(self, pixel_kpts: np.ndarray) -> dict:
        """
        Categorize detected keypoints by their role on the field.

        Returns dict with keys:
            corners, center_circle, center_line, goal_left, goal_right, all
        """
        categories = {
            "near_touchline": list(range(0, 6)),          # 01-06: Left touchline
            "near_area":      list(range(6, 13)),         # 07-13: Near penalty/goal area inner
            "center_line":    list(range(13, 17)),        # 15-18: Centre line + CC top/bottom
            "far_area":       list(range(17, 24)),        # 20-26: Far penalty/goal area inner
            "far_touchline":  list(range(24, 30)),        # 27-32: Right touchline
            "center_circle":  [30, 31],                   # 14, 19: CC left, CC right
        }

        result = {}
        for name, indices in categories.items():
            pts = np.array([pixel_kpts[i] for i in indices])
            result[name] = pts

        result["all"] = pixel_kpts
        return result

    @staticmethod
    def get_world_keypoints() -> np.ndarray:
        """Get the 32 world coordinates (32, 3) array in Roboflow 120×70."""
        return WORLD_KEYPOINTS.copy()

    @staticmethod
    def roboflow_to_fifa(points: np.ndarray) -> np.ndarray:
        """
        Convert points from Roboflow 120×70 (x=length, y=width) to FIFA 68×105 (x=width, y=length).

        Args:
            points: (N, 2) or (N, 3) array in Roboflow coordinates.

        Returns:
            Same shape array in FIFA coordinates.
        """
        if points.ndim == 1:
            x_rob, y_rob = points[0], points[1]
            x_fifa = (70.0 - y_rob) * (68.0 / 70.0)
            y_fifa = (120.0 - x_rob) * (105.0 / 120.0)
            return np.array([x_fifa, y_fifa])
        else:
            x_rob, y_rob = points[:, 0], points[:, 1]
            x_fifa = (70.0 - y_rob) * (68.0 / 70.0)
            y_fifa = (120.0 - x_rob) * (105.0 / 120.0)
            result = np.zeros_like(points)
            result[:, 0] = x_fifa
            result[:, 1] = y_fifa
            if points.shape[1] >= 3:
                result[:, 2] = points[:, 2]
            return result

    @staticmethod
    def get_fifa_conversion_matrix() -> np.ndarray:
        """Get the 3×3 matrix that converts Roboflow 120×70 → FIFA 68×105."""
        return _CONV_ROBOFLOW_TO_FIFA.copy()


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python pitch_keypoint_detector.py <image_or_video>")
        sys.exit(1)

    detector = PitchKeypointDetector()
    path = sys.argv[1]
    frame = cv2.imread(path) if path.endswith((".jpg", ".png")) else None

    if frame is None:
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"ERROR: Cannot read '{path}'")
            sys.exit(1)

    H, kpts, info = detector.calibrate(frame)
    print(f"\nResult: {info}")
    if H is not None:
        print(f"Homography:\n{H}")
