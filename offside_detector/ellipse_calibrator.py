#!/usr/bin/env python3
"""
Ellipse-based Center Circle Calibrator

Detects the center circle as an ellipse in the image and computes the
perspective homography from the known circle geometry.

Theory:
  - The center circle in reality is a perfect circle (radius 9.15m, center at (34, 52.5))
  - Under perspective projection, it appears as an ellipse in the image
  - From the ellipse parameters (center, axes, rotation), we can recover the homography

The ellipse gives us 5 constraints:
  1. Center pixel → world center (2 constraints)
  2. Major axis length + orientation → one world direction (scale + rotation)
  3. Minor axis length → foreshortening factor = 2nd world direction scale

Combined with the known field orientation (attack_dir), we can construct
a full perspective homography that accurately maps the playing field.

Usage:
    from .ellipse_calibrator import EllipseCalibrator
    cal = EllipseCalibrator()
    H = cal.calibrate(frame, attack_dir="left_to_right")
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List


class EllipseCalibrator:
    """
    Calibrate the field using ellipse detection of the center circle.
    No user clicks needed — fully automatic.
    """

    CENTER_X = 34.0
    CENTER_Y = 52.5
    CIRCLE_RADIUS = 9.15  # meters
    FIELD_WIDTH = 68.0
    FIELD_LENGTH = 105.0

    @staticmethod
    def detect_center_circle_ellipse(frame: np.ndarray) -> Optional[Tuple]:
        """
        Detect the center circle as an ellipse in the image.

        Strategy:
          1. Isolate green field region via HSV mask
          2. Find white contours on the field
          3. Filter contours by size and circularity
          4. Fit ellipse to the best candidate

        Returns:
            ((cx, cy), (major_axis, minor_axis), angle_deg) or None
        """
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Isolate green field
        lower_green = np.array([35, 30, 30])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Dilate to include white lines on green
        kernel = np.ones((7, 7), np.uint8)
        green_mask = cv2.dilate(green_mask, kernel, iterations=2)

        # Find white/bright regions within the field
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Use CLAHE for contrast enhancement (helps with uneven broadcast lighting)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        _, white_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # Combine: white AND on-field
        combined = cv2.bitwise_and(white_mask, green_mask)

        # Edge detection
        edges = cv2.Canny(combined, 30, 90)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Filter and score candidates
        best_ellipse = None
        best_score = 0

        for cnt in contours:
            if len(cnt) < 20:  # Need enough points for ellipse fit
                continue

            # Filter by size: circle should be 5-40% of image width
            area = cv2.contourArea(cnt)
            min_area = (w * 0.05) ** 2 * np.pi / 4
            max_area = (w * 0.4) ** 2 * np.pi / 4
            if area < min_area or area > max_area:
                continue

            try:
                ellipse = cv2.fitEllipse(cnt)
            except cv2.error:
                continue

            (cx, cy), (ma, MA), angle = ellipse

            # Filter unrealistic aspect ratios (circle → ellipse should have AR between 0.3 and 1.0)
            ar = min(ma, MA) / max(ma, MA)
            if ar < 0.25:
                continue

            # Score: prefer ellipses near image center, with good aspect ratio
            center_dist = np.sqrt((cx - w/2)**2 + (cy - h/2)**2) / max(w, h)
            score = ar * (1.0 - center_dist * 0.5)

            if score > best_score:
                best_score = score
                best_ellipse = ellipse

        if best_ellipse is not None:
            (cx, cy), (ma, MA), angle = ellipse
            # Ensure major_axis >= minor_axis
            if ma < MA:
                ma, MA = MA, ma
                angle = angle + 90
            print(f"[EllipseDetect] Found: center=({cx:.0f},{cy:.0f}) "
                  f"axes=({ma:.0f},{MA:.0f}) angle={angle:.1f}° score={best_score:.2f}")
            return ((cx, cy), (ma, MA), angle % 180)

        return None

    def calibrate(
        self,
        frame: np.ndarray,
        attack_dir: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        Full calibration pipeline: detect ellipse → compute homography.

        Args:
            frame: BGR video frame
            attack_dir: "left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"

        Returns:
            3x3 homography matrix (pixel → FIFA 68x105), or None
        """
        ellipse = self.detect_center_circle_ellipse(frame)
        if ellipse is None:
            print("[EllipseCalib] No ellipse detected")
            return None

        return self._build_from_ellipse(ellipse, attack_dir, frame.shape[:2])

    def _build_from_ellipse(
        self,
        ellipse: Tuple,
        attack_dir: Optional[str],
        frame_shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """
        Compute perspective homography from ellipse parameters.

        The ellipse is the projection of the center circle (radius 9.15m).
        The ellipse center maps to the circle center (34, 52.5) in world.

        From the ellipse, we can extract 4 point correspondences:
          - Ellipse center → world center
          - Major axis endpoints → horizontal/vertical diameter endpoints
          - Minor axis endpoints → vertical/horizontal diameter endpoints

        These 5+ points are enough for a robust homography via DLT.
        """
        (cx_px, cy_px), (major_axis, minor_axis), angle_deg = ellipse
        angle_rad = np.radians(angle_deg)

        h, w = frame_shape
        R = self.CIRCLE_RADIUS

        # Determine field orientation
        is_horizontal_field = attack_dir is not None and attack_dir in (
            "left_to_right", "right_to_left"
        )

        # Pixel points: ellipse center + 4 axis endpoints
        # Major axis direction vector (unit)
        mx = np.cos(angle_rad)
        my = np.sin(angle_rad)

        # Minor axis direction vector (perpendicular)
        nx = -my
        ny = mx

        pixel_pts = np.array([
            [cx_px, cy_px],                           # 0: center
            [cx_px + mx * major_axis / 2, cy_px + my * major_axis / 2],  # 1: major +
            [cx_px - mx * major_axis / 2, cy_px - my * major_axis / 2],  # 2: major -
            [cx_px + nx * minor_axis / 2, cy_px + ny * minor_axis / 2],  # 3: minor +
            [cx_px - nx * minor_axis / 2, cy_px - ny * minor_axis / 2],  # 4: minor -
        ], dtype=np.float32)

        # Determine which image axis corresponds to which world axis
        # In a horizontally-oriented field (goals left-right):
        #   - The vertical image axis maps to world Y (length)
        #   - The horizontal image axis maps to world X (width)
        # The ellipse major axis direction tells us which is which

        if is_horizontal_field:
            # Horizontal field: goals at x=0 and x=68
            # Major axis closer to vertical → maps to world Y (length)
            if abs(my) > abs(mx):
                # Major ≈ vertical image → world Y (center line direction)
                world_pts = np.array([
                    [self.CENTER_X, self.CENTER_Y],                     # center
                    [self.CENTER_X, self.CENTER_Y + R],                 # major + → toward far goal
                    [self.CENTER_X, self.CENTER_Y - R],                 # major - → toward near goal
                    [self.CENTER_X + R, self.CENTER_Y],                 # minor + → right
                    [self.CENTER_X - R, self.CENTER_Y],                 # minor - → left
                ], dtype=np.float32)
            else:
                # Major ≈ horizontal image → world X (width direction)
                world_pts = np.array([
                    [self.CENTER_X, self.CENTER_Y],
                    [self.CENTER_X + R, self.CENTER_Y],
                    [self.CENTER_X - R, self.CENTER_Y],
                    [self.CENTER_X, self.CENTER_Y + R],
                    [self.CENTER_X, self.CENTER_Y - R],
                ], dtype=np.float32)
        else:
            # Vertical field: goals at y=0 and y=105
            if abs(my) > abs(mx):
                # Major ≈ vertical image → world Y (length)
                world_pts = np.array([
                    [self.CENTER_X, self.CENTER_Y],
                    [self.CENTER_X, self.CENTER_Y + R],
                    [self.CENTER_X, self.CENTER_Y - R],
                    [self.CENTER_X + R, self.CENTER_Y],
                    [self.CENTER_X - R, self.CENTER_Y],
                ], dtype=np.float32)
            else:
                world_pts = np.array([
                    [self.CENTER_X, self.CENTER_Y],
                    [self.CENTER_X + R, self.CENTER_Y],
                    [self.CENTER_X - R, self.CENTER_Y],
                    [self.CENTER_X, self.CENTER_Y + R],
                    [self.CENTER_X, self.CENTER_Y - R],
                ], dtype=np.float32)

        # Compute homography via RANSAC (5 points → robust)
        H, mask = cv2.findHomography(
            pixel_pts, world_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
        )

        if H is None:
            print("[EllipseCalib] Homography computation failed")
            return None

        # Validate: check reprojection error
        proj = cv2.perspectiveTransform(pixel_pts.reshape(-1, 1, 2), H)
        errors = np.linalg.norm(proj.reshape(-1, 2) - world_pts, axis=1)
        rmse = np.sqrt(np.mean(errors ** 2))
        max_err = np.max(errors)

        print(f"[EllipseCalib] Homography: RMSE={rmse:.2f}m max_err={max_err:.2f}m "
              f"inliers={int(mask.sum())}/5")

        # Reject poor calibrations
        if rmse > 5.0 or max_err > 10.0:
            print(f"[EllipseCalib] Rejected: RMSE={rmse:.2f}m > 5.0m")
            return None

        return H

    def calibrate_with_vp(
        self,
        frame: np.ndarray,
        attack_dir: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        Enhanced calibration: ellipse + vanishing point constraint.

        Combines ellipse geometry with field line VP for higher accuracy.
        """
        # Step 1: Ellipse-based homography
        H_ellipse = self.calibrate(frame, attack_dir)
        if H_ellipse is None:
            return None

        # Step 2: Try VP constraint from field lines
        try:
            from .pitch_keypoint_detector import PitchKeypointDetector
            field_lines = PitchKeypointDetector.detect_field_lines(frame)
            if field_lines and len(field_lines) >= 2:
                vp, vp_quality = PitchKeypointDetector.find_vp_from_lines(field_lines)
                if vp is not None and vp_quality > 0.3:
                    # Use VP to refine the homography
                    H_refined = self._refine_with_vp(H_ellipse, vp, vp_quality)
                    if H_refined is not None:
                        print(f"[EllipseCalib] VP refinement applied (quality={vp_quality:.2f})")
                        return H_refined
        except ImportError:
            pass

        return H_ellipse

    @staticmethod
    def _refine_with_vp(
        H_initial: np.ndarray,
        vp: np.ndarray,
        vp_quality: float,
    ) -> Optional[np.ndarray]:
        """
        Refine homography using VP constraint.

        The VP tells us where parallel field lines (along world-X) converge.
        Column 1 of H maps world-X direction to pixel space → should pass through VP.
        """
        vpx, vpy = float(vp[0]), float(vp[1])

        # Sample points from the initial homography
        # Generate virtual point pairs along world-X at different world-Y
        src_pts = []
        dst_pts = []

        # Use the 4 corners of the field
        for wy in [0, 52.5, 105]:
            for wx in [0, 34, 68]:
                px, py = EllipseCalibrator._world_to_pixel_single(H_initial, wx, wy)
                if px is not None:
                    src_pts.append([px, py])
                    dst_pts.append([wx, wy])

        if len(src_pts) < 4:
            return None

        src = np.array(src_pts, dtype=np.float32).reshape(-1, 1, 2)
        dst = np.array(dst_pts, dtype=np.float32).reshape(-1, 1, 2)

        # Re-compute with VP constraint as additional point pairs
        # Add virtual constraining points: for each pair with same wx, different wy,
        # the line through their projections should pass through VP
        vp_weight = max(5.0, 20.0 * vp_quality)

        # Build augmented point set
        all_src = list(src_pts)
        all_dst = list(dst_pts)
        weights = [1.0] * len(src_pts)

        # VP constraint: add the VP as a far-field point at infinity
        # This forces column 1 of H to map to VP direction
        far_factor = 1000.0
        for (px, py), (wx, wy) in zip(src_pts, dst_pts):
            # Extrapolate from (px,py) through VP to infinity along world-X
            dx = px - vpx
            dy = py - vpy
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 1:
                continue
            # Far point along same direction
            far_px = vpx + dx / dist * far_factor
            far_py = vpy + dy / dist * far_factor
            # Corresponds to world-X → ±∞
            all_src.append([far_px, far_py])
            all_dst.append([wx + np.sign(wx - 34) * 500, wy])
            weights.append(vp_weight * 0.1)

        if len(all_src) < 6:
            return None

        all_src_arr = np.array(all_src, dtype=np.float32).reshape(-1, 1, 2)
        all_dst_arr = np.array(all_dst, dtype=np.float32).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            all_src_arr, all_dst_arr,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
        )

        return H

    @staticmethod
    def _world_to_pixel_single(H: np.ndarray, wx: float, wy: float) -> Optional[Tuple[float, float]]:
        """Project a single world point to pixel using inverse homography."""
        try:
            invH = np.linalg.inv(H)
            pt = np.array([[[wx, wy]]], dtype=np.float32)
            pix = cv2.perspectiveTransform(pt, invH)
            return float(pix[0, 0, 0]), float(pix[0, 0, 1])
        except Exception:
            return None