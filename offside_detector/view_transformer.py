"""
View Transformer Module

Handles coordinate transformations between pixel space and world space
using homography matrices computed from field keypoint detection.

Provides:
  - pixel_to_world: pixel coords → meters on the field
  - world_to_pixel: meters → pixel coords on the image
  - is_in_field: check if a world coordinate is inside the pitch
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List


class ViewTransformer:
    """
    Coordinate transformer using homography.

    This is the core module that maps player foot positions from the video
    frame (pixel coordinates) to the actual field positions (world coordinates
    in meters), enabling accurate offside line calculation.

    Usage:
        transformer = ViewTransformer()
        transformer.set_homography(homography_matrix)
        world_pos = transformer.pixel_to_world(foot_x, foot_y)
    """

    def __init__(self):
        self.homography: Optional[np.ndarray] = None
        self.inverse_homography: Optional[np.ndarray] = None
        self.field_width: float = 68.0
        self.field_length: float = 105.0

    def set_homography(
        self,
        homography: np.ndarray,
        field_width: float = 68.0,
        field_length: float = 105.0,
    ):
        """Set the homography matrix and its inverse."""
        self.homography = homography
        self.inverse_homography = np.linalg.inv(homography)
        self.field_width = field_width
        self.field_length = field_length

    @property
    def is_ready(self) -> bool:
        return self.homography is not None and self.inverse_homography is not None

    def pixel_to_world(self, x: float, y: float) -> Optional[np.ndarray]:
        """
        Convert a single pixel point to world coordinates.

        Args:
            x, y: Pixel coordinates in the image

        Returns:
            [wx, wy] in meters, or None if transform not available.
            wx = width direction (0..68), wy = length direction (0..105)
        """
        if not self.is_ready:
            return None

        pt = np.array([[[x, y]]], dtype=np.float32)
        try:
            world = cv2.perspectiveTransform(pt, self.homography)
            return world[0, 0]  # (wx, wy) where wx∈0-68, wy∈0-105
        except Exception:
            return None

    def world_to_pixel(self, wx: float, wy: float) -> Optional[np.ndarray]:
        """
        Convert world coordinates to pixel coordinates.

        Args:
            wx, wy: World coordinates in meters
                      wx = width direction (0..68)
                      wy = length direction (0..105)

        Returns:
            [px, py] in image pixels
        """
        if not self.is_ready:
            return None

        pt = np.array([[[wx, wy]]], dtype=np.float32)
        try:
            pixel = cv2.perspectiveTransform(pt, self.inverse_homography)
            return pixel[0, 0]
        except Exception as e:
            return None

    def batch_pixel_to_world(self, points: np.ndarray) -> Optional[np.ndarray]:
        """
        Convert multiple pixel points to world coordinates.

        Args:
            points: (N, 2) array of pixel coordinates

        Returns:
            (N, 2) array of world coordinates in meters
        """
        if not self.is_ready or len(points) == 0:
            return None

        pts = points.reshape(-1, 1, 2).astype(np.float32)
        try:
            world = cv2.perspectiveTransform(pts, self.homography)
            return world.reshape(-1, 2)
        except Exception:
            return None

    def batch_world_to_pixel(self, points: np.ndarray) -> Optional[np.ndarray]:
        """Convert multiple world points to pixel coordinates."""
        if not self.is_ready or len(points) == 0:
            return None

        pts = points.reshape(-1, 1, 2).astype(np.float32)
        try:
            pixel = cv2.perspectiveTransform(pts, self.inverse_homography)
            return pixel.reshape(-1, 2)
        except Exception:
            return None

    def is_in_field(self, wx: float, wy: float, margin: float = 5.0) -> bool:
        """Check if a world coordinate is within the field boundaries."""
        return (
            -margin <= wx <= self.field_width + margin and
            -margin <= wy <= self.field_length + margin
        )

    def get_offside_line_world(
        self,
        last_defender_position: Optional[np.ndarray],
        ball_position: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        """
        Calculate the offside line in world coordinates.

        Rule: A player is offside if they are closer to the opponent's goal line
        than BOTH the ball AND the second-last opponent.

        The offside line in y-coordinate is the one CLOSER to the opponent's goal
        (i.e., larger y if attacking toward y=105, or smaller y if toward y=0).

        Returns:
            y-coordinate of the offside line, or None if undetermined
        """
        if last_defender_position is None:
            return None

        ld_y = last_defender_position[1]

        # If ball position is available, use it for offside determination
        # Halfway line (52.5m) is used to determine attack direction
        if ld_y > self.field_length / 2:
            return ld_y  # defending near goal at y=105
        else:
            return ld_y  # defending near goal at y=0


class ManualViewTransformer:
    """
    Manual view transformer using user-specified calibration points.

    This is a fallback when automatic field detection fails.
    User clicks 4 known points on the field to establish homography.

    Usage:
        mvt = ManualViewTransformer()
        mvt.calibrate(
            image_points=[[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
            world_points=[[wx1,wy1], [wx2,wy2], [wx3,wy3], [wx4,wy4]],
        )
    """

    def __init__(self):
        self.homography: Optional[np.ndarray] = None
        self.inverse_homography: Optional[np.ndarray] = None

    def calibrate(
        self,
        image_points: List[Tuple[float, float]],
        world_points: List[Tuple[float, float]],
    ) -> bool:
        """
        Calibrate using 4+ corresponding point pairs.

        Args:
            image_points: 4+ pixel coordinates [(x1,y1), ...]
            world_points: 4+ world coordinates [(wx1,wy1), ...]

        Returns:
            True if calibration succeeded
        """
        if len(image_points) < 4 or len(world_points) < 4:
            return False

        src = np.array(image_points, dtype=np.float32)
        dst = np.array(world_points, dtype=np.float32)

        try:
            self.homography, mask = cv2.findHomography(src, dst, method=0)
            self.inverse_homography = np.linalg.inv(self.homography)
            return True
        except Exception:
            return False

    def pixel_to_world(self, x: float, y: float) -> Optional[np.ndarray]:
        if self.homography is None:
            return None
        pt = np.array([[[x, y]]], dtype=np.float32)
        world = cv2.perspectiveTransform(pt, self.homography)
        return world[0, 0]
