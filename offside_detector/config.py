"""
Soccer Field Configuration

Defines the standard FIFA football pitch with 32 keypoints detected by
Roboflow's football-field-detection model (YOLOv8-pose).

Keypoint indices correspond to the model output order.
World coordinates: x=0..68 (width), y=0..105 (length)
Center circle at (x=34, y=52.5)

The 32 keypoints cover:
  0-3:   Pitch corners (TL, TR, BR, BL in counter-clockwise order)
  4-7:   Goal line intersections
  8-11:  Penalty area corners (left penalty area, attacker side first)
  12-15: Goal area corners (left goal area, attacker side first)
  16-19: Center circle points (left, top, right, bottom)
  20-25: Left/Right penalty area endpoints (top/bottom)
  26-29: Center line points
  30-31: Center spot + penalty spot (one side)

IMPORTANT: The keypoint order may vary based on the specific Roboflow model version.
If detection seems off, check the model's actual keypoint mapping and update accordingly.
"""

import numpy as np
from typing import List, Tuple, Dict


class SoccerFieldConfiguration:
    """FIFA standard pitch dimensions and 32 keypoint coordinates."""

    # Pitch dimensions in meters
    WIDTH = 68.0    # x-axis
    LENGTH = 105.0  # y-axis

    # Markings dimensions (meters)
    PENALTY_AREA_WIDTH = 40.32   # 16.5m from each post = 7.32 + 2*16.5
    PENALTY_AREA_DEPTH = 16.5
    GOAL_AREA_WIDTH = 18.32      # 5.5m from each post = 7.32 + 2*5.5
    GOAL_AREA_DEPTH = 5.5
    GOAL_WIDTH = 7.32
    CENTER_CIRCLE_RADIUS = 9.15
    PENALTY_SPOT_DISTANCE = 11.0  # from goal line
    PENALTY_ARC_RADIUS = 9.15

    # 32 keypoint labels for clarity
    KEYPOINT_LABELS = [
        # 0-3: Pitch corners (counter-clockwise, starting from attacker-left)
        "corner_top_left",
        "corner_top_right",
        "corner_bottom_right",
        "corner_bottom_left",

        # 4-5: Goal line - left penalty area endpoints
        "penalty_area_left_top",
        "penalty_area_left_bottom",

        # 6-7: Goal line - right penalty area endpoints
        "penalty_area_right_top",
        "penalty_area_right_bottom",

        # 8-11: Left penalty area corners (attacker side → defender side)
        "left_penalty_area_att_top",
        "left_penalty_area_def_top",
        "left_penalty_area_def_bottom",
        "left_penalty_area_att_bottom",

        # 12-15: Left goal area corners
        "left_goal_area_att_top",
        "left_goal_area_def_top",
        "left_goal_area_def_bottom",
        "left_goal_area_att_bottom",

        # 16-19: Center circle (left, top, right, bottom)
        "center_circle_left",
        "center_circle_top",
        "center_circle_right",
        "center_circle_bottom",

        # 20-23: Right penalty area corners
        "right_penalty_area_att_top",
        "right_penalty_area_def_top",
        "right_penalty_area_def_bottom",
        "right_penalty_area_att_bottom",

        # 24-27: Right goal area corners
        "right_goal_area_att_top",
        "right_goal_area_def_top",
        "right_goal_area_def_bottom",
        "right_goal_area_att_bottom",

        # 28-29: Center line endpoints
        "center_line_left",
        "center_line_right",

        # 30-31: Spots
        "center_spot",
        "penalty_spot",
    ]

    @classmethod
    def get_world_coordinates(cls) -> np.ndarray:
        """
        Returns 32 keypoint world coordinates as (N, 2) numpy array.
        Coordinates are in meters: x ∈ [0, 68], y ∈ [0, 105].

        Coordinate system:
          y=0 --- top (attacker side for one half)
          |
          y=105 --- bottom (defender side)
          x=0 left, x=68 right
        """
        W, L = cls.WIDTH, cls.LENGTH
        mid_x = W / 2
        mid_y = L / 2
        half_w = W / 2
        half_pa_w = cls.PENALTY_AREA_WIDTH / 2
        half_ga_w = cls.GOAL_AREA_WIDTH / 2

        coords = [
            # 0-3: Pitch corners (counter-clockwise)
            [0, 0],           # corner_top_left
            [W, 0],           # corner_top_right
            [W, L],           # corner_bottom_right
            [0, L],           # corner_bottom_left

            # 4-5: Left penalty area endpoints on goal line
            [mid_x - half_pa_w, 0],
            [mid_x - half_pa_w, L],

            # 6-7: Right penalty area endpoints on goal line
            [mid_x + half_pa_w, 0],
            [mid_x + half_pa_w, L],

            # 8-11: Left penalty area corners
            [mid_x - half_pa_w, 0],                      # att_top
            [mid_x - half_pa_w, cls.PENALTY_AREA_DEPTH],   # def_top
            [mid_x - half_pa_w, L - cls.PENALTY_AREA_DEPTH],  # def_bottom
            [mid_x - half_pa_w, L],                       # att_bottom

            # 12-15: Left goal area corners
            [mid_x - half_ga_w, 0],
            [mid_x - half_ga_w, cls.GOAL_AREA_DEPTH],
            [mid_x - half_ga_w, L - cls.GOAL_AREA_DEPTH],
            [mid_x - half_ga_w, L],

            # 16-19: Center circle points
            [mid_x - cls.CENTER_CIRCLE_RADIUS, mid_y],
            [mid_x, mid_y - cls.CENTER_CIRCLE_RADIUS],
            [mid_x + cls.CENTER_CIRCLE_RADIUS, mid_y],
            [mid_x, mid_y + cls.CENTER_CIRCLE_RADIUS],

            # 20-23: Right penalty area corners
            [mid_x + half_pa_w, 0],
            [mid_x + half_pa_w, cls.PENALTY_AREA_DEPTH],
            [mid_x + half_pa_w, L - cls.PENALTY_AREA_DEPTH],
            [mid_x + half_pa_w, L],

            # 24-27: Right goal area corners
            [mid_x + half_ga_w, 0],
            [mid_x + half_ga_w, cls.GOAL_AREA_DEPTH],
            [mid_x + half_ga_w, L - cls.GOAL_AREA_DEPTH],
            [mid_x + half_ga_w, L],

            # 28-29: Center line endpoints
            [0, mid_y],
            [W, mid_y],

            # 30-31: Spots
            [mid_x, mid_y],                                # center spot
            [mid_x, L - cls.PENALTY_SPOT_DISTANCE],       # penalty spot (one side)
        ]

        return np.array(coords, dtype=np.float32)

    @classmethod
    def get_keypoint_indices(cls, category: str) -> List[int]:
        """Get keypoint indices for a specific category."""
        mapping = {
            "corners": [0, 1, 2, 3],
            "penalty_area": [4, 5, 6, 7],
            "left_penalty": [8, 9, 10, 11],
            "left_goal": [12, 13, 14, 15],
            "center_circle": [16, 17, 18, 19],
            "right_penalty": [20, 21, 22, 23],
            "right_goal": [24, 25, 26, 27],
            "center_line": [28, 29],
            "spots": [30, 31],
        }
        return mapping.get(category, [])


# YOLOv8 local model configuration
YOLO_CONFIG = {
    "model_variants": {
        "n": "yolov8n.pt",    # ~6MB, fastest, recommended for CPU
        "s": "yolov8s.pt",    # ~22MB, balanced
        "m": "yolov8m.pt",    # ~52MB, most accurate
    },
    "default_model": "n",
    "person_class_id": 0,     # COCO class index for "person"
}

# Detection confidence thresholds
DETECTION_CONFIDENCE = {
    "player": 0.35,           # minimum confidence for player detection
    "goalkeeper": 0.35,       # minimum confidence for goalkeeper
    "field_keypoint": 0.5,    # (for Roboflow auto mode)
}

# Roboflow API model IDs (optional, for auto field detection)
ROBOFLOW_MODELS = {
    "field_detection": "football-field-detection-f07vi/14",
}

# Visualization settings
VISUALIZATION = {
    "offside_line_color": (0, 0, 255),       # Red in BGR
    "offside_line_thickness": 1,
    "player_box_thickness": 1,
    "last_defender_box_thickness": 2,
    "offside_player_box_color": (0, 0, 255), # Red
    "onside_player_box_color": (0, 255, 0),  # Green
    "defender_box_color": (255, 0, 0),       # Blue
    "goalkeeper_box_color": (0, 165, 255),   # Orange
    "referee_box_color": (0, 255, 255),      # Yellow
    "defending_player_box_color": (255, 100, 0),  # Blue
    "ball_color": (255, 255, 255),           # White
    "field_overlay_alpha": 0.3,
    "text_color": (255, 255, 255),
    "text_bg_color": (0, 0, 0),
}
