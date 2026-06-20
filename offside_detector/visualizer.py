"""
Visualizer Module

Draws offside analysis overlays on video frames:
  - Offside line (across the full pitch width)
  - All players with color-coded bounding boxes and role labels
    * Offside attacker: red box + "OFFSIDE #ID" label
    * Onside attacker: green box + "ATK #ID" label
    * Defending player: blue box + "DEF #ID" label
    * Goalkeeper: orange box + "GK #ID" label
    * Referee: yellow box + "REF #ID" label
  - Ball indicator (white circle)
  - Last defender indicator
  - Mini pitch map (top-down view) with player positions
  - Frame information overlay (frame number, offside status)
"""

import numpy as np
import cv2
from typing import Optional, List, Dict, Tuple

from .offside_analyzer import OffsideResult, AttackDirection
from .player_detector import PlayerDetection, FrameDetection
from .config import VISUALIZATION, SoccerFieldConfiguration


class OffsideVisualizer:
    """
    Draws offside analysis overlays on video frames.

    Usage:
        visualizer = OffsideVisualizer()
        annotated = visualizer.draw(frame, offside_result, frame_detection)
        cv2.imwrite("output.jpg", annotated)
    """

    def __init__(self):
        self.field_config = SoccerFieldConfiguration()
        self._colors = {
            "offside_line": VISUALIZATION["offside_line_color"],
            "offside_player": VISUALIZATION["offside_player_box_color"],
            "onside_player": VISUALIZATION["onside_player_box_color"],
            "defender": VISUALIZATION["defender_box_color"],
            "goalkeeper": VISUALIZATION["goalkeeper_box_color"],
            "referee": VISUALIZATION["referee_box_color"],
            "defending_player": VISUALIZATION["defending_player_box_color"],
            "ball": VISUALIZATION["ball_color"],
            "text": VISUALIZATION["text_color"],
            "text_bg": VISUALIZATION["text_bg_color"],
        }
        self.line_thickness = 1
        self.box_thickness = 1
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw(
        self,
        frame: np.ndarray,
        offside_result: OffsideResult,
        frame_det: Optional[FrameDetection] = None,
        show_minimap: bool = True,
        show_player_info: bool = True,
    ) -> np.ndarray:
        """
        Draw offside analysis overlay on a frame.

        Args:
            frame: BGR video frame
            offside_result: Analysis result for this frame
            frame_det: All detections (players, GKs, referees, ball) for this frame.
                       If None, only offside-relevant players from offside_result
                       are drawn (legacy mode).
            show_minimap: Whether to show the pitch minimap
            show_player_info: Whether to label players

        Returns:
            Annotated frame (copy)
        """
        result = frame.copy()

        # Draw offside line
        result = self._draw_offside_line(result, offside_result)

        # Draw all players with color-coded boxes and labels
        result = self._draw_players(result, offside_result, frame_det)

        # Draw offside indicator
        result = self._draw_offside_indicator(result, offside_result)

        # Draw pitch minimap
        if show_minimap:
            result = self._draw_minimap(result, offside_result)

        return result

    def _draw_offside_line(
        self, frame: np.ndarray, result: OffsideResult
    ) -> np.ndarray:
        """Draw the offside reference line across the pitch using Homography projection."""
        if result.offside_line_pixels is None:
            return frame

        p1 = tuple(result.offside_line_pixels[0].astype(int))
        p2 = tuple(result.offside_line_pixels[1].astype(int))

        # Draw thin red line
        cv2.line(
            frame, p1, p2,
            self._colors["offside_line"],
            self.line_thickness,
            cv2.LINE_AA,
        )

        # Draw very faint semi-transparent overlay on the offside side
        overlay = frame.copy()
        h, w = frame.shape[:2]

        if result.attack_direction == AttackDirection.TOP_TO_BOTTOM:
            # Offside zone is below the line
            pts = np.array([
                [p1[0], p1[1]], [p2[0], p2[1]],
                [w, h], [0, h]
            ], dtype=np.int32)
        elif result.attack_direction == AttackDirection.BOTTOM_TO_TOP:
            # Offside zone is above the line
            pts = np.array([
                [0, 0], [w, 0],
                [p2[0], p2[1]], [p1[0], p1[1]]
            ], dtype=np.int32)
        elif result.attack_direction == AttackDirection.LEFT_TO_RIGHT:
            # Offside zone is to the right of the line
            pts = np.array([
                [p1[0], p1[1]], [p2[0], p2[1]],
                [w, h], [w, 0]
            ], dtype=np.int32)
        elif result.attack_direction == AttackDirection.RIGHT_TO_LEFT:
            # Offside zone is to the left of the line
            pts = np.array([
                [0, 0], [0, h],
                [p2[0], p2[1]], [p1[0], p1[1]]
            ], dtype=np.int32)
        else:
            pts = None

        if pts is not None:
            cv2.fillPoly(overlay, [pts], (0, 0, 200))
            cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

        return frame

    def _draw_players(
        self, frame: np.ndarray, result: OffsideResult,
        frame_det: Optional[FrameDetection] = None,
    ) -> np.ndarray:
        """Draw all players with color-coded thin bounding boxes and role labels.

        Color scheme:
          - Attacking player:   green box + "ATK #ID"
          - Defending player:   blue box + "DEF #ID"
          - Goalkeeper:         orange box + "GK #ID"
          - Referee:            yellow box + "REF #ID"
          - Ball:               small white circle
          - Last defender:      blue box + "LAST DEFENDER" (slightly thicker)

        No per-player offside judgment is displayed — only the offside line.
        """

        # --- If frame_det is provided, draw ALL players from it ---
        if frame_det is not None:
            drawn_ids = set()

            # --- Goalkeepers (orange, thin) ---
            for p in frame_det.goalkeepers:
                bbox = p.bbox
                x1, y1, x2, y2 = map(int, bbox)
                color = self._colors["goalkeeper"]
                label = f"GK #{p.track_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)
                self._draw_label(frame, label, x1, y1 - 10, color)
                drawn_ids.add(p.track_id)

            # --- Referees (yellow, thin) ---
            for p in frame_det.referees:
                bbox = p.bbox
                x1, y1, x2, y2 = map(int, bbox)
                color = self._colors["referee"]
                label = f"REF #{p.track_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)
                self._draw_label(frame, label, x1, y1 - 10, color)
                drawn_ids.add(p.track_id)

            # --- Outfield players (green/blue by team, thin) ---
            for p in frame_det.players:
                if p.track_id in drawn_ids:
                    continue
                bbox = p.bbox
                x1, y1, x2, y2 = map(int, bbox)

                if p.team == "attacking":
                    color = self._colors["onside_player"]
                    label = f"ATK #{p.track_id}"
                elif p.team == "defending":
                    color = self._colors["defending_player"]
                    label = f"DEF #{p.track_id}"
                else:
                    color = (128, 128, 128)  # Gray for unknown
                    label = f"#{p.track_id}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)
                self._draw_label(frame, label, x1, y1 - 10, color)
                drawn_ids.add(p.track_id)

            # --- Ball indicator ---
            if frame_det.ball is not None:
                bx, by = frame_det.ball.foot_position
                bx, by = int(bx), int(by)
                cv2.circle(frame, (bx, by), 4, self._colors["ball"], -1)
                cv2.circle(frame, (bx, by), 5, (0, 0, 0), 1)  # outline

        # Last defender: blue box (slightly thicker)
        if result.second_last_defender:
            bbox = result.second_last_defender.get("bbox")
            if bbox:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2),
                    self._colors["defender"], 2,
                )
                label = "LAST DEFENDER"
                self._draw_label(frame, label, x1, y1 - 10, self._colors["defender"])

        return frame

    def _draw_offside_indicator(
        self, frame: np.ndarray, result: OffsideResult
    ) -> np.ndarray:
        """Draw frame number only (no offside status banner)."""
        h, w = frame.shape[:2]

        # Frame number at bottom-left
        cv2.putText(
            frame, f"Frame: {result.frame_idx}",
            (10, h - 15),
            self.font, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
        )

        return frame

    def _draw_minimap(
        self, frame: np.ndarray, result: OffsideResult
    ) -> np.ndarray:
        """Draw a small pitch minimap in the top-right corner showing player positions."""
        h, w = frame.shape[:2]

        # Minimap dimensions
        map_w = 160
        map_h = 250
        margin = 15

        map_x1 = w - map_w - margin
        map_y1 = margin

        # Draw pitch background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (map_x1 - 5, map_y1 - 5),
            (map_x1 + map_w + 5, map_y1 + map_h + 5),
            (30, 30, 30), -1,
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Scale: map pixel / world meter
        scale_x = map_w / 68.0
        scale_y = map_h / 105.0

        def world_to_map(wx, wy):
            mx = map_x1 + wx * scale_x
            my = map_y1 + wy * scale_y
            return int(mx), int(my)

        # Draw pitch outline
        pitch_pts = [
            world_to_map(0, 0), world_to_map(68, 0),
            world_to_map(68, 105), world_to_map(0, 105),
        ]
        cv2.polylines(frame, [np.array(pitch_pts)], True, (100, 100, 100), 1)

        # Draw halfway line
        cv2.line(
            frame,
            world_to_map(0, 52.5), world_to_map(68, 52.5),
            (100, 100, 100), 1,
        )

        # Draw center circle
        cx, cy = world_to_map(34, 52.5)
        cv2.circle(frame, (cx, cy), int(9.15 * scale_x), (100, 100, 100), 1)

        # Draw offside line on minimap
        if result.offside_line_y is not None:
            off_y = int(result.offside_line_y)
            if 0 <= off_y <= 105:
                p1 = world_to_map(0, result.offside_line_y)
                p2 = world_to_map(68, result.offside_line_y)
                cv2.line(frame, p1, p2, (0, 0, 255), 1)
                # Semi-transparent offside zone
                overlay = frame.copy()
                if result.attack_direction == AttackDirection.TOP_TO_BOTTOM:
                    pts = np.array([
                        world_to_map(0, result.offside_line_y),
                        world_to_map(68, result.offside_line_y),
                        world_to_map(68, 105), world_to_map(0, 105),
                    ])
                else:
                    pts = np.array([
                        world_to_map(0, 0), world_to_map(68, 0),
                        world_to_map(68, result.offside_line_y),
                        world_to_map(0, result.offside_line_y),
                    ])
                cv2.fillPoly(overlay, [pts], (0, 0, 200))
                cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)

        # Draw all players on minimap (team colors only, no offside markers)
        for p in result.all_players:
            wx, wy = p["world_x"], p["world_y"]
            if wx is None or wy is None:
                continue
            mx, my = world_to_map(wx, wy)
            class_name = p.get("class_name", "player")
            team = p.get("team", "unknown")

            if class_name == "goalkeeper":
                color = (0, 165, 255)   # Orange
                r = 3
            elif class_name == "referee":
                color = (0, 255, 255)   # Yellow
                r = 2
            elif team == "attacking":
                color = (0, 200, 0)     # Green
                r = 2
            elif team == "defending":
                color = (255, 100, 0)   # Blue
                r = 2
            else:
                color = (150, 150, 150)  # Gray
                r = 2

            cv2.circle(frame, (mx, my), r, color, -1)
            cv2.circle(frame, (mx, my), r, (255, 255, 255), 1)  # white outline

        # Draw last defender (blue dot)
        if result.second_last_defender:
            p = result.second_last_defender
            mx, my = world_to_map(p["world_x"], p["world_y"])
            cv2.circle(frame, (mx, my), 5, (255, 0, 0), -1)
            cv2.circle(frame, (mx, my), 5, (255, 255, 255), 1)

        return frame

    def _draw_label(
        self, frame: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int],
    ):
        """Draw a text label with background."""
        if y < 5:
            y = 20

        text_size = cv2.getTextSize(text, self.font, 0.4, 1)[0]
        cv2.rectangle(
            frame,
            (x, y - text_size[1] - 4),
            (x + text_size[0] + 4, y + 2),
            color, -1,
        )
        cv2.putText(
            frame, text, (x + 2, y),
            self.font, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
        )

    def create_comparison_frame(
        self,
        original: np.ndarray,
        annotated: np.ndarray,
    ) -> np.ndarray:
        """Create a side-by-side comparison of original vs annotated frame."""
        h = max(original.shape[0], annotated.shape[0])
        w = original.shape[1] + annotated.shape[1]

        # Resize to same height
        o = cv2.resize(original, (original.shape[1], h))
        a = cv2.resize(annotated, (annotated.shape[1], h))

        combined = np.hstack([o, a])

        # Labels
        cv2.putText(
            combined, "Original", (10, 30),
            self.font, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            combined, "Offside Analysis", (original.shape[1] + 10, 30),
            self.font, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
        )

        return combined
