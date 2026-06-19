"""
Offside Analyzer Module

Core offside rule logic. Determines:
  1. Which team is attacking and which is defending
  2. Who is the last defender (excluding goalkeeper)
  3. Where the offside line should be drawn
  4. Which attacking players are in offside position
  5. Whether those players are actively involved in play

Based on IFAB Law 11 - Offside:
  - A player is in an offside position if any part of the head, body or feet
    is nearer to the opponents' goal line than both the ball and the
    second-last opponent.
  - The hands and arms of all players are not considered.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from enum import Enum

from .player_detector import PlayerDetection, FrameDetection
from .view_transformer import ViewTransformer


class AttackDirection(Enum):
    """Which direction the attacking team is playing."""
    TOP_TO_BOTTOM = "top_to_bottom"    # attacking toward y=105
    BOTTOM_TO_TOP = "bottom_to_top"    # attacking toward y=0
    UNKNOWN = "unknown"


@dataclass
class OffsideResult:
    """Result of offside analysis for one frame."""
    frame_idx: int
    attack_direction: AttackDirection = AttackDirection.UNKNOWN
    offside_line_y: Optional[float] = None       # world y coordinate
    offside_line_pixels: Optional[np.ndarray] = None  # [p1, p2] pixel line
    last_defender: Optional[Dict] = None         # info about last defender
    second_last_defender: Optional[Dict] = None  # info about second-last defender
    offside_players: List[Dict] = field(default_factory=list)  # players in offside
    onside_attacking_players: List[Dict] = field(default_factory=list)
    all_players: List[Dict] = field(default_factory=list)  # ALL players with world coords (for minimap)
    is_offside_situation: bool = False           # Whether any attacker is offside
    ball_world_position: Optional[np.ndarray] = None


class OffsideAnalyzer:
    """
    Analyzes each frame to determine offside situations.

    Pipeline per frame:
      1. Convert all detections to world coordinates
      2. Determine attack direction from world coords (using GK position)
      3. Find second-last defender (the offside reference line)
      4. Compare attacking players' positions against the offside line
      5. Generate offside visualization data

    IMPORTANT: Attack direction is determined from WORLD coordinates,
    NOT pixel coordinates (which are unreliable under perspective).
    """

    def __init__(self, transformer: ViewTransformer):
        self.transformer = transformer
        self.field_length = 105.0
        self.field_width = 68.0
        self._attack_direction: AttackDirection = AttackDirection.UNKNOWN
        self._direction_locked: bool = False
        self._direction_votes: List[str] = []  # Accumulated votes before locking
        self._LOCK_THRESHOLD: int = 30  # Frames needed to lock direction
        self._total_frames_analyzed: int = 0
        self.attack_dir_override: Optional[str] = None  # "top_to_bottom" or "bottom_to_top"

    def analyze(self, frame_det: FrameDetection) -> OffsideResult:
        """
        Analyze a single frame's detections for offside.

        Args:
            frame_det: All detections in this frame

        Returns:
            OffsideResult with complete analysis
        """
        result = OffsideResult(frame_idx=frame_det.frame_idx)

        # Step 1: Convert ALL detections to world coordinates FIRST
        field_players = self._get_players_with_world_positions(frame_det)

        if len(field_players) < 2:
            return result

        # Step 2: Determine attack direction from WORLD coordinates
        attack_dir = self._determine_attack_direction(field_players)
        result.attack_direction = attack_dir

        if attack_dir == AttackDirection.UNKNOWN:
            return result

        # Step 3: Group by team
        attackers, defenders = self._separate_teams(field_players, attack_dir)

        # ── DEBUG: frame summary ──
        print(f"\n[Frame {frame_det.frame_idx}] {len(field_players)} players → "
              f"{len(attackers)} ATK + {len(defenders)} DEF "
              f"(dir={attack_dir.value})")

        if len(defenders) == 0:
            print(f"  ⚠ NO defenders found! Returning empty result.")
            return result

        # Step 4: Find second-last defender
        second_last_def = self._find_second_last_defender(defenders, attack_dir)
        if second_last_def is None:
            return result

        # Step 5: Calculate offside line
        offside_y = second_last_def["world_y"]
        result.offside_line_y = offside_y
        result.second_last_defender = second_last_def

        # Step 6: Check attacking players for offside
        offside_players = self._check_offside_players(
            attackers, offside_y, attack_dir
        )
        result.offside_players = offside_players
        result.is_offside_situation = len(offside_players) > 0

        # Step 7: Find onside attacking players
        result.onside_attacking_players = [
            p for p in attackers if p not in offside_players
        ]

        # Step 8: Calculate offside line in pixel coordinates
        result.offside_line_pixels = self._calculate_offside_pixel_line(offside_y)

        # Step 9: Ball position
        if frame_det.ball and frame_det.ball.foot_position is not None:
            ball_world = self.transformer.pixel_to_world(
                frame_det.ball.foot_position[0],
                frame_det.ball.foot_position[1],
            )
            result.ball_world_position = ball_world

        # Debug output every 30 frames
        if frame_det.frame_idx % 30 == 0:
            gk = next((p for p in field_players
                        if p.get("class_name") == "goalkeeper"), None)
            if gk:
                print(f"[Frame {frame_det.frame_idx}] GK world_y={gk['world_y']:.1f}, "
                      f"attack_dir={attack_dir.value}, "
                      f"offside_line_y={offside_y:.1f}")

        # Store ALL players with world coords (for minimap visualization)
        result.all_players = field_players

        return result

    def _determine_attack_direction(
        self, field_players: List[Dict]
    ) -> AttackDirection:
        """
        Determine attack direction from WORLD coordinates.

        Strategy (improved for stability):
        1. Manual override takes priority.
        2. If direction is locked, return locked direction immediately.
        3. During warm-up (first N frames): vote on direction each frame
           based on ALL players' average world_y positions.
        4. After accumulating enough votes: pick the majority direction
           and lock it for the rest of the sequence.

        This avoids the "offside line flying around" caused by per-frame
        GK detection errors flipping the attack direction.
        """
        # Manual override takes priority
        if self.attack_dir_override is not None:
            override = AttackDirection(self.attack_dir_override)
            self._attack_direction = override
            self._direction_locked = True
            return override

        # If already locked, return cached direction
        if self._direction_locked:
            return self._attack_direction

        if len(field_players) < 2:
            self._direction_votes.append("unknown")
            return AttackDirection.UNKNOWN

        # Vote based on ALL players' world_y average
        # The team with smaller avg world_y is closer to y=0 (defending y=0)
        # → the attacking team moves toward y=105 = TOP_TO_BOTTOM
        team_avgs = {}
        for p in field_players:
            t = p.get("team", "unknown")
            if t not in team_avgs:
                team_avgs[t] = []
            team_avgs[t].append(p["world_y"])

        midpoint = self.field_length / 2.0  # 52.5

        if len(team_avgs) >= 2:
            avgs = {t: np.mean(ys) for t, ys in team_avgs.items()}
            sorted_teams = sorted(avgs.items(), key=lambda x: x[1])
            defending_avg_y = avgs[sorted_teams[0][0]]

            if defending_avg_y < midpoint:
                vote = "top_to_bottom"   # defenders near y=0, attack toward y=105
            else:
                vote = "bottom_to_top"   # defenders near y=105, attack toward y=0
        else:
            # Single team: use world_y average vs midpoint
            all_ys = [p["world_y"] for p in field_players]
            avg_y = np.mean(all_ys)
            vote = "top_to_bottom" if avg_y < midpoint else "bottom_to_top"

        self._direction_votes.append(vote)

        # Lock direction after collecting enough frames
        if len(self._direction_votes) >= self._LOCK_THRESHOLD:
            top_votes = self._direction_votes.count("top_to_bottom")
            bottom_votes = self._direction_votes.count("bottom_to_top")

            if top_votes > bottom_votes:
                self._attack_direction = AttackDirection.TOP_TO_BOTTOM
            elif bottom_votes > top_votes:
                self._attack_direction = AttackDirection.BOTTOM_TO_TOP
            else:
                # Tie: use latest vote
                self._attack_direction = AttackDirection(
                    self._direction_votes[-1] if vote != "unknown" else "top_to_bottom"
                )

            self._direction_locked = True
            print(f"[AttackDir] DIRECTION LOCKED after {len(self._direction_votes)} frames: "
                  f"{self._attack_direction.value} "
                  f"(top={top_votes}, bottom={bottom_votes})")
            return self._attack_direction

        # Before lock threshold: use majority vote for direction
        # (allows offside detection during warmup, stabilizes as more votes accumulate)
        top_votes = self._direction_votes.count("top_to_bottom")
        bottom_votes = self._direction_votes.count("bottom_to_top")
        vote = "top_to_bottom" if top_votes >= bottom_votes else "bottom_to_top"
        return AttackDirection(vote)

    def _get_players_with_world_positions(
        self, frame_det: FrameDetection
    ) -> List[Dict]:
        """
        Convert all player detections to world coordinates.
        Excludes referees and players whose position cannot be mapped.

        World coordinates: world_x = width (0-68), world_y = length (0-105).
        These come directly from the homography matrix — no swapping needed.
        """
        players = []

        for p in frame_det.players:
            world = self.transformer.pixel_to_world(
                p.foot_position[0], p.foot_position[1]
            )
            if world is not None:
                players.append({
                    "track_id": p.track_id,
                    "world_x": float(world[0]),
                    "world_y": float(world[1]),
                    "pixel_x": float(p.foot_position[0]),
                    "pixel_y": float(p.foot_position[1]),
                    "team": p.team,
                    "class_name": p.class_name,
                    "confidence": float(p.confidence),
                    "bbox": p.bbox.tolist(),
                })

        # Include goalkeepers
        for p in frame_det.goalkeepers:
            world = self.transformer.pixel_to_world(
                p.foot_position[0], p.foot_position[1]
            )
            if world is not None:
                players.append({
                    "track_id": p.track_id,
                    "world_x": float(world[0]),
                    "world_y": float(world[1]),
                    "pixel_x": float(p.foot_position[0]),
                    "pixel_y": float(p.foot_position[1]),
                    "team": p.team,
                    "class_name": "goalkeeper",
                    "confidence": float(p.confidence),
                    "bbox": p.bbox.tolist(),
                })

        return players

    def _separate_teams(
        self, players: List[Dict], attack_dir: AttackDirection
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Separate players into attacking and defending teams.

        Returns:
            (attackers, defenders) including goalkeepers as defenders
        """
        attackers = []
        defenders = []

        for p in players:
            if p["class_name"] == "goalkeeper":
                defenders.append(p)
            elif p["team"] == "attacking":
                attackers.append(p)
            elif p["team"] == "defending":
                defenders.append(p)
            else:
                # Unknown team — use world position heuristic
                midpoint = self.field_length / 2.0
                if attack_dir == AttackDirection.TOP_TO_BOTTOM:
                    # Attacking toward y=105: players with world_y > midpoint are attackers
                    if p["world_y"] > midpoint:
                        attackers.append(p)
                    else:
                        defenders.append(p)
                else:  # BOTTOM_TO_TOP
                    if p["world_y"] < midpoint:
                        attackers.append(p)
                    else:
                        defenders.append(p)

        return attackers, defenders

    def _find_second_last_defender(
        self, defenders: List[Dict], attack_dir: AttackDirection
    ) -> Optional[Dict]:
        """
        Find the SECOND-LAST DEFENDER (the offside reference line).

        IFAB Law 11: A player is in offside position if they are nearer to the
        opponents' goal line than the second-last opponent.

        The "last defender" is the one CLOSEST to their own goal.
        The "second-last defender" is the one just before that.

        Attack direction determines which goal we measure distance to:
          - TOP_TO_BOTTOM (attacking toward y=105): distance = |world_y - 105|
          - BOTTOM_TO_TOP (attacking toward y=0):  distance = |world_y - 0|
        """
        if len(defenders) < 1:
            return None

        if attack_dir == AttackDirection.TOP_TO_BOTTOM:
            # Attacking toward y=105 (far goal line)
            # Closest to goal = highest world_y → sort DESCENDING
            sorted_defs = sorted(defenders, key=lambda p: -p["world_y"])
        else:
            # Attacking toward y=0 (near goal line)
            # Closest to goal = lowest world_y → sort ASCENDING
            sorted_defs = sorted(defenders, key=lambda p: p["world_y"])

        # ── DEBUG: log all defenders sorted by proximity to goal ──
        self._log_defenders(sorted_defs, attack_dir)

        if len(sorted_defs) >= 2:
            return sorted_defs[1]  # Second-last defender = offside reference line
        else:
            # Only one defender: use them as the reference
            return sorted_defs[0]

    def _log_defenders(self, sorted_defs: List[Dict], attack_dir: AttackDirection):
        """Log defender positions for manual verification."""
        print(f"\n  [Defenders] ({len(sorted_defs)} total, "
              f"attack_dir={attack_dir.value}):")
        goal_y = 105 if attack_dir == AttackDirection.TOP_TO_BOTTOM else 0
        for i, d in enumerate(sorted_defs):
            dist_to_goal = abs(d["world_y"] - goal_y)
            marker = ""
            if i == 0:
                marker = " ← LAST (GK?)"
            elif i == 1:
                marker = " ← SECOND-LAST → OFFSIDE LINE"
            print(f"    [{i}] id=#{d.get('track_id','?')} {d.get('class_name','?')} "
                  f"team={d.get('team','?')} world_y={d['world_y']:.1f} "
                  f"dist_to_goal={dist_to_goal:.1f}{marker}")

    def _check_offside_players(
        self,
        attackers: List[Dict],
        offside_y: float,
        attack_dir: AttackDirection,
    ) -> List[Dict]:
        """
        Determine which attacking players are in offside position.

        A player is in offside position if they are closer to the goal line
        than the offside reference line.

        Note: This only checks position, not involvement in play.
        """
        offside = []

        for p in attackers:
            is_offside = False
            if attack_dir == AttackDirection.TOP_TO_BOTTOM:
                # Attacking toward y=105:
                # Player is offside if world_y > offside_y (closer to y=105)
                if p["world_y"] > offside_y + 0.1:
                    is_offside = True
            else:
                # Attacking toward y=0:
                # Player is offside if world_y < offside_y (closer to y=0)
                if p["world_y"] < offside_y - 0.1:
                    is_offside = True

            if is_offside:
                p_copy = dict(p)
                p_copy["distance_offside"] = abs(p["world_y"] - offside_y)
                offside.append(p_copy)

        return offside

    def _calculate_offside_pixel_line(self, world_y: float) -> Optional[np.ndarray]:
        """
        Calculate the offside line as pixel coordinates for drawing.

        Returns a line from left touchline to right touchline at world_y.
        """
        if world_y is None:
            return None

        # Sample points across the width of the field
        left_pixel = self.transformer.world_to_pixel(0, world_y)
        right_pixel = self.transformer.world_to_pixel(self.field_width, world_y)

        if left_pixel is not None and right_pixel is not None:
            return np.array([left_pixel, right_pixel], dtype=np.float32)

        return None

    def analyze_batch(
        self, frame_detections: List[FrameDetection]
    ) -> List[OffsideResult]:
        """Analyze multiple frames and return all results."""
        results = []
        for fd in frame_detections:
            result = self.analyze(fd)
            results.append(result)
        return results

    def summary(self, results: List[OffsideResult]) -> Dict:
        """Generate a summary of offside analysis across frames."""
        offside_frames = [r for r in results if r.is_offside_situation]
        return {
            "total_frames": len(results),
            "offside_frames": len(offside_frames),
            "offside_frame_indices": [r.frame_idx for r in offside_frames],
            "total_offside_instances": sum(len(r.offside_players) for r in offside_frames),
            "offside_percentage": len(offside_frames) / len(results) * 100 if results else 0,
        }
