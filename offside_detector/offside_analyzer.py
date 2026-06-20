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
import os

from .player_detector import PlayerDetection, FrameDetection
from .view_transformer import ViewTransformer


class AttackDirection(Enum):
    """Which direction the attacking team is playing."""
    TOP_TO_BOTTOM = "top_to_bottom"    # attacking toward y=105 (goal at bottom)
    BOTTOM_TO_TOP = "bottom_to_top"    # attacking toward y=0 (goal at top)
    LEFT_TO_RIGHT = "left_to_right"    # attacking toward x=68 (goal at right)
    RIGHT_TO_LEFT = "right_to_left"    # attacking toward x=0 (goal at left)
    UNKNOWN = "unknown"


def _is_horizontal_attack(attack_dir: AttackDirection) -> bool:
    """Check if attack direction is horizontal (field length along image X axis)."""
    return attack_dir in (AttackDirection.LEFT_TO_RIGHT, AttackDirection.RIGHT_TO_LEFT)


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

    def _debug_log(self, msg: str):
        """Append a debug message to output/debug_log.txt."""
        log_path = os.path.join(os.path.dirname(__file__), "..", "output", "debug_log.txt")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

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

        # Step 3: Fix team assignments for horizontal fields
        # K-means uses y-position (vertical assumption), swap if horizontal
        if _is_horizontal_attack(attack_dir):
            self._fix_horizontal_teams(field_players, attack_dir)

        # Step 4: Group by team
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
        if _is_horizontal_attack(attack_dir):
            offside_y = second_last_def["world_x"]  # length axis is X for horizontal
        else:
            offside_y = second_last_def["world_y"]   # length axis is Y for vertical
        result.offside_line_y = offside_y
        result.second_last_defender = second_last_def

        # Step 6: Ball position (must retrieve BEFORE checking offside)
        if frame_det.ball and frame_det.ball.foot_position is not None:
            ball_world = self.transformer.pixel_to_world(
                frame_det.ball.foot_position[0],
                frame_det.ball.foot_position[1],
            )
            result.ball_world_position = ball_world

        # Step 7: Check attacking players for offside
        # P0-FIX: Pass ball_world_position so IFAB Law 11 "nearer than BOTH ball AND second-last opponent" is enforced
        offside_players = self._check_offside_players(
            attackers, offside_y, attack_dir, result.ball_world_position
        )
        result.offside_players = offside_players
        result.is_offside_situation = len(offside_players) > 0

        # Step 8: Find onside attacking players
        result.onside_attacking_players = [
            p for p in attackers if p not in offside_players
        ]

        # Step 9: Calculate offside line in pixel coordinates
        result.offside_line_pixels = self._calculate_offside_pixel_line(offside_y)

        # Debug output every 30 frames
        if frame_det.frame_idx % 30 == 0:
            gk = next((p for p in field_players
                        if p.get("class_name") == "goalkeeper"), None)
            ball_info = ""
            if result.ball_world_position is not None:
                ball_info = (f"ball=({result.ball_world_position[0]:.1f},"
                            f"{result.ball_world_position[1]:.1f})")
            if gk:
                print(f"[Frame {frame_det.frame_idx}] GK world_y={gk['world_y']:.1f}, "
                      f"attack_dir={attack_dir.value}, "
                      f"offside_line_y={offside_y:.1f} {ball_info}")

        # Store ALL players with world coords (for minimap visualization)
        result.all_players = field_players

        return result

    def _determine_attack_direction(
        self, field_players: List[Dict]
    ) -> AttackDirection:
        """
        Determine attack direction from WORLD coordinates.

        Strategy (improved for stability, P0-FIX added horizontal support):
        1. Manual override takes priority.
        2. If direction is locked, return locked direction immediately.
        3. During warm-up (first N frames): detect field orientation first
           (vertical = goals at y=0 and y=105, horizontal = goals at x=0 and x=68)
           by comparing player spread in world_x vs world_y.
        4. Vote on direction each frame based on player distribution.
        5. After accumulating enough votes: pick the majority direction
           and lock it for the rest of the sequence.

        This avoids the "offside line flying around" caused by per-frame
        GK detection errors flipping the attack direction.
        """
        # Manual override takes priority
        if self.attack_dir_override is not None:
            override = AttackDirection(self.attack_dir_override)
            self._attack_direction = override
            self._direction_locked = True
            # Log only once per run (on first call)
            if self._total_frames_analyzed == 0:
                print(f"[AttackDir] MANUAL OVERRIDE: {override.value}")
                self._debug_log(f"[AttackDir] MANUAL OVERRIDE: {override.value}")
            self._total_frames_analyzed += 1
            return override

        self._total_frames_analyzed += 1

        # If already locked, return cached direction
        if self._direction_locked:
            return self._attack_direction

        if len(field_players) < 2:
            self._direction_votes.append("unknown")
            return AttackDirection.UNKNOWN

        # P0-FIX: Detect field orientation by comparing player spread in each axis
        # In a VERTICAL field (goals at y=0 and y=105), players spread along y-axis.
        # In a HORIZONTAL field (goals at x=0 and x=68), players spread along x-axis.
        all_wx = np.array([p["world_x"] for p in field_players])
        all_wy = np.array([p["world_y"] for p in field_players])
        std_x = float(np.std(all_wx))
        std_y = float(np.std(all_wy))

        # If players are spread much more along X than Y, it's a horizontal field
        is_horizontal = (std_x > std_y * 1.5 and std_x > 10.0)

        self._debug_log(
            f"[AttackDir] Frame orientation: std_x={std_x:.1f} std_y={std_y:.1f} "
            f"→ {'HORIZONTAL' if is_horizontal else 'VERTICAL'} field"
        )

        if is_horizontal:
            # Horizontal field: goals at x=0 and x=68
            # Determine which goal the attacking team is heading toward
            # Group by team and compute average world_x
            team_avgs = {}
            for p in field_players:
                t = p.get("team", "unknown")
                if t not in team_avgs:
                    team_avgs[t] = []
                team_avgs[t].append(p["world_x"])

            midpoint = self.field_width / 2.0  # 34.0

            if len(team_avgs) >= 2:
                avgs = {t: np.mean(xs) for t, xs in team_avgs.items()}
                sorted_teams = sorted(avgs.items(), key=lambda x: x[1])
                defending_avg_x = avgs[sorted_teams[0][0]]

                if defending_avg_x > midpoint:
                    vote = "left_to_right"   # defenders near x=68 (their own goal), attack from left
                else:
                    vote = "right_to_left"   # defenders near x=0 (their own goal), attack from right
            else:
                # Single team: use world_x average vs midpoint
                avg_x = float(np.mean(all_wx))
                vote = "left_to_right" if avg_x < midpoint else "right_to_left"
        else:
            # Vertical field: goals at y=0 and y=105 (original logic)
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
                avg_y = float(np.mean(all_wy))
                vote = "top_to_bottom" if avg_y < midpoint else "bottom_to_top"

        self._direction_votes.append(vote)

        # Lock direction after collecting enough frames
        if len(self._direction_votes) >= self._LOCK_THRESHOLD:
            top_votes = self._direction_votes.count("top_to_bottom")
            bottom_votes = self._direction_votes.count("bottom_to_top")
            left_votes = self._direction_votes.count("left_to_right")
            right_votes = self._direction_votes.count("right_to_left")

            # Find the most voted direction
            vote_counts = {
                "top_to_bottom": top_votes,
                "bottom_to_top": bottom_votes,
                "left_to_right": left_votes,
                "right_to_left": right_votes,
            }
            max_votes = max(vote_counts.values())
            winners = [d for d, c in vote_counts.items() if c == max_votes]

            if len(winners) == 1:
                self._attack_direction = AttackDirection(winners[0])
            else:
                # Tie: use latest vote among the tied directions
                latest_winner = None
                for v in reversed(self._direction_votes):
                    if v in winners:
                        latest_winner = v
                        break
                self._attack_direction = AttackDirection(
                    latest_winner if latest_winner else "top_to_bottom"
                )

            self._direction_locked = True
            self._debug_log(
                f"[AttackDir] DIRECTION LOCKED after {len(self._direction_votes)} frames: "
                f"{self._attack_direction.value} "
                f"(top={top_votes}, bottom={bottom_votes}, left={left_votes}, right={right_votes})"
            )
            print(f"[AttackDir] DIRECTION LOCKED after {len(self._direction_votes)} frames: "
                  f"{self._attack_direction.value} "
                  f"(top={top_votes}, bottom={bottom_votes}, left={left_votes}, right={right_votes})")
            return self._attack_direction

        # Before lock threshold: use majority vote for direction
        vote_counts = {}
        for v in self._direction_votes:
            vote_counts[v] = vote_counts.get(v, 0) + 1
        max_votes = max(vote_counts.values())
        winners = [d for d, c in vote_counts.items() if c == max_votes]
        vote = winners[0] if winners else "top_to_bottom"
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
                px, py = float(p.foot_position[0]), float(p.foot_position[1])
                wx, wy = float(world[0]), float(world[1])
                players.append({
                    "track_id": p.track_id,
                    "world_x": wx,
                    "world_y": wy,
                    "pixel_x": px,
                    "pixel_y": py,
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

    def _fix_horizontal_teams(self, field_players: List[Dict], attack_dir: AttackDirection):
        """
        K-means jersey classifier assumes vertical field (uses y-position to
        assign attacking/defending). For horizontal fields, we swap teams
        based on player's world_x position relative to the midfield.

        This corrects misclassifications like white-jersey defenders being
        labeled as "attacking" because they're higher up in the image (y-axis)
        but are actually on the defending side of a horizontal field (x-axis).
        """
        swapped = 0
        for p in field_players:
            team = p.get("team", "unknown")
            if team not in ("attacking", "defending"):
                continue
            world_x = p.get("world_x", 34.0)
            midpoint = self.field_width / 2.0  # 34.0

            # For LEFT_TO_RIGHT: defenders should be at small world_x (near x=0),
            # attackers at large world_x (near x=68)
            # For LEFT_TO_RIGHT: defenders protect right goal (x=68), so defenders
            # are at LARGE world_x. Attackers push from left (small world_x).
            if attack_dir == AttackDirection.LEFT_TO_RIGHT:
                correct_team = "defending" if world_x > midpoint else "attacking"
            else:  # RIGHT_TO_LEFT: defenders protect left goal (x=0)
                correct_team = "defending" if world_x < midpoint else "attacking"

            if team != correct_team:
                p["team"] = correct_team
                swapped += 1

        if swapped > 0:
            self._debug_log(f"[TeamFix] Horizontal field: swapped {swapped} players")
            print(f"  [TeamFix] Horizontal field ({attack_dir.value}): "
                  f"swapped {swapped}/{len(field_players)} player teams")

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
                # Unknown team — use position heuristic
                # Prefer pixel position when world coords may be unreliable
                if _is_horizontal_attack(attack_dir):
                    midpoint = self.field_width / 2.0  # 34
                    # Check if world_x is reliable
                    wx = p.get("world_x", 0)
                    px = p.get("pixel_x", 0)
                    img_mid = px  # placeholder - we use pixel midpoint heuristic
                    # For left_to_right: right half of image = attackers (near goal they attack)
                    # Use pixel_x as fallback when world_x is clearly wrong
                    use_pixel = abs(wx) > 100 or wx < -10  # OOB detection
                    if attack_dir == AttackDirection.LEFT_TO_RIGHT:
                        if use_pixel:
                            # Image is ~1184 wide; attackers on right side
                            if px > 1184 / 2:
                                attackers.append(p)
                            else:
                                defenders.append(p)
                        elif p["world_x"] > midpoint:
                            attackers.append(p)
                        else:
                            defenders.append(p)
                    else:  # RIGHT_TO_LEFT
                        if p["world_x"] < midpoint:
                            attackers.append(p)
                        else:
                            defenders.append(p)
                else:
                    midpoint = self.field_length / 2.0  # 52.5
                    if attack_dir == AttackDirection.TOP_TO_BOTTOM:
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

        Attack direction determines which axis and goal we measure distance to:
          - TOP_TO_BOTTOM (attacking toward y=105):  sort by world_y DESCENDING
          - BOTTOM_TO_TOP (attacking toward y=0):    sort by world_y ASCENDING
          - LEFT_TO_RIGHT (attacking toward x=68):   sort by world_x DESCENDING
          - RIGHT_TO_LEFT (attacking toward x=0):     sort by world_x ASCENDING

        FALLBACK: If world coordinates are clearly unreliable (many defenders
        have coords outside field bounds), fall back to PIXEL-POSITION sorting.
        This handles the case where a 2-point similarity transform produces
        garbage for points far from calibration region.
        """
        if len(defenders) < 1:
            return None

        # ── Determine if world coords are reliable ──
        use_pixel_fallback = self._should_use_pixel_fallback(defenders, attack_dir)

        if _is_horizontal_attack(attack_dir):
            if use_pixel_fallback:
                if attack_dir == AttackDirection.LEFT_TO_RIGHT:
                    # Defending goal at x=68. Last defender = highest pixel_x.
                    # Sort DESCENDING: [last_def, 2nd_last, ..., farthest_from_goal]
                    sorted_defs = sorted(defenders, key=lambda p: -p.get("pixel_x", 0))
                else:  # RIGHT_TO_LEFT: defending goal at x=0
                    sorted_defs = sorted(defenders, key=lambda p: p.get("pixel_x", 0))
            else:
                if attack_dir == AttackDirection.LEFT_TO_RIGHT:
                    sorted_defs = sorted(defenders, key=lambda p: -p["world_x"])
                else:  # RIGHT_TO_LEFT
                    sorted_defs = sorted(defenders, key=lambda p: p["world_x"])
        else:
            # Vertical attack: field length along world Y axis
            if use_pixel_fallback:
                if attack_dir == AttackDirection.TOP_TO_BOTTOM:
                    sorted_defs = sorted(defenders, key=lambda p: -p.get("pixel_y", 0))
                else:
                    sorted_defs = sorted(defenders, key=lambda p: p.get("pixel_y", 0))
            else:
                if attack_dir == AttackDirection.TOP_TO_BOTTOM:
                    sorted_defs = sorted(defenders, key=lambda p: -p["world_y"])
                else:
                    sorted_defs = sorted(defenders, key=lambda p: p["world_y"])

        # DEBUG: log all defenders sorted by proximity to goal
        self._log_defenders(sorted_defs, attack_dir, use_pixel_fallback)

        if len(sorted_defs) >= 2:
            return sorted_defs[1]  # Second-last defender = offside reference line
        else:
            return sorted_defs[0]

    def _should_use_pixel_fallback(
        self, defenders: List[Dict], attack_dir: AttackDirection
    ) -> bool:
        """
        Check if world coordinates are reliable enough for sorting.

        Returns True if too many defenders have out-of-bounds world coordinates,
        indicating the homography is inaccurate far from calibration points.

        Thresholds: world_x must be in [0, 68], world_y in [0, 105].
        If >40% of defenders are out of bounds, use pixel fallback.
        """
        if len(defenders) == 0:
            return False

        out_of_bounds_count = 0
        for d in defenders:
            wx = d.get("world_x")
            wy = d.get("world_y")
            if wx is None or wy is None:
                out_of_bounds_count += 1
                continue
            # Check if within field bounds (with small tolerance)
            if _is_horizontal_attack(attack_dir):
                if wx < -5 or wx > 73:  # tolerance of 5m beyond field edge
                    out_of_bounds_count += 1
            else:
                if wy < -5 or wy > 110:
                    out_of_bounds_count += 1

        ratio = out_of_bounds_count / len(defenders)
        use_fallback = ratio > 0.4  # >40% OOB → use pixel fallback

        if use_fallback:
            self._debug_log(f"[DefenderSort] PIXEL FALLBACK activated: "
                           f"{out_of_bounds_count}/{len(defenders)} defenders "
                           f"OOB ({ratio*100:.0f}%)")

        return use_fallback

    def _log_defenders(self, sorted_defs: List[Dict], attack_dir: AttackDirection,
                       use_pixel_fallback: bool = False):
        """Log defender positions for manual verification."""
        mode_label = "PIXEL" if use_pixel_fallback else "WORLD"
        print(f"\n  [Defenders] ({len(sorted_defs)} total, "
              f"attack_dir={attack_dir.value}, sort={mode_label}):")
        horizontal = _is_horizontal_attack(attack_dir)
        if use_pixel_fallback:
            if horizontal:
                axis_name = "pixel_x"
                goal_val = 999999 if attack_dir == AttackDirection.LEFT_TO_RIGHT else 0
            else:
                axis_name = "pixel_y"
                goal_val = 999999 if attack_dir == AttackDirection.TOP_TO_BOTTOM else 0
        else:
            if horizontal:
                goal_val = 68 if attack_dir == AttackDirection.LEFT_TO_RIGHT else 0
                axis_name = "world_x"
            else:
                goal_val = 105 if attack_dir == AttackDirection.TOP_TO_BOTTOM else 0
                axis_name = "world_y"
        for i, d in enumerate(sorted_defs):
            pos_val = d.get(axis_name, 0)
            # Also show world coords for reference when using pixel fallback
            extra_info = ""
            if use_pixel_fallback:
                extra_info = f" (world={d.get('world_x',0):.1f},{d.get('world_y',0):.1f})"
            dist_to_goal = abs(pos_val - goal_val) if goal_val < 999999 else pos_val
            marker = ""
            if i == 0:
                marker = " <- LAST (GK?)"
            elif i == 1:
                marker = " <- SECOND-LAST -> OFFSIDE LINE"
            print(f"    [{i}] id=#{d.get('track_id','?')} {d.get('class_name','?')} "
                  f"{axis_name}={pos_val:.1f}{extra_info} "
                  f"dist={dist_to_goal:.1f}{marker}")

    def _check_offside_players(
        self,
        attackers: List[Dict],
        offside_y: float,
        attack_dir: AttackDirection,
        ball_world_pos: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """
        Determine which attacking players are in offside position.

        IFAB Law 11: A player is in offside position if they are nearer to
        the opponents' goal line than BOTH the ball AND the second-last opponent.

        For horizontal attacks, offside_y actually contains world_x value.

        Args:
            attackers: List of attacking player dicts with world_x, world_y
            offside_y: Offside line position (world_y or world_x depending on direction)
            attack_dir: Direction of attack
            ball_world_pos: [wx, wy] world position of the ball, or None if unavailable
        """
        offside = []
        horizontal = _is_horizontal_attack(attack_dir)

        # P0-FIX: Determine ball's position along the attack axis
        ball_pos_along = None
        if ball_world_pos is not None:
            if horizontal:
                ball_pos_along = ball_world_pos[0]  # world_x for horizontal attacks
            else:
                ball_pos_along = ball_world_pos[1]  # world_y for vertical attacks

        for p in attackers:
            # Check 1: Is player beyond the offside line (closer to opponent's goal)?
            beyond_offside_line = False
            player_pos_along = p["world_x"] if horizontal else p["world_y"]

            if horizontal:
                if attack_dir == AttackDirection.LEFT_TO_RIGHT:
                    # Attacking toward x=68: offside if world_x > offside_line
                    if p["world_x"] > offside_y + 0.1:
                        beyond_offside_line = True
                else:  # RIGHT_TO_LEFT
                    # Attacking toward x=0: offside if world_x < offside_line
                    if p["world_x"] < offside_y - 0.1:
                        beyond_offside_line = True
            else:
                if attack_dir == AttackDirection.TOP_TO_BOTTOM:
                    # Attacking toward y=105: offside if world_y > offside_y
                    if p["world_y"] > offside_y + 0.1:
                        beyond_offside_line = True
                else:  # BOTTOM_TO_TOP
                    # Attacking toward y=0: offside if world_y < offside_y
                    if p["world_y"] < offside_y - 0.1:
                        beyond_offside_line = True

            if not beyond_offside_line:
                continue

            # Check 2: Is player also beyond the BALL?
            # (IFAB: must be nearer than BOTH ball AND second-last opponent)
            if ball_pos_along is not None:
                beyond_ball = False
                if horizontal:
                    if attack_dir == AttackDirection.LEFT_TO_RIGHT:
                        # Attacking toward x=68: ball must be behind player (smaller x)
                        if player_pos_along > ball_pos_along + 0.1:
                            beyond_ball = True
                    else:  # RIGHT_TO_LEFT
                        # Attacking toward x=0: ball must be behind player (larger x)
                        if player_pos_along < ball_pos_along - 0.1:
                            beyond_ball = True
                else:
                    if attack_dir == AttackDirection.TOP_TO_BOTTOM:
                        # Attacking toward y=105: ball behind = smaller y
                        if player_pos_along > ball_pos_along + 0.1:
                            beyond_ball = True
                    else:  # BOTTOM_TO_TOP
                        # Attacking toward y=0: ball behind = larger y
                        if player_pos_along < ball_pos_along - 0.1:
                            beyond_ball = True

                if not beyond_ball:
                    # Player is beyond offside line but NOT beyond ball → NOT offside
                    self._debug_log(
                        f"[OffsideCheck] id=#{p.get('track_id','?')} "
                        f"beyond_line={beyond_offside_line} but "
                        f"NOT beyond ball (player={player_pos_along:.1f}, "
                        f"ball={ball_pos_along:.1f}, dir={attack_dir.value})"
                    )
                    continue

            # Both conditions met → offside position
            p_copy = dict(p)
            if horizontal:
                p_copy["distance_offside"] = abs(p["world_x"] - offside_y)
            else:
                p_copy["distance_offside"] = abs(p["world_y"] - offside_y)
            offside.append(p_copy)

            # Debug log
            ball_info = f"ball={ball_pos_along:.1f}" if ball_pos_along is not None else "ball=N/A"
            self._debug_log(
                f"[OffsideCheck] OFFSIDE id=#{p.get('track_id','?')} "
                f"player_pos={player_pos_along:.1f} offside_line={offside_y:.1f} "
                f"{ball_info} dist={p_copy['distance_offside']:.1f}m"
            )

        return offside

    def _calculate_offside_pixel_line(self, offside_pos: float) -> Optional[np.ndarray]:
        """
        Calculate the offside line as pixel coordinates for drawing.

        For vertical attacks (TOP_TO_BOTTOM/BOTTOM_TO_TOP):
          Returns a line from left touchline (x=0) to right touchline (x=68)
          at the given world_y position.
        For horizontal attacks (LEFT_TO_RIGHT/RIGHT_TO_LEFT):
          Returns a line from bottom goal-line (y=0) to top goal-line (y=105)
          at the given world_x position.
        """
        if offside_pos is None:
            return None

        attack_dir = self._attack_direction
        if _is_horizontal_attack(attack_dir):
            # Horizontal attack: draw vertical offside line (parallel to goal lines)
            top_pixel = self.transformer.world_to_pixel(offside_pos, 0)
            bot_pixel = self.transformer.world_to_pixel(offside_pos, self.field_length)
            if top_pixel is not None and bot_pixel is not None:
                return np.array([top_pixel, bot_pixel], dtype=np.float32)
        else:
            # Vertical attack: draw horizontal offside line (parallel to goal lines)
            left_pixel = self.transformer.world_to_pixel(0, offside_pos)
            right_pixel = self.transformer.world_to_pixel(self.field_width, offside_pos)
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
