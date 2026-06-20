"""
P0 Fix Verification Script - No external model dependencies.
Tests:
  1. IFAB Law 11: ball comparison in offside checking
  2. Horizontal attack direction auto-detection
Uses sys.modules injection to bypass ultralytics import.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

# Inject mock player_detector BEFORE any import of offside_analyzer
@dataclass
class PlayerDetection:
    track_id: int
    class_name: str = "player"
    bbox: np.ndarray = field(default_factory=lambda: np.array([0, 0, 100, 200]))
    confidence: float = 0.8
    foot_position: np.ndarray = field(default_factory=lambda: np.array([50, 200]))
    team: str = "unknown"
    in_offside: bool = False
    world_position: Optional[np.ndarray] = None

@dataclass
class FrameDetection:
    frame_idx: int
    players: List[PlayerDetection] = field(default_factory=list)
    ball: Optional[PlayerDetection] = None
    goalkeepers: List[PlayerDetection] = field(default_factory=list)
    referees: List[PlayerDetection] = field(default_factory=list)

import types
mock_pd = types.ModuleType("offside_detector.player_detector")
mock_pd.PlayerDetection = PlayerDetection
mock_pd.FrameDetection = FrameDetection
sys.modules["offside_detector.player_detector"] = mock_pd

# Mock ViewTransformer
class MockTransformer:
    def __init__(self, H=None):
        if H is None:
            self.homography = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]], dtype=np.float32)
        else:
            self.homography = H
        self.inverse_homography = np.linalg.inv(self.homography)
        self.field_width = 68.0
        self.field_length = 105.0

    @property
    def is_ready(self):
        return True

    def pixel_to_world(self, x, y):
        import cv2
        pt = np.array([[[x, y]]], dtype=np.float32)
        return cv2.perspectiveTransform(pt, self.homography)[0, 0]

    def world_to_pixel(self, wx, wy):
        import cv2
        pt = np.array([[[wx, wy]]], dtype=np.float32)
        return cv2.perspectiveTransform(pt, self.inverse_homography)[0, 0]

from offside_detector.offside_analyzer import (
    OffsideAnalyzer, OffsideResult, AttackDirection, _is_horizontal_attack
)


def test_ball_comparison():
    """P0-1: Verify ball position is checked in offside decisions."""
    print("=" * 70)
    print("TEST P0-1: Ball Comparison in Offside Checking")
    print("=" * 70)

    transformer = MockTransformer()
    analyzer = OffsideAnalyzer(transformer)
    analyzer.attack_dir_override = "top_to_bottom"

    # TOP_TO_BOTTOM: defending goal at y=0, attacking toward y=105
    # Defenders at world_y=55 (GK, last) and world_y=45 (2nd-last -> offside line=45)
    # Ball at world_y=55 (beyond offside line)
    # pixel_to_world(px, py) = (px*0.1, py*0.1)

    defenders = [
        PlayerDetection(track_id=1, class_name="goalkeeper", team="defending",
                        foot_position=np.array([300, 550])),  # world=(30,55) last def
        PlayerDetection(track_id=2, class_name="player", team="defending",
                        foot_position=np.array([300, 450])),  # world=(30,45) 2nd-last -> line
    ]

    attackers = [
        # track_id=10: world_y=50 -> beyond line(45) but NOT beyond ball(55) -> NOT offside
        PlayerDetection(track_id=10, class_name="player", team="attacking",
                        foot_position=np.array([300, 500])),
        # track_id=11: world_y=60 -> beyond line(45) AND beyond ball(55) -> OFFSIDE
        PlayerDetection(track_id=11, class_name="player", team="attacking",
                        foot_position=np.array([300, 600])),
        # track_id=12: world_y=40 -> before line(45) -> NOT offside
        PlayerDetection(track_id=12, class_name="player", team="attacking",
                        foot_position=np.array([300, 400])),
    ]

    ball = PlayerDetection(track_id=99, class_name="ball",
                           foot_position=np.array([300, 550]))  # world_y=55

    frame_det = FrameDetection(
        frame_idx=0, players=attackers, ball=ball,
        goalkeepers=defenders[:1], referees=[]
    )
    frame_det.players.append(defenders[1])

    result = analyzer.analyze(frame_det)

    print(f"\n  Attack direction: {result.attack_direction.value}")
    print(f"  Offside line (world_y): {result.offside_line_y}")
    if result.ball_world_position is not None:
        bw = result.ball_world_position
        print(f"  Ball world position: ({bw[0]:.1f}, {bw[1]:.1f})")
    offside_ids = [p["track_id"] for p in result.offside_players]
    print(f"  Offside players: {offside_ids}")

    errors = []
    if 11 not in offside_ids:
        errors.append("track_id=11 (beyond both line & ball) should be OFFSIDE")
    if 10 in offside_ids:
        errors.append("track_id=10 (beyond line but NOT ball) should be NOT offside")
    if 12 in offside_ids:
        errors.append("track_id=12 (before line) should be NOT offside")

    if errors:
        print("\n  [FAILED]:")
        for e in errors:
            print(f"    - {e}")
        return False
    else:
        print("\n  [PASSED]: Ball comparison works correctly")
        return True


def test_horizontal_attack_detection():
    """P0-2: Verify horizontal attack direction is auto-detected."""
    print("\n" + "=" * 70)
    print("TEST P0-2: Horizontal Attack Direction Auto-Detection")
    print("=" * 70)

    transformer = MockTransformer()
    analyzer = OffsideAnalyzer(transformer)
    analyzer._direction_locked = False
    analyzer._direction_votes = []
    analyzer.attack_dir_override = None

    horizontal_players = []
    for i in range(6):
        horizontal_players.append({
            "track_id": i + 1,
            "world_x": 10.0 + np.random.uniform(-2, 2),
            "world_y": 35.0 + np.random.uniform(-10, 10),
            "pixel_x": 100.0, "pixel_y": 350.0,
            "team": "defending" if i < 3 else "attacking",
            "class_name": "goalkeeper" if i == 0 else "player",
            "confidence": 0.9, "bbox": [0, 0, 100, 200],
        })
    for i in range(6):
        horizontal_players.append({
            "track_id": i + 10,
            "world_x": 55.0 + np.random.uniform(-2, 2),
            "world_y": 35.0 + np.random.uniform(-10, 10),
            "pixel_x": 550.0, "pixel_y": 350.0,
            "team": "attacking", "class_name": "player",
            "confidence": 0.9, "bbox": [0, 0, 100, 200],
        })

    locked = None
    for _ in range(30):
        locked = analyzer._determine_attack_direction(horizontal_players)

    print(f"\n  std_x = {np.std([p['world_x'] for p in horizontal_players]):.1f} (expected large)")
    print(f"  std_y = {np.std([p['world_y'] for p in horizontal_players]):.1f} (expected small)")
    print(f"  Locked direction: {locked.value}")

    if locked in (AttackDirection.LEFT_TO_RIGHT, AttackDirection.RIGHT_TO_LEFT):
        print("\n  [PASSED]: Horizontal attack direction detected correctly")
        return True
    else:
        print(f"\n  [FAILED]: Expected horizontal, got {locked.value}")
        return False


def test_vertical_attack_still_works():
    """Verify vertical attack direction still works after P0-2 changes."""
    print("\n" + "=" * 70)
    print("TEST P0-2b: Vertical Attack Direction Still Works")
    print("=" * 70)

    transformer = MockTransformer()
    analyzer = OffsideAnalyzer(transformer)
    analyzer._direction_locked = False
    analyzer._direction_votes = []
    analyzer.attack_dir_override = None

    vertical_players = []
    for i in range(6):
        vertical_players.append({
            "track_id": i + 1,
            "world_x": 34.0 + np.random.uniform(-10, 10),
            "world_y": 10.0 + np.random.uniform(-5, 5),
            "pixel_x": 340.0, "pixel_y": 100.0,
            "team": "defending" if i < 3 else "attacking",
            "class_name": "goalkeeper" if i == 0 else "player",
            "confidence": 0.9, "bbox": [0, 0, 100, 200],
        })
    for i in range(6):
        vertical_players.append({
            "track_id": i + 10,
            "world_x": 34.0 + np.random.uniform(-10, 10),
            "world_y": 95.0 + np.random.uniform(-5, 5),
            "pixel_x": 340.0, "pixel_y": 950.0,
            "team": "attacking", "class_name": "player",
            "confidence": 0.9, "bbox": [0, 0, 100, 200],
        })

    locked = None
    for _ in range(30):
        locked = analyzer._determine_attack_direction(vertical_players)

    print(f"\n  std_x = {np.std([p['world_x'] for p in vertical_players]):.1f} (expected small)")
    print(f"  std_y = {np.std([p['world_y'] for p in vertical_players]):.1f} (expected large)")
    print(f"  Locked direction: {locked.value}")

    if locked in (AttackDirection.TOP_TO_BOTTOM, AttackDirection.BOTTOM_TO_TOP):
        print("\n  [PASSED]: Vertical attack direction still works")
        return True
    else:
        print(f"\n  [FAILED]: Expected vertical, got {locked.value}")
        return False


if __name__ == "__main__":
    results = []
    results.append(("P0-1: Ball Comparison", test_ball_comparison()))
    results.append(("P0-2: Horizontal Attack Detection", test_horizontal_attack_detection()))
    results.append(("P0-2b: Vertical Attack Still Works", test_vertical_attack_still_works()))

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}]  {name}")

    if all_pass:
        print("\n  ALL P0 FIXES VERIFIED!")
    else:
        print("\n  SOME TESTS FAILED - review output above")

    sys.exit(0 if all_pass else 1)