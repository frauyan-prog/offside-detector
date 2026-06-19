"""
Player Detection & Tracking Module (Football-Specific YOLOv8)

Uses ultralytics YOLOv8 with a football-specific pretrained model
(football-player-detection.pt) that detects 4 classes:
  ball, goalkeeper, player, referee

Tracking via supervision ByteTrack. Team classification via jersey
color clustering (K-means HSV).

No external API needed — runs 100% locally.

Requires: ultralytics, supervision, opencv-python, numpy, torch
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from ultralytics import YOLO
import supervision as sv


@dataclass
class PlayerDetection:
    """Single player detection in one frame."""
    track_id: int
    class_name: str           # "player", "goalkeeper", "referee"
    bbox: np.ndarray          # [x1, y1, x2, y2] pixel coordinates
    confidence: float
    foot_position: np.ndarray  # (x, y) bottom-center of bbox
    team: str = "unknown"      # "attacking", "defending", "unknown"
    in_offside: bool = False
    world_position: Optional[np.ndarray] = None


@dataclass
class FrameDetection:
    """All detections in a single frame."""
    frame_idx: int
    players: List[PlayerDetection] = field(default_factory=list)
    ball: Optional[PlayerDetection] = None
    goalkeepers: List[PlayerDetection] = field(default_factory=list)
    referees: List[PlayerDetection] = field(default_factory=list)

    def all_players(self) -> List[PlayerDetection]:
        return self.players + self.goalkeepers + self.referees


class PlayerDetector:
    """
    Detects and tracks football players using a football-specific YOLOv8 model.

    The model detects 4 classes directly:
      0: ball, 1: goalkeeper, 2: player, 3: referee

    No heuristic-based GK/referee detection needed — the model handles it.

    Usage:
        detector = PlayerDetector()
        det = detector.detect_and_track(frame, frame_idx)
    """

    # Default model: football-specific YOLOv8 with 4 classes
    # Also supports generic YOLOv8 variants for fallback
    MODEL_VARIANTS = {
        "n": "yolov8n.pt",      # ~6MB, generic person detection
        "s": "yolov8s.pt",      # ~22MB, generic
        "m": "yolov8m.pt",      # ~52MB, generic
        "football": "football-player-detection.pt",  # ~21MB, 4-class
    }

    def __init__(
        self,
        model_size: str = "football",
        confidence: float = 0.25,
        iou_threshold: float = 0.5,
    ):
        """
        Args:
            model_size: "football" (4-class football model), "n", "s", or "m"
            confidence: Minimum detection confidence
            iou_threshold: NMS IoU threshold
        """
        if model_size not in self.MODEL_VARIANTS:
            raise ValueError(f"Model size must be one of {list(self.MODEL_VARIANTS)}")

        model_name = self.MODEL_VARIANTS[model_size]
        self._model_size = model_size

        # Find model file — look in models/ dir, then root, then cwd
        import os
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        local_paths = [
            os.path.join(models_dir, model_name),
            os.path.join(os.path.dirname(models_dir), model_name),
            os.path.join(os.getcwd(), model_name),
        ]
        model_path = None
        for lp in local_paths:
            if os.path.exists(lp):
                model_path = lp
                break

        if model_path is None:
            raise FileNotFoundError(
                f"Model file '{model_name}' not found. Searched: {local_paths}"
            )

        print(f"[PlayerDetector] Loading football model: {model_path}")
        self.model = YOLO(model_path)

        # Try GPU first, fall back to CPU
        try:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"[PlayerDetector] GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                print("[PlayerDetector] No GPU found, using CPU")
        except ImportError:
            self.device = "cpu"
            print("[PlayerDetector] PyTorch not found, using CPU")

        print(f"[PlayerDetector] Model loaded. Classes: {self.model.names}")
        print(f"[PlayerDetector] Device: {self.device}")

        self.confidence = confidence
        self.iou_threshold = iou_threshold

        # ByteTrack tracker from supervision
        self.tracker = sv.ByteTrack(
            track_activation_threshold=confidence,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30,
        )

        # Team classification state
        self.player_team: Dict[int, str] = {}
        self._team_color_samples: List[dict] = []       # [{track_id, hs}] collected over time
        self._team_sample_frames: int = 0               # how many frames sampled
        self._team_sample_max: int = 10                 # sample first N frames before clustering
        self._team_cluster_centers: Dict[str, np.ndarray] = {}  # cluster → mean HS

        # Field boundary filtering (set via set_field_homography)
        self._field_H: Optional[np.ndarray] = None
        self._field_H_inv: Optional[np.ndarray] = None

    def detect_and_track(
        self, frame: np.ndarray, frame_idx: int, detect_teams: bool = True,
    ) -> FrameDetection:
        """
        Detect and track players in a single frame.

        Args:
            frame: BGR video frame
            frame_idx: Current frame index
            detect_teams: Whether to classify teams by jersey color

        Returns:
            FrameDetection with all tracked players
        """
        # Step 1: YOLOv8 inference — detect all 4 classes (ball, goalkeeper, player, referee)
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=None,          # Detect all classes
            verbose=False,
            device=self.device,
        )

        # Step 2: Convert to supervision Detections
        if len(results) == 0 or results[0].boxes is None:
            return FrameDetection(frame_idx=frame_idx)

        detections = sv.Detections.from_ultralytics(results[0])

        if len(detections) == 0:
            return FrameDetection(frame_idx=frame_idx)

        # Step 3: Track with ByteTrack
        detections = self.tracker.update_with_detections(detections)

        # Step 4: Classify players (goalkeeper, referee, outfield)
        classified = self._classify_players(frame, detections, frame_idx)

        # Step 5: Team classification by jersey color
        if detect_teams and len(classified) >= 2:
            self._classify_teams(frame, classified)

        # Step 5.5: Field boundary filtering (if homography set)
        if self._field_H is not None:
            classified = [
                c for c in classified
                if self._is_on_field(np.array([c["x1"], c["y1"], c["x2"], c["y2"]]))
            ]

        # Step 6: Build FrameDetection result
        return self._build_result(classified, frame_idx)

    def _classify_players(
        self, frame: np.ndarray, detections: sv.Detections, frame_idx: int,
    ) -> List[dict]:
        """
        Map YOLO class IDs to player roles.

        Model outputs 4 classes:
          0 → ball, 1 → goalkeeper, 2 → player, 3 → referee

        No heuristics needed — the football-specific model handles it.
        """
        h, w = frame.shape[:2]
        classified = []

        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else 0
            conf = float(detections.confidence[i]) if detections.confidence is not None else 0

            # Map model class ID → role name
            cls_id = int(detections.class_id[i]) if detections.class_id is not None else 2
            yolo_name = self.model.names.get(cls_id, "player")
            class_name = yolo_name  # ball, goalkeeper, player, referee

            classified.append({
                "track_id": track_id,
                "class": class_name,
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "confidence": conf,
            })

        return classified

    def set_field_homography(self, H: np.ndarray):
        """
        Set the homography matrix for field boundary filtering.

        Once set, detections that map to world coordinates outside
        the field (0-120 × 0-70 in RoboFlow coords) will be filtered.
        """
        self._field_H = H.copy() if H is not None else None
        self._field_H_inv = None
        if H is not None and abs(np.linalg.det(H)) > 1e-10:
            self._field_H_inv = np.linalg.inv(H)

    def _is_on_field(self, bbox: np.ndarray) -> bool:
        """
        Check if a bounding box is on the field using homography.

        Transforms the foot position (bottom-center of bbox) to world
        coordinates. Returns True if within field boundaries.
        """
        if self._field_H is None:
            return True  # No homography set, assume on field

        x1, y1, x2, y2 = bbox
        foot_x = (x1 + x2) / 2.0
        foot_y = y2

        pt_pixel = np.array([[foot_x, foot_y]], dtype=np.float32)
        try:
            pt_world = cv2.perspectiveTransform(
                pt_pixel.reshape(-1, 1, 2), self._field_H,
            ).reshape(2)
            wx, wy = pt_world[0], pt_world[1]
            # Field boundaries (RoboFlow coords: 120×70)
            margin = 5.0  # Allow small margin outside field
            return (-margin < wx < 120 + margin and
                    -margin < wy < 70 + margin)
        except Exception:
            return True  # Transform failed, keep the detection

    @staticmethod
    def _extract_jersey_hsv(
        frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
    ) -> Optional[Tuple[float, float, float]]:
        """Extract mean HSV (H, S, V) from the upper body (jersey) region."""
        h, w = frame.shape[:2]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        if x2c <= x1c or y2c <= y1c:
            return None

        # Upper 1/3 of body = jersey region
        mid_y = y1c + (y2c - y1c) // 3
        jersey = frame[y1c:mid_y, x1c:x2c]
        if jersey.size == 0:
            return None

        jersey = cv2.GaussianBlur(jersey, (3, 3), 0)
        hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
        mean_vals = hsv.reshape(-1, 3).mean(axis=0)
        return float(mean_vals[0]), float(mean_vals[1]), float(mean_vals[2])

    def _classify_teams(self, frame: np.ndarray, tracked: List[dict]):
        """
        Classify outfield players into attacking/defending by jersey color.

        Strategy (revised, 2026-06-17):
          Phase 1 (first N frames): Collect jersey color samples from all
            detected players. After enough samples, run K-means to determine
            the two team color clusters. Store cluster centers.
          Phase 2 (after classification): For each unlabeled player or new
            track, assign team based on nearest cluster center in HS space.
          Phase 3 (periodic re-check): If many players are "unknown", re-run
            K-means to adapt to changing lighting or new kits.

        Also detects referees by dark uniform heuristic (low V + low S).
        """
        h, w = frame.shape[:2]

        # ── Phase 1: Collect samples ──
        if self._team_sample_frames < self._team_sample_max:
            for det in tracked:
                if det["class"] != "player":
                    continue
                tid = det["track_id"]
                feat = self._extract_jersey_color(frame, det)
                if feat is not None:
                    # feat = [L, a, b, S, y_center] — store 4D features + y_pos
                    self._team_color_samples.append({
                        "track_id": tid,
                        "features": feat[:4].copy(),  # [L, a, b, S]
                        "y_pos": float(feat[4]),
                    })
            self._team_sample_frames += 1

            # Run K-means when we have enough data
            if self._team_sample_frames >= self._team_sample_max and \
               len(self._team_color_samples) >= 4:
                self._run_team_clustering()

            return  # Still collecting

        # ── Phase 2: Assign teams from cluster centers ──
        if self._team_cluster_centers:
            # Build per-track-id feature list from current frame
            track_features: Dict[int, np.ndarray] = {}
            for det in tracked:
                if det["class"] != "player":
                    continue
                tid = det["track_id"]
                if tid in self.player_team:
                    continue  # Already labeled
                if tid not in track_features:
                    feat = self._extract_jersey_color(frame, det)
                    if feat is not None:
                        track_features[tid] = feat[:4]  # [L, a, b, S]

            # Assign unlabeled players to nearest cluster in 4D feature space
            for tid, feat in track_features.items():
                best_team, best_dist = None, float("inf")
                for team, center in self._team_cluster_centers.items():
                    dist = np.linalg.norm(feat - center)
                    if dist < best_dist:
                        best_dist = dist
                        best_team = team
                # Normalized 4D feature distance threshold (L:100, a:256, b:256, S:255)
                # Max possible distance ≈ sqrt(1²+2²+2²+1²) = sqrt(10) ≈ 3.16
                if best_team and best_dist < 1.5:  # ~half max distance
                    self.player_team[tid] = best_team

            # ── Phase 3: Periodic re-cluster if too many unknowns ──
            n_unknown = sum(
                1 for d in tracked
                if d["class"] == "player" and
                d["track_id"] not in self.player_team
            )
            n_total = sum(1 for d in tracked if d["class"] == "player")
            if n_total >= 4 and n_unknown / max(n_total, 1) > 0.5:
                # Re-sample and re-cluster
                self._team_color_samples.clear()
                self._team_sample_frames = self._team_sample_max - 3
                # Don't clear existing cluster centers yet, but allow override
        else:
            # Clusters not yet formed — keep collecting
            self._team_sample_frames -= 1  # Retry sampling

    def _run_team_clustering(self):
        """Run K-means on LAB+S (4D) color features, assign attacking/defending
        by spatial position (y-center) rather than hue ordering."""
        if len(self._team_color_samples) < 4:
            return

        # 4D features: [L, a, b, S] — normalize per channel
        raw = np.array([s["features"] for s in self._team_color_samples], dtype=np.float32)
        y_pos = np.array([s["y_pos"] for s in self._team_color_samples], dtype=np.float32)
        tids = [s["track_id"] for s in self._team_color_samples]

        # Normalize: L/100, a/128, b/128, S/255  → each roughly in [-1,1] or [0,1]
        features_norm = raw.copy()
        features_norm[:, 0] /= 100.0   # L:  0..100 → 0..1
        features_norm[:, 1] /= 128.0   # a: -128..127 → -1..1
        features_norm[:, 2] /= 128.0   # b: -128..127 → -1..1
        features_norm[:, 3] /= 255.0   # S:  0..255 → 0..1

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
        _, labels, centers = cv2.kmeans(
            features_norm, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS,
        )
        labels = labels.flatten()

        # Determine attacking vs defending by average y-position.
        # In broadcast view, attackers push toward the far goal (higher y in image).
        # Cluster with higher avg y → attacking team.
        cluster_y_avg: Dict[int, float] = {}
        for label_int in [0, 1]:
            mask = labels == label_int
            cluster_y_avg[label_int] = float(y_pos[mask].mean()) if mask.any() else 0.0

        labels_sorted = sorted(cluster_y_avg.keys(),
                               key=lambda l: cluster_y_avg[l])
        # Higher y → attacking
        team_names = {
            labels_sorted[0]: "defending",   # lower avg y (closer to camera)
            labels_sorted[1]: "attacking",   # higher avg y (further from camera)
        }

        # Store cluster centers (normalized 4D)
        for label_int, team_name in team_names.items():
            self._team_cluster_centers[team_name] = centers[label_int].copy()

        # Assign teams to all sampled players
        for i, tid in enumerate(tids):
            self.player_team[tid] = team_names[int(labels[i])]

        # Compute per-team stats for logging
        team_stats = {}
        for label_int, name in team_names.items():
            mask = labels == label_int
            team_stats[name] = {
                "count": int(mask.sum()),
                "L_avg": float(raw[mask, 0].mean()),
                "S_avg": float(raw[mask, 3].mean()),
                "y_avg": float(y_pos[mask].mean()),
            }
        print(f"[TeamClass] LAB+S clustering ({len(tids)} samples): "
              f"ATK={team_stats['attacking']} "
              f"DEF={team_stats['defending']}")

    def _extract_jersey_color(self, frame: np.ndarray, det: dict) -> Optional[np.ndarray]:
        """Extract LAB+S color features + y-center from jersey region.

        Returns: np.ndarray [L, a, b, S, y_center] (5 floats) or None.
        L∈[0,100], a∈[-128,127], b∈[-128,127], S∈[0,255], y_center in pixels.
        """
        x1, y1, x2, y2 = map(int, [det["x1"], det["y1"], det["x2"], det["y2"]])
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        # Upper 1/3 of body = jersey region
        mid_y = y1 + (y2 - y1) // 3
        jersey = frame[y1:mid_y, x1:x2]
        if jersey.size == 0:
            return None

        jersey = cv2.GaussianBlur(jersey, (3, 3), 0)

        # HSV → S (saturation, 0-255)
        hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
        mean_s = float(hsv[:, :, 1].mean())

        # LAB → L (0-100), a (-128,127), b (-128,127)
        lab = cv2.cvtColor(jersey, cv2.COLOR_BGR2LAB)
        mean_vals = lab.reshape(-1, 3).mean(axis=0)
        mean_l = float(mean_vals[0])
        mean_a = float(mean_vals[1])
        mean_b = float(mean_vals[2])

        # y-center of bounding box (for spatial heuristic)
        y_center = float((y1 + y2) / 2.0)

        return np.array([mean_l, mean_a, mean_b, mean_s, y_center], dtype=np.float32)

    def _build_result(self, tracked: List[dict], frame_idx: int) -> FrameDetection:
        """Build FrameDetection from tracked detections."""
        result = FrameDetection(frame_idx=frame_idx)

        for det in tracked:
            bbox = np.array(
                [det["x1"], det["y1"], det["x2"], det["y2"]], dtype=np.float32
            )
            foot = np.array(
                [(det["x1"] + det["x2"]) / 2, det["y2"]], dtype=np.float32
            )

            player = PlayerDetection(
                track_id=det.get("track_id", 0),
                class_name=det["class"],
                bbox=bbox,
                confidence=det.get("confidence", 0),
                foot_position=foot,
                team=self.player_team.get(det.get("track_id", 0), "unknown"),
            )

            cls = det["class"]
            if cls == "goalkeeper":
                result.goalkeepers.append(player)
            elif cls == "referee":
                result.referees.append(player)
            else:
                result.players.append(player)

        return result

    def reset(self):
        """Reset tracker and team assignments (for new video)."""
        self.tracker.reset()
        self.player_team.clear()
        self._team_color_samples.clear()
        self._team_sample_frames = 0
        self._team_cluster_centers.clear()
        self._field_H = None
        self._field_H_inv = None
