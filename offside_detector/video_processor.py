"""
Video Processor Module

Main pipeline that ties together all modules:
  1. Read video frames
  2. Calibrate field (manual or auto) → compute homography
  3. Detect and track players (local YOLOv8)
  4. Transform coordinates
  5. Analyze offside
  6. Visualize results
  7. Output annotated video + JSON data

No external API required — all models run locally.
"""

import json
import time
import os
from pathlib import Path
from typing import Optional, Dict, List, Generator, Tuple
from dataclasses import dataclass, field

import cv2
import numpy as np

from .field_detector import FieldDetector, ManualFieldCalibrator, FieldDetectionResult
from .player_detector import PlayerDetector, FrameDetection
from .view_transformer import ViewTransformer
from .offside_analyzer import OffsideAnalyzer, OffsideResult
from .visualizer import OffsideVisualizer


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class ProcessingConfig:
    """Configuration for video processing."""
    # Sampling
    process_every_n_frames: int = 1           # Process every N frames
    field_detect_every_n_frames: int = 10     # Re-detect field every N frames (handles camera movement)
    skip_frames_start: int = 0                # Skip first N frames
    max_frames: Optional[int] = None          # Process at most N frames

    # Calibration
    calibration_file: Optional[str] = None    # Path to pre-saved calibration JSON
    calibration_mode: str = "manual"          # "manual", "auto", "interactive"

    # Model
    yolo_model_size: str = "football"           # YOLO variant: "football", "n", "s", "m"
    player_confidence: float = 0.35           # Player detection confidence

    # Output
    output_video_path: Optional[str] = None
    output_json_path: Optional[str] = None
    output_comparison: bool = False           # Side-by-side original vs annotated

    # Visualization
    show_minimap: bool = True
    show_player_info: bool = True
    show_comparison_view: bool = False

    # Performance
    resize_width: Optional[int] = None         # Resize frames for faster processing
    skip_when_no_offside: bool = False         # Only write frames with offside

    # Attack direction override (None=auto-detect from GK position)
    attack_dir_override: Optional[str] = None  # "top_to_bottom" or "bottom_to_top"


@dataclass
class ProcessingResult:
    """Result of video processing."""
    video_path: str
    output_video: Optional[str]
    output_json: Optional[str]
    total_frames: int
    processed_frames: int
    offside_frames: int
    total_offside_instances: int
    processing_time: float
    frame_results: List[OffsideResult] = field(default_factory=list)


class VideoProcessor:
    """
    Main video processing pipeline for offside detection.

    All models run locally — no external API needed.

    Usage:
        # Manual calibration (recommended for first run)
        processor = VideoProcessor()
        result = processor.process("match.mp4", interactive_calibrate=True)

        # With pre-saved calibration
        processor = VideoProcessor()
        result = processor.process("match.mp4", calibration_file="calib.json")

        # With Roboflow API (auto calibration)
        processor = VideoProcessor(api_key="your_key")
        result = processor.process("match.mp4")
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

        # Initialize core modules (no API needed for player detection)
        self.player_detector: Optional[PlayerDetector] = None
        self.transformer = ViewTransformer()
        self.analyzer = OffsideAnalyzer(self.transformer)
        self.visualizer = OffsideVisualizer()

        # Field detection
        self.field_detector = FieldDetector(api_key=api_key)
        self.calibrator = ManualFieldCalibrator()
        self._field_result: Optional[FieldDetectionResult] = None
        self._last_valid_field: Optional[FieldDetectionResult] = None

        # Cached YOLO-pose detector for auto mode recalibration (lazy load, once)
        self._auto_kp_detector = None

    def process(
        self,
        video_path: str,
        config: Optional[ProcessingConfig] = None,
        interactive_calibrate: bool = False,
        calibration_file: Optional[str] = None,
        calib_mode: Optional[str] = None,
        attack_dir_override: Optional[str] = None,
    ) -> ProcessingResult:
        """
        Process a video file and detect offside situations.

        Args:
            video_path: Path to input video file
            config: Processing configuration
            interactive_calibrate: If True, open calibration window on first frame
            calibration_file: Path to pre-saved calibration JSON
            calib_mode: "penalty_area", "four_corner", "generic", or None for auto-prompt
            attack_dir_override: Attack direction for center_circle mode (e.g., "left_to_right")

        Returns:
            ProcessingResult with statistics and output paths
        """
        if config is None:
            config = ProcessingConfig()

        # Override config with direct args
        if calibration_file:
            config.calibration_file = calibration_file
        if interactive_calibrate:
            config.calibration_mode = "interactive"

        # Pass attack direction override to analyzer
        self.analyzer.attack_dir_override = config.attack_dir_override
        if config.attack_dir_override:
            print(f"[VideoProcessor] Attack direction override: {config.attack_dir_override}")

        start_time = time.time()

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\n[VideoProcessor] Video: {video_path}")
        print(f"[VideoProcessor] Frames: {total_frames}, FPS: {fps:.1f}, "
              f"Size: {width}x{height}")

        # --- Step 0: Calibration ---
        self._setup_calibration(cap, config, calib_mode, video_path,
                                attack_dir_override=attack_dir_override)

        if not self.transformer.is_ready:
            print("[VideoProcessor] ERROR: Field calibration failed. Cannot proceed.")
            cap.release()
            return ProcessingResult(
                video_path=video_path,
                output_video=None,
                output_json=None,
                total_frames=total_frames, processed_frames=0,
                offside_frames=0, total_offside_instances=0,
                processing_time=0,
            )

        # --- Step 0.5: Initialize player detector ---
        self.player_detector = PlayerDetector(
            model_size=config.yolo_model_size,
            confidence=config.player_confidence,
        )

        # Pass field homography to player detector for boundary filtering
        if self.transformer.is_ready and self.transformer.homography is not None:
            self.player_detector.set_field_homography(self.transformer.homography)
            print("[VideoProcessor] Field homography set on player detector for boundary filtering")

        # --- Setup output writer ---
        out_writer = None
        if config.output_video_path:
            os.makedirs(os.path.dirname(config.output_video_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_w = width
            out_h = height
            if config.show_comparison_view:
                out_w = width * 2
            if config.resize_width:
                scale = config.resize_width / width
                out_w, out_h = int(width * scale), int(height * scale)
            # Adjust output fps to match effective frame rate after sampling
            output_fps = fps / config.process_every_n_frames
            output_fps = max(output_fps, 0.5)  # Minimum 0.5 fps
            print(f"[VideoProcessor] Output FPS: {output_fps:.1f} "
                  f"(original={fps:.1f} / every={config.process_every_n_frames})")
            out_writer = cv2.VideoWriter(
                config.output_video_path, fourcc, output_fps, (out_w, out_h),
            )

        # --- Process frames ---
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
        frame_results: List[OffsideResult] = []
        processed_count = 0
        offside_frame_count = 0
        total_offside_instances = 0
        frame_idx = 0

        print(f"\n[VideoProcessor] Starting frame processing...\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames
            if frame_idx < config.skip_frames_start:
                frame_idx += 1
                continue

            # Process every N frames
            if frame_idx % config.process_every_n_frames != 0:
                frame_idx += 1
                continue

            # Check max frames limit
            if config.max_frames and processed_count >= config.max_frames:
                break

            # Resize if configured
            if config.resize_width:
                scale = config.resize_width / frame.shape[1]
                new_h = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (config.resize_width, new_h))

            # --- Step 0: Auto-recalibrate Homography periodically (handles camera movement) ---
            if (config.field_detect_every_n_frames > 0
                    and processed_count > 0
                    and processed_count % config.field_detect_every_n_frames == 0):
                self._recalibrate_from_current_frame(frame, frame_idx)

            # --- Step 1: Player Detection & Tracking ---
            frame_det = self.player_detector.detect_and_track(
                frame, frame_idx, detect_teams=True,
            )

            # --- Step 2: Offside Analysis ---
            offside_result = self.analyzer.analyze(frame_det)
            frame_results.append(offside_result)

            if offside_result.is_offside_situation:
                offside_frame_count += 1
                total_offside_instances += len(offside_result.offside_players)

            # --- Step 3: Visualization ---
            annotated = self.visualizer.draw(
                frame, offside_result,
                frame_det=frame_det,
                show_minimap=config.show_minimap,
                show_player_info=config.show_player_info,
            )

            # Write output frame
            if out_writer is not None:
                if config.skip_when_no_offside and not offside_result.is_offside_situation:
                    pass  # Skip non-offside frames
                elif config.show_comparison_view:
                    comparison = self.visualizer.create_comparison_frame(frame, annotated)
                    out_writer.write(comparison)
                else:
                    out_writer.write(annotated)

            processed_count += 1

            # Progress
            if processed_count % 50 == 0:
                elapsed = time.time() - start_time
                fps_proc = processed_count / elapsed if elapsed > 0 else 0
                print(f"[VideoProcessor] Frame {frame_idx}: {processed_count} processed, "
                      f"{offside_frame_count} offside, {fps_proc:.1f} fps")

            frame_idx += 1

        # Cleanup
        cap.release()
        if out_writer is not None:
            out_writer.release()
        cv2.destroyAllWindows()

        processing_time = time.time() - start_time

        # Generate result
        result = ProcessingResult(
            video_path=video_path,
            output_video=config.output_video_path,
            output_json=config.output_json_path,
            total_frames=total_frames,
            processed_frames=processed_count,
            offside_frames=offside_frame_count,
            total_offside_instances=total_offside_instances,
            processing_time=processing_time,
            frame_results=frame_results,
        )

        # Save JSON data
        if config.output_json_path:
            self._save_json_result(result, config.output_json_path)

        # Print summary
        self._print_summary(result)

        return result

    def _setup_calibration(self, cap, config: ProcessingConfig, calib_mode: Optional[str] = None,
                            video_path: Optional[str] = None, attack_dir_override: Optional[str] = None):
        """Setup field calibration before processing."""
        ret, first_frame = cap.read()
        if not ret:
            print("[VideoProcessor] Cannot read first frame for calibration")
            return

        if config.resize_width:
            scale = config.resize_width / first_frame.shape[1]
            new_h = int(first_frame.shape[0] * scale)
            first_frame = cv2.resize(first_frame, (config.resize_width, new_h))

        # Priority 1: Load from calibration file
        if config.calibration_file and os.path.exists(config.calibration_file):
            print(f"[VideoProcessor] Loading calibration from: {config.calibration_file}")
            result = self.calibrator.load_calibration(config.calibration_file)
            if result and result.is_reliable:
                self._apply_calibration(result)
                print(f"[VideoProcessor] Calibration loaded successfully.")
                return
            else:
                print(f"[VideoProcessor] Failed to load calibration file.")

        # Priority 2: Auto calibration (local YOLOv8-pose model, no API needed)
        if config.calibration_mode == "auto":
            print("[VideoProcessor] Running auto keypoint detection...")
            result = self.calibrator.calibrate_auto(first_frame, video_path=video_path)
            if result and result.is_reliable:
                self._apply_calibration(result)
                # Save calibration for future use
                calib_path = str(Path(config.output_video_path).with_suffix("")) if config.output_video_path else "./output/auto_calib"
                calib_path = f"{calib_path}_calibration.json"
                self.calibrator.save_calibration(result, calib_path)
                print(f"[VideoProcessor] Auto calibration successful! Saved: {calib_path}")
                return
            else:
                print("[VideoProcessor] Auto detection failed, falling back to manual...")
                result = self.calibrator.calibrate_interactive(first_frame, mode=calib_mode,
                                                              attack_dir=attack_dir_override)
                if result and result.is_reliable:
                    self._apply_calibration(result)
                    # Save calibration for future use
                    if config.output_video_path:
                        calib_path = config.output_video_path.replace(".mp4", "_calibration.json")
                        self.calibrator.save_calibration(result, calib_path)
                    return
                else:
                    print("[VideoProcessor] Interactive calibration cancelled.")

        # Priority 3: Auto detection (Roboflow API)
        if self.api_key:
            print("[VideoProcessor] Attempting automatic field detection...")
            result = self.field_detector.detect(first_frame)
            if result.is_reliable:
                self._apply_calibration(result)
                print(f"[VideoProcessor] Auto calibration: {result.num_valid_keypoints} keypoints")
                return
            else:
                print("[VideoProcessor] Auto detection failed, falling back to manual...")

        # Priority 4: Interactive calibration
        # Use calib_mode parameter if provided, otherwise fall back to config.calibration_mode
        effective_mode = calib_mode if calib_mode else config.calibration_mode
        # Always enter interactive mode if we reach here (Priority 1-3 all failed)
        print(f"[VideoProcessor] Switching to interactive calibration (mode={effective_mode})...")
        if effective_mode == "interactive":
            result = self.calibrator.calibrate_interactive(first_frame, mode=None,
                                                              attack_dir=attack_dir_override)
        else:
            result = self.calibrator.calibrate_interactive(first_frame, mode=effective_mode,
                                                          attack_dir=attack_dir_override)
        if result and result.is_reliable:
            self._apply_calibration(result)
            if config.output_video_path:
                calib_path = config.output_video_path.replace(".mp4", "_calibration.json")
                self.calibrator.save_calibration(result, calib_path)
        else:
            print("[VideoProcessor] Interactive calibration cancelled or failed.")

    def _apply_calibration(self, result: FieldDetectionResult):
        """Apply calibration result to the transformer."""
        self._field_result = result
        self._last_valid_field = result
        if result.homography is not None:
            self.transformer.set_homography(result.homography)

    def _recalibrate_from_current_frame(self, frame: np.ndarray, frame_idx: int):
        """
        Automatically re-detect the field and update Homography.

        Two strategies based on calibration mode:
        - center_circle / interactive: Use VP from field lines + stored click points (fast, pure CV)
        - auto: Use cached YOLO-pose 32-keypoint detector for full independent recalibration
        """
        if self._field_result is None:
            return

        calib_mode = self._field_result.calibration_mode

        # ── Strategy A: Auto mode → YOLO-pose keypoint re-detection ──
        if calib_mode == "auto":
            self._recalibrate_auto(frame, frame_idx)
            return

        # ── Strategy B: Center-circle / manual modes → VP from field lines ──
        self._recalibrate_from_vp(frame, frame_idx)

    def _recalibrate_auto(self, frame: np.ndarray, frame_idx: int):
        """Re-detect 32 keypoints using cached YOLO-pose model, compute new Homography."""
        try:
            from .pitch_keypoint_detector import PitchKeypointDetector
        except ImportError:
            return

        # Lazy-load the YOLO-pose detector (once per video)
        if self._auto_kp_detector is None:
            try:
                self._auto_kp_detector = PitchKeypointDetector()
                print(f"  [Recalib Auto] YOLO-pose detector loaded (1-time init)")
            except Exception as e:
                print(f"  [Recalib Auto] Failed to load detector: {e}")
                return

        try:
            # Detect 32 keypoints on current frame
            pixel_kpts = self._auto_kp_detector.detect(frame)
            if pixel_kpts is None:
                return

            n_detected = int(np.sum(pixel_kpts[:, 2] > 0.0))
            if n_detected < 4:
                return

            # Compute homography with multi-subset voter
            from .field_detector import FieldDetectionResult
            H, voter_mask, voter_metrics = self._auto_kp_detector.voter_homography(
                pixel_kpts, min_confidence=0.2, ransac_threshold=10.0
            )

            if H is None:
                return

            # Convert Roboflow → FIFA
            from .pitch_keypoint_detector import _CONV_ROBOFLOW_TO_FIFA
            H_fifa = _CONV_ROBOFLOW_TO_FIFA @ H

            # Validate
            valid, detail = self._auto_kp_detector.validate_homography(H_fifa)
            if not valid and detail.get("goal_line_angle_deg", 999) > 10.0:
                return

            result = FieldDetectionResult(
                keypoints=pixel_kpts[:, :2],
                confidences=pixel_kpts[:, 2],
                homography=H_fifa,
                inverse_homography=np.linalg.inv(H_fifa),
                is_reliable=True,
                num_valid_keypoints=n_detected,
                calibration_mode="auto",
                world_keypoints=self._auto_kp_detector.get_world_keypoints()[:, :2],
                mean_confidence=float(np.mean(pixel_kpts[:, 2])),
                voter_rmse=voter_metrics.get("rmse", 0.0),
                subset=voter_metrics.get("subset_name", "voter"),
            )

            self.transformer.set_homography(H_fifa)
            if self.player_detector is not None:
                self.player_detector.set_field_homography(H_fifa)
            self._field_result = result
            self.analyzer._direction_locked = False
            self.analyzer._direction_votes = []
            self.analyzer._total_frames_analyzed = 0
            print(f"  [Recalib Auto Frame {frame_idx}] {n_detected}/32 kpts "
                  f"RMSE={voter_metrics.get('rmse', 0):.1f}m "
                  f"subset={voter_metrics.get('subset_name', '?')}")
        except Exception as e:
            print(f"  [Recalib Auto Frame {frame_idx}] Failed: {e}")

    def _recalibrate_from_vp(self, frame: np.ndarray, frame_idx: int):
        """VP-based recalibration for center_circle and manual modes."""
        try:
            from .pitch_keypoint_detector import PitchKeypointDetector
        except ImportError:
            return

        stored_kpts = self._field_result.keypoints
        if stored_kpts is None or len(stored_kpts) < 2:
            return

        try:
            # Use STATIC methods (no model load) for line detection
            field_lines = PitchKeypointDetector.detect_field_lines(frame)
            if not field_lines or len(field_lines) < 2:
                return

            vp, vp_quality = PitchKeypointDetector.find_vp_from_lines(field_lines)
            if vp is None or vp_quality < 0.3:
                return

            from .field_detector import CenterCircleCalibrator
            cc = CenterCircleCalibrator()
            pixel_pts = [(float(stored_kpts[i][0]), float(stored_kpts[i][1]))
                         for i in range(min(2, len(stored_kpts)))]
            attack_dir = self._field_result.attack_dir

            vp_result = cc._build_from_vp(
                frame, pixel_pts, attack_dir=attack_dir,
                vp=vp, vp_quality=vp_quality, kp_detector=None
            )

            if vp_result is not None and vp_result.is_reliable:
                self.transformer.set_homography(vp_result.homography)
                if self.player_detector is not None:
                    self.player_detector.set_field_homography(vp_result.homography)
                self._field_result = vp_result
                self.analyzer._direction_locked = False
                self.analyzer._direction_votes = []
                self.analyzer._total_frames_analyzed = 0
                print(f"  [Recalib VP Frame {frame_idx}] VP=({vp[0]:.0f},{vp[1]:.0f}) "
                      f"quality={vp_quality:.2f} n_lines={len(field_lines)}")
        except Exception as e:
            print(f"  [Recalib VP Frame {frame_idx}] Failed: {e}")

    def _save_json_result(self, result: ProcessingResult, json_path: str):
        """Save processing results to JSON file."""
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        frames_data = []
        for fr in result.frame_results:
            frames_data.append({
                "frame_idx": fr.frame_idx,
                "attack_direction": fr.attack_direction.value,
                "offside_line_y": fr.offside_line_y,
                "is_offside_situation": fr.is_offside_situation,
                "offside_players": [
                    {
                        "track_id": p["track_id"],
                        "world_x": float(p["world_x"]),
                        "world_y": float(p["world_y"]),
                        "distance_offside": float(p.get("distance_offside", 0)),
                        "team": p.get("team", "unknown"),
                    }
                    for p in fr.offside_players
                ],
                "last_defender": {
                    "track_id": fr.second_last_defender["track_id"],
                    "world_x": float(fr.second_last_defender["world_x"]),
                    "world_y": float(fr.second_last_defender["world_y"]),
                } if fr.second_last_defender else None,
            })

        output = {
            "video_path": result.video_path,
            "total_frames": result.total_frames,
            "processed_frames": result.processed_frames,
            "offside_frames": result.offside_frames,
            "total_offside_instances": result.total_offside_instances,
            "processing_time_seconds": result.processing_time,
            "frames": frames_data,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)

        print(f"\n[VideoProcessor] JSON data saved to {json_path}")

    def _print_summary(self, result: ProcessingResult):
        """Print processing summary."""
        print("\n" + "=" * 60)
        print("  OFFSIDE DETECTION PROCESSING COMPLETE")
        print("=" * 60)
        print(f"  Video:              {result.video_path}")
        print(f"  Total frames:       {result.total_frames}")
        print(f"  Processed frames:   {result.processed_frames}")
        print(f"  Offside frames:     {result.offside_frames}")
        print(f"  Offside instances:  {result.total_offside_instances}")
        print(f"  Processing time:    {result.processing_time:.1f}s")
        if result.processed_frames > 0:
            print(f"  FPS (processing):   {result.processed_frames / result.processing_time:.1f}")
        if result.output_video:
            print(f"  Output video:       {result.output_video}")
        if result.output_json:
            print(f"  Output JSON:        {result.output_json}")
        print("=" * 60)

    def process_stream(
        self,
        video_path: str,
        start_frame: int = 0,
        calibration_file: Optional[str] = None,
    ) -> Generator[Tuple[int, np.ndarray, OffsideResult], None, None]:
        """
        Generator for streaming frame-by-frame processing.
        Useful for real-time applications and web interfaces.

        Yields:
            (frame_idx, annotated_frame, offside_result)
        """
        if self.player_detector is None:
            self.player_detector = PlayerDetector()

        cap = cv2.VideoCapture(video_path)

        # Setup calibration if not ready
        if not self.transformer.is_ready:
            if calibration_file and os.path.exists(calibration_file):
                result = self.calibrator.load_calibration(calibration_file)
                if result and result.is_reliable:
                    self._apply_calibration(result)

            if not self.transformer.is_ready and not self.api_key:
                ret, first_frame = cap.read()
                if ret:
                    result = self.calibrator.calibrate_interactive(first_frame)
                    if result and result.is_reliable:
                        self._apply_calibration(result)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = start_frame

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if not self.transformer.is_ready:
                frame_idx += 1
                continue

            # Player detection
            frame_det = self.player_detector.detect_and_track(frame, frame_idx)

            # Offside analysis
            offside_result = self.analyzer.analyze(frame_det)

            # Visualization
            annotated = self.visualizer.draw(frame, offside_result, frame_det=frame_det)

            yield frame_idx, annotated, offside_result
            frame_idx += 1

        cap.release()
