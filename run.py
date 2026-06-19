"""
Offside Detector CLI

Command-line interface for processing football match videos and detecting
offside situations using local YOLOv8 models. No external API required.

Usage:
    # Basic usage (opens calibration window)
    python run.py --video match.mp4

    # With pre-saved calibration
    python run.py --video match.mp4 --calibration calib.json

    # Stream mode (for web integration)
    python run.py --video match.mp4 --mode stream --port 8080

    # Fast preview
    python run.py --video match.mp4 --resize 960 --max-frames 300
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from offside_detector.video_processor import VideoProcessor, ProcessingConfig


def main():
    parser = argparse.ArgumentParser(
        description="Automated Football Offside Detection System (Local YOLOv8)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive calibration (click 4 corners on first frame)
  python run.py --video match.mp4

  # Use pre-saved calibration
  python run.py --video match.mp4 --calibration calib.json

  # Fast preview (resize + limit frames)
  python run.py --video match.mp4 --resize 960 --max-frames 300

  # Process every 3 frames for speed
  python run.py --video match.mp4 --every 3

  # Web server mode
  python run.py --video match.mp4 --mode stream --port 8080

  # Auto calibration (local YOLOv8-pose, no API key needed!)
  python run.py --video match.mp4 --calib-mode auto

  # Auto calibration with pre-saved calibration
  python run.py --video match.mp4 --calib-mode auto --calibration calib.json
        """,
    )

    # Required arguments
    parser.add_argument(
        "--video", "-v", type=str, required=True,
        help="Path to input video file",
    )

    # Calibration
    parser.add_argument(
        "--calibration", "-c", type=str, default=None,
        help="Path to pre-saved calibration JSON file (skip interactive calibration)",
    )
    parser.add_argument(
        "--calib-mode", type=str, choices=["auto", "penalty_area", "center_circle", "four_corner", "generic"],
        default=None,
        help="Calibration mode: auto (AI auto-detect keypoints, recommended!), "
             "center_circle (2 points for midfield), "
             "penalty_area (near goal, 4 points), four_corner (full pitch, 4 points), "
             "generic (any rectangle, 4 points)",
    )
    parser.add_argument(
        "--api-key", "-k", type=str,
        default=os.environ.get("ROBOFLOW_API_KEY"),
        help="Roboflow API key for auto field detection (optional)",
    )

    # Output configuration
    parser.add_argument(
        "--output", "-o", type=str, default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--no-json", action="store_true",
        help="Skip JSON data output",
    )
    parser.add_argument(
        "--comparison", action="store_true",
        help="Output side-by-side comparison view",
    )

    # Processing configuration
    parser.add_argument(
        "--every", "-e", type=int, default=1,
        help="Process every N frames (default: 1)",
    )
    parser.add_argument(
        "--max-frames", "-m", type=int, default=None,
        help="Maximum frames to process",
    )
    parser.add_argument(
        "--skip", "-s", type=int, default=0,
        help="Skip first N frames",
    )
    parser.add_argument(
        "--resize", "-r", type=int, default=None,
        help="Resize frames to this width (faster, e.g. 960)",
    )

    # Model configuration
    parser.add_argument(
        "--yolo-model", type=str, choices=["football", "n", "s", "m"], default="football",
        help="YOLOv8 model size: n=nano (fastest), s=small, m=medium (default: n)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.35,
        help="Player detection confidence threshold (default: 0.35)",
    )

    # Visualization
    parser.add_argument(
        "--no-minimap", action="store_true",
        help="Hide pitch minimap overlay",
    )
    parser.add_argument(
        "--only-offside", action="store_true",
        help="Only output frames with offside",
    )
    parser.add_argument(
        "--attack-dir", type=str, choices=["top_to_bottom", "bottom_to_top"],
        default=None,
        help="Manually specify attack direction (override auto-detection). "
             "top_to_bottom = attacking toward y=105 (goal at bottom), "
             "bottom_to_top = attacking toward y=0 (goal at top)",
    )

    # Mode
    parser.add_argument(
        "--mode", type=str, choices=["batch", "stream", "interactive"],
        default="batch",
        help="Processing mode (default: batch)",
    )
    parser.add_argument(
        "--port", "-p", type=int, default=8080,
        help="Port for web server in stream mode",
    )

    args = parser.parse_args()

    # Prepare output paths
    os.makedirs(args.output, exist_ok=True)
    video_name = Path(args.video).stem
    output_video = os.path.join(args.output, f"{video_name}_offside.mp4")
    output_json = None if args.no_json else os.path.join(
        args.output, f"{video_name}_analysis.json"
    )

    # Build configuration
    config = ProcessingConfig(
        process_every_n_frames=args.every,
        skip_frames_start=args.skip,
        max_frames=args.max_frames,
        output_video_path=output_video,
        output_json_path=output_json,
        show_comparison_view=args.comparison,
        show_minimap=not args.no_minimap,
        show_player_info=True,
        skip_when_no_offside=args.only_offside,
        resize_width=args.resize,
        calibration_file=args.calibration,
        calibration_mode="auto" if args.calib_mode == "auto" else ("interactive" if not args.calibration and not args.api_key else "manual"),
        yolo_model_size=args.yolo_model,
        player_confidence=args.confidence,
        attack_dir_override=args.attack_dir,
    )

    # Execute based on mode
    if args.mode == "batch":
        run_batch(args, config)
    elif args.mode == "stream":
        run_stream(args, config)
    elif args.mode == "interactive":
        run_interactive(args, config)


def run_batch(args, config):
    """Batch processing mode — process entire video and save output."""
    processor = VideoProcessor(api_key=args.api_key)

    print(f"\n{'=' * 60}")
    print(f"  OFFSIDE DETECTOR — Batch Mode")
    print(f"{'=' * 60}")
    print(f"  Video:     {args.video}")
    model_label = "Football YOLOv8" if args.yolo_model == "football" else f"YOLOv8{args.yolo_model}"
    print(f"  Model:     {model_label}")
    print(f"  Output:    {args.output}/")
    if args.resize:
        print(f"  Resize:    {args.resize}px width")
    if args.max_frames:
        print(f"  Limit:     {args.max_frames} frames")
    print(f"{'=' * 60}\n")

    # Only force interactive mode when no auto calibration is active
    force_interactive = (
        not args.calibration
        and not args.api_key
        and args.calib_mode != "auto"
    )

    result = processor.process(
        args.video, config,
        interactive_calibrate=force_interactive,
        calibration_file=args.calibration,
        calib_mode=args.calib_mode,
    )

    return result


def run_stream(args, config):
    """Stream mode — start a web server for browser-based viewing."""
    try:
        from web.app import start_server
        start_server(
            video_path=args.video,
            api_key=args.api_key,
            calibration_file=args.calibration,
            output_dir=args.output,
            port=args.port,
        )
    except ImportError as e:
        print(f"ERROR: Could not import web module: {e}")
        print("Please install Flask: pip install flask")
        sys.exit(1)


def run_interactive(args, config):
    """Interactive mode — display results in a window."""
    import cv2

    processor = VideoProcessor(api_key=args.api_key)

    print(f"\n{'=' * 60}")
    print(f"  OFFSIDE DETECTOR — Interactive Mode")
    print(f"{'=' * 60}")
    print("  Controls: 'q'=quit, 's'=skip, 'p'=pause")
    print(f"{'=' * 60}\n")

    for frame_idx, annotated, offside_result in processor.process_stream(
        args.video,
        calibration_file=args.calibration,
    ):
        cv2.imshow("Offside Detection", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(0)  # Wait indefinitely
        elif key == ord('s'):
            pass  # Continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
