"""
Web Application for Offside Detection — Integrated Calibration + Processing
"""

import os
import sys
import json
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, Dict, List

# Ensure project root is on the path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import numpy as np
from flask import (
    Flask, render_template, request, jsonify,
    send_from_directory, Response, send_file,
)

from offside_detector.video_processor import VideoProcessor, ProcessingConfig


app = Flask(__name__,
    template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
    static_folder=os.path.join(os.path.dirname(__file__), 'static'),
)

# ── Directories ────────────────────────────────────────────────────────────────
_UPLOAD_DIR   = os.path.join(_project_root, 'uploads')
_OUTPUT_DIR   = os.path.join(_project_root, 'output')
_TASK_STATE_DIR = os.path.join(os.path.dirname(__file__), 'task_state')
for d in [_UPLOAD_DIR, _OUTPUT_DIR, _TASK_STATE_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Global state ───────────────────────────────────────────────────────────────
_processor: Optional[VideoProcessor] = None
_current_task: Optional[Dict] = None
_api_key: Optional[str] = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _save_task_state():
    if _current_task is None:
        return
    state_file = os.path.join(_TASK_STATE_DIR, f"{_current_task['id']}.json")
    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump(_current_task, f, ensure_ascii=False, indent=2)


def _load_task_state(task_id: str) -> Optional[Dict]:
    global _current_task
    if _current_task and _current_task.get('id') == task_id:
        return _current_task
    state_file = os.path.join(_TASK_STATE_DIR, f"{task_id}.json")
    if os.path.exists(state_file):
        with open(state_file, 'r', encoding='utf-8') as f:
            _current_task = json.load(f)
        return _current_task
    return None


def init_processor(api_key: str = None):
    global _processor, _api_key
    _api_key = api_key
    _processor = VideoProcessor(api_key=api_key)


def _process_video_background(video_path: str, calib_path: str,
                               every_n: int, max_frames: int, task_id: str):
    """Background thread for video processing."""
    global _current_task
    try:
        from offside_detector.video_processor import VideoProcessor, ProcessingConfig

        processor = VideoProcessor()
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_video = os.path.join(_OUTPUT_DIR, video_name + "_offside.mp4")
        output_json  = os.path.join(_OUTPUT_DIR, video_name + "_analysis.json")

        cfg = ProcessingConfig(
            process_every_n_frames=every_n,
            max_frames=max_frames if max_frames > 0 else None,
            output_video_path=output_video,
            output_json_path=output_json,
            show_minimap=True,
            calibration_file=calib_path,
            calibration_mode="manual",
        )

        result = processor.process(video_path, config=cfg)

        if _current_task:
            _current_task["status"] = "completed"
            _current_task["progress"] = 100
            _current_task["results"] = {
                "offside_frames": result.offside_frames,
                "total_frames": result.total_frames,
                "processed_frames": result.processed_frames,
                "total_offside_instances": result.total_offside_instances,
                "processing_time_seconds": result.processing_time,
                "output_video": result.output_video,
                "output_json": result.output_json,
            }
            _save_task_state()
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"[Process Error] {err_msg}")
        if _current_task:
            _current_task["status"] = "error"
            _current_task["error"] = str(e)
            _current_task["traceback"] = err_msg
            _save_task_state()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


# ── API: List video files ──────────────────────────────────────────────────────

@app.route('/api/videos')
def api_list_videos():
    """List video files in project root and uploads directory."""
    videos = []
    for directory in [_project_root, _UPLOAD_DIR]:
        if not os.path.exists(directory):
            continue
        for f in os.listdir(directory):
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
                fpath = os.path.join(directory, f)
                videos.append({
                    "name": f,
                    "path": fpath,
                    "size_mb": round(os.path.getsize(fpath) / 1024 / 1024, 1),
                })
    return jsonify({"videos": videos})


# ── API: Extract first frame ──────────────────────────────────────────────────

@app.route('/api/video/first_frame')
def api_first_frame():
    """
    Extract and return the first frame of a video as a JPEG.
    Query params: path=<video_path>
    """
    video_path = request.args.get('path', '')
    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "Video not found"}), 404

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "Cannot open video"}), 400

    ret, frame = cap.read()
    cap.release()
    if not ret:
        return jsonify({"error": "Cannot read frame"}), 400

    # Encode as JPEG
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buf.tobytes(), mimetype='image/jpeg')


# ── API: Start processing with calibration ─────────────────────────────────────

@app.route('/api/process', methods=['POST'])
def api_process():
    """
    Start processing a video with manual calibration.

    Body (JSON):
      video_path:  path to video file
      calibration: { pixel_points: [[x,y],...], world_points: [[wx,wy],...] }
      every_n:     frame sampling interval (default 3)
      max_frames:  max frames to process (0 = all)
    """
    global _current_task

    data = request.get_json()
    video_path = data.get('video_path', '')
    calibration = data.get('calibration', None)
    every_n = int(data.get('every_n', 3))
    max_frames = int(data.get('max_frames', 0))

    if not os.path.exists(video_path):
        return jsonify({"error": f"Video not found: {video_path}"}), 404

    if not calibration:
        return jsonify({"error": "Calibration data required"}), 400

    # Save calibration JSON
    calib_path = os.path.join(
        _OUTPUT_DIR,
        os.path.splitext(os.path.basename(video_path))[0] + "_calibration.json"
    )
    calib_data = {
        "calibration_mode": "generic",
        "pixel_points": calibration["pixel_points"],
        "world_points": calibration["world_points"],
        "image_size": calibration.get("image_size", [0, 0]),
        "note": "Manual calibration from web UI",
    }
    with open(calib_path, 'w', encoding='utf-8') as f:
        json.dump(calib_data, f, indent=2, ensure_ascii=False)

    # Start background processing
    task_id = f"task_{int(time.time())}"
    _current_task = {
        "id": task_id,
        "status": "processing",
        "progress": 0,
        "video_path": video_path,
        "calib_path": calib_path,
        "start_time": time.time(),
    }
    _save_task_state()

    thread = threading.Thread(
        target=_process_video_background,
        args=(video_path, calib_path, every_n, max_frames, task_id),
        daemon=True,
    )
    thread.start()

    return jsonify({"task_id": task_id, "calib_path": calib_path})


# ── API: Poll progress ─────────────────────────────────────────────────────────

@app.route('/api/progress/<task_id>')
def api_progress(task_id):
    task = _load_task_state(task_id)
    if task:
        return jsonify(task)
    return jsonify({"error": "Task not found"}), 404


# ── API: Download result video ─────────────────────────────────────────────────

@app.route('/api/download/<task_id>')
def api_download(task_id):
    task = _load_task_state(task_id)
    if not task or task.get("status") != "completed":
        return jsonify({"error": "Task not completed"}), 400

    video_path = task["results"].get("output_video", '')
    if os.path.exists(video_path):
        return send_file(video_path, mimetype='video/mp4', as_attachment=True)
    return jsonify({"error": "Video not found"}), 404


# ── API: Get output video for inline playback ──────────────────────────────────

@app.route('/api/output_video/<task_id>')
def api_output_video(task_id):
    task = _load_task_state(task_id)
    if not task or task.get("status") != "completed":
        return jsonify({"error": "Task not completed"}), 400

    video_path = task["results"].get("output_video", '')
    if os.path.exists(video_path):
        return send_file(video_path, mimetype='video/mp4')
    return jsonify({"error": "Video not found"}), 404


# ── Server start ───────────────────────────────────────────────────────────────

def start_server(port=8080):
    init_processor()
    print(f"\n[Offside Web] http://localhost:{port}")
    print(f"[Offside Web] Upload dir: {_UPLOAD_DIR}")
    print(f"[Offside Web] Output dir: {_OUTPUT_DIR}\n")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    start_server(args.port)
