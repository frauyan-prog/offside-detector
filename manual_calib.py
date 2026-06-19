#!/usr/bin/env python3
"""
手动标定工具：弹出视频第一帧，鼠标点击4个点，自动保存标定JSON
用法：python manual_calib.py -v VIDEO.mp4 --mode penalty_area
"""

import argparse
import json
import cv2
import numpy as np
import os

def manual_calibrate(video_path, mode='four_corner', output_json=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return None
    
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("ERROR: Cannot read first frame")
        return None
    
    h, w = frame.shape[:2]
    print(f"\n{'='*60}")
    print(f"  手动标定工具 - {mode}")
    print(f"  视频: {video_path}")
    print(f"  帧大小: {w}x{h}")
    print(f"{'='*60}\n")
    
    if mode == 'four_corner':
        instructions = [
            "① 点击：左上角（靠近镜头的左角）-> 世界坐标 (0, 105)",
            "② 点击：右上角（靠近镜头的右角）-> 世界坐标 (68, 105)",
            "③ 点击：右下角（远端右角）-> 世界坐标 (68, 0)",
            "④ 点击：左下角（远端左角）-> 世界坐标 (0, 0)",
        ]
        world_points = [(0, 105), (68, 105), (68, 0), (0, 0)]
    elif mode == 'penalty_area':
        instructions = [
            "① 点击：禁区上左角 -> 世界坐标 (0, 58.25)",
            "② 点击：禁区上右角 -> 世界坐标 (16.5, 58.25)",
            "③ 点击：禁区下右角 -> 世界坐标 (16.5, 42.75)",
            "④ 点击：禁区下左角 -> 世界坐标 (0, 42.75)",
        ]
        world_points = [(0, 58.25), (16.5, 58.25), (16.5, 42.75), (0, 42.75)]
    else:
        print(f"ERROR: Unknown mode {mode}")
        return None
    
    print("点击顺序：")
    for inst in instructions:
        print(f"  {inst}")
    print("\n提示：右键删除最后一个点，按 'r' 重置，按 's' 保存，按 'q' 退出\n")
    
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal points, frame, display_frame
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                pts_copy = world_points[len(points)]
                points.append((x, y, pts_copy[0], pts_copy[1]))
                print(f"  点 {len(points)}: 像素({x}, {y}) -> 世界({pts_copy[0]}, {pts_copy[1]})")
            else:
                print("  已选满4个点！右键删除或按 's' 保存")
        elif event == cv2.EVENT_RBUTTONDOWN:
            if points:
                removed = points.pop()
                print(f"  删除点: {removed}")
    
    # Create window
    win_name = "Manual Calibration - Click 4 points"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, mouse_callback)
    
    # Scale frame for display
    max_h = 800
    scale = min(1.0, max_h / h)
    disp_w, disp_h = int(w * scale), int(h * scale)
    cv2.resizeWindow(win_name, disp_w, disp_h)
    
    while True:
        display = frame.copy()
        
        # Draw points
        colors = [(0, 212, 255), (0, 255, 136), (255, 170, 0), (255, 68, 102)]
        for i, (px, py, wx, wy) in enumerate(points):
            color = colors[i]
            cv2.drawMarker(display, (px, py), color, cv2.MARKER_CROSS, 20, 2)
            cv2.circle(display, (px, py), 6, color, 2)
            cv2.putText(display, f"P{i+1}({wx},{wy})", (px+10, py-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw instructions
        y_offset = 30
        for i, inst in enumerate(instructions):
            color = (0, 255, 0) if i < len(points) else (200, 200, 200)
            cv2.putText(display, inst, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20
        
        cv2.putText(display, f"已选: {len(points)}/4  |  's'保存  'r'重置  'q'退出", 
                   (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Show scaled
        disp = cv2.resize(display, (disp_w, disp_h))
        cv2.imshow(win_name, disp)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("退出，未保存")
            break
        elif key == ord('s'):
            if len(points) == 4:
                # Save JSON
                if output_json is None:
                    base = os.path.splitext(video_path)[0]
                    output_json = f"{base}_{mode}_calibration.json"
                
                pixel_pts = [[p[0], p[1]] for p in points]
                world_pts = [[p[2], p[3]] for p in points]
                
                calib = {
                    "calibration_mode": "generic",
                    "pixel_points": pixel_pts,
                    "world_points": world_pts,
                    "image_size": [w, h],
                    "note": f"Manual calibration: {mode}"
                }
                
                with open(output_json, 'w') as f:
                    json.dump(calib, f, indent=2)
                
                print(f"\n✅ 标定JSON已保存: {output_json}")
                print(f"   像素点: {pixel_pts}")
                print(f"   世界坐标: {world_pts}")
                print(f"\n运行命令：")
                print(f"  python run.py -v \"{video_path}\" --calibration \"{output_json}\"")
                cv2.destroyAllWindows()
                return output_json
            else:
                print(f"需要4个点，当前只有 {len(points)} 个！")
        elif key == ord('r'):
            points = []
            print("已重置所有点")
    
    cv2.destroyAllWindows()
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='手动标定工具')
    parser.add_argument('-v', '--video', required=True, help='视频文件')
    parser.add_argument('--mode', choices=['four_corner', 'penalty_area'], 
                       default='four_corner', help='标定模式')
    parser.add_argument('-o', '--output', help='输出JSON文件路径')
    args = parser.parse_args()
    
    manual_calibrate(args.video, args.mode, args.output)
