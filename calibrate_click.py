"""独立标定工具 — 点击4个点，自动生成JSON"""

import cv2
import json
import sys
import os
import numpy as np

VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "13-有意处理球.mp4"

# 读取首帧
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()
if not ret:
    print("无法读取视频首帧")
    sys.exit(1)

h, w = frame.shape[:2]
print(f"图像尺寸: {w}x{h}")

clicked = []
FRAME = frame.copy()
WINDOW = "标定窗口 — 按顺序点击4个角 (←退格撤销, ESC取消, ENTER确认)"

def on_mouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
        clicked.append((x, y))
        print(f"  点 {len(clicked)}: ({x}, {y})")

cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW, w, h)
cv2.setMouseCallback(WINDOW, on_mouse)

print("\n按顺序点击 4 个场角：")
print("  ① 左上角  ② 右上角  ③ 右下角  ④ 左下角")
print("  退格键 撤销上一个点 | ESC 取消 | ENTER 确认\n")

while True:
    display = FRAME.copy()

    # 画已选点
    for i, (px, py) in enumerate(clicked):
        cv2.circle(display, (px, py), 8, (0, 255, 255), -1)
        cv2.putText(display, str(i+1), (px+12, py-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if len(clicked) >= 2:
        pts = np.array(clicked, np.int32)
        cv2.polylines(display, [pts], len(clicked) == 4, (0, 255, 255), 2)

    # 提示文字
    if len(clicked) < 4:
        cv2.putText(display, f"点击第 {len(clicked)+1} 个点", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        cv2.putText(display, "按 ENTER 确认", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow(WINDOW, display)
    key = cv2.waitKey(30) & 0xFF

    if key == 27:  # ESC
        print("取消")
        cv2.destroyAllWindows()
        sys.exit(0)
    elif key == 8:  # Backspace
        if clicked:
            removed = clicked.pop()
            print(f"  撤销: ({removed[0]}, {removed[1]})")
    elif key == 13 and len(clicked) == 4:  # Enter
        break

cv2.destroyAllWindows()

# 四个角的世界坐标 (FIFA 105×68)
# TL=(0,0), TR=(68,0), BR=(68,105), BL=(0,105)
world_coords = [
    [0, 0],       # TL
    [68, 0],      # TR
    [68, 105],    # BR
    [0, 105],     # BL
]

# 保存 JSON
output_path = os.path.splitext(VIDEO_PATH)[0] + "_calibration.json"
if os.path.sep in output_path:
    pass
else:
    output_path = "output/" + os.path.basename(output_path)

os.makedirs(os.path.dirname(output_path), exist_ok=True)

data = {
    "calibration_mode": "four_corner",
    "pixel_points": [[float(x), float(y)] for x, y in clicked],
    "world_points": world_coords,
    "image_size": [w, h],
}

with open(output_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"\n✅ 标定已保存: {output_path}")
print(f"   像素点: {clicked}")
print(f"\n运行命令:")
print(f'  python run.py -v "{VIDEO_PATH}" --calibration "{output_path}" -o "output/demo" --every 2 --max-frames 100')
