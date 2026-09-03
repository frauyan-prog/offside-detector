# -*- coding: utf-8 -*-
"""
测试运行器：在 CodeBuddy Terminal (PowerShell) 里运行本项目。

存在原因：PowerShell 对中文路径有编码问题（cd 到项目目录会失败），
所有路径改由 Python 内部拼接后直接传给子进程，绕开 shell 解析。

用法（在项目根目录执行）：
    python test_run.py                                  # 13 号视频，auto 标定
    python test_run.py 12                               # 指定视频编号 11/12/13/14
    python test_run.py 13 --calib-mode center_circle    # center_circle（自动找标定文件）
    python test_run.py 13 --max-frames 300              # 只处理前 300 帧
    python test_run.py 13 --calib-mode center_circle --max-frames 300

其余参数（--resize / --every / --skip / --confidence 等）原样透传给 run.py。
"""
import os
import subprocess
import sys

PROJECT = os.path.dirname(os.path.abspath(__file__))
VENV_PY = os.path.join(PROJECT, ".venv", "Scripts", "python.exe")
RUN_PY = os.path.join(PROJECT, "run.py")

VIDEOS = {
    "11": "11-非有意触球-25中场-30-233-北京-梅州-越位-获利-6.mp4",
    "12": "12-非有意触球-25中超-5-37-成都蓉城-大连英博海发-越位-获利-21.mp4",
    "13": "13-有意处理球.mp4",
    "14": "14-阻碍视线-25中超-28-224-上海申花-大连英博-越位-干扰比赛-45+1.mp4",
}

# 后面跟一个值的参数，透传时必须成对，否则会被当成视频编号
VALUE_FLAGS = {
    "--calib-mode", "--calibration", "-c",
    "--max-frames", "-m", "--resize", "-r",
    "--every", "-e", "--skip", "-s",
    "--attack-dir", "--confidence", "--yolo-model",
    "--output", "-o", "--mode", "--port",
}


def parse_args(argv):
    """拆出位置参数（视频编号）和要透传给 run.py 的参数。"""
    positional, flags = [], []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("-"):
            if a in VALUE_FLAGS and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                flags += [a, argv[i + 1]]
                i += 2
            else:
                flags += [a]
                i += 1
        else:
            positional.append(a)
            i += 1
    return positional, flags


def flag_value(flags, *names):
    """取出某个带值参数的值，未指定返回 None。"""
    for n in names:
        if n in flags:
            idx = flags.index(n)
            if idx + 1 < len(flags):
                return flags[idx + 1]
    return None


def main():
    positional, flags = parse_args(sys.argv[1:])

    key = positional[0] if positional else "13"
    if key not in VIDEOS:
        print(f"未知视频编号: {key}，可选: {list(VIDEOS.keys())}")
        sys.exit(1)

    video = os.path.join(PROJECT, VIDEOS[key])
    if not os.path.exists(video):
        print(f"视频不存在: {video}")
        sys.exit(1)

    cmd = [VENV_PY, RUN_PY, "--video", video]

    mode = flag_value(flags, "--calib-mode")
    calib_file = flag_value(flags, "--calibration", "-c")

    # center_circle 便捷：未显式给标定文件时，自动找 output/<视频名>_center_circle_calib.json
    if mode == "center_circle" and not calib_file:
        stem = os.path.splitext(VIDEOS[key])[0]
        cand = os.path.join(PROJECT, "output", f"{stem}_center_circle_calib.json")
        if os.path.exists(cand):
            calib_file = cand
        else:
            print(f"⚠ 未找到 center_circle 标定文件: {cand}")
            print("  将进入交互式点击标定")

    if calib_file:
        cmd += ["--calibration", str(calib_file)]

    # 未指定任何标定来源时保持原默认：auto
    if mode is None and calib_file is None:
        cmd += ["--calib-mode", "auto"]

    cmd += flags

    print("=" * 70)
    print("  Python :", VENV_PY)
    print("  视频   :", os.path.basename(video))
    print("  标定   :", mode or ("标定文件" if calib_file else "auto"))
    if calib_file:
        print("  标定档 :", os.path.basename(str(calib_file)))
    print("  命令   :", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    print("=" * 70)
    sys.stdout.flush()

    proc = subprocess.run(cmd, encoding="utf-8", errors="replace")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
