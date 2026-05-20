"""共享证据输出工具。"""

import os
import shutil
import subprocess

import cv2

DEFAULT_CLIP_DURATION = 15
DEFAULT_PRE_VIOLATION_PAD = 3
DEFAULT_POST_VIOLATION_PAD = 5


def resolve_ffmpeg():
    cmd = shutil.which("ffmpeg")
    if cmd:
        return cmd

    for candidate in [r"C:\ffmpeg6.0\bin\ffmpeg.exe", r"C:\ffmpeg\bin\ffmpeg.exe"]:
        if os.path.isfile(candidate):
            return candidate

    return None


def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames / fps if fps > 0 else 0


def cut_video_clip(video_path, output_path, anchor_time, pre_violation_pad=DEFAULT_PRE_VIOLATION_PAD, clip_duration=DEFAULT_CLIP_DURATION, progress_callback=None):
    if progress_callback is None:
        progress_callback = lambda msg, pct: None

    ffmpeg_cmd = resolve_ffmpeg()
    if not ffmpeg_cmd:
        progress_callback("  [ERROR] ffmpeg 未安装，无法生成视频片段。", -1)
        return None

    video_duration = get_video_duration(video_path)
    start_sec = max(0, anchor_time - pre_violation_pad)
    end_sec = min(video_duration, start_sec + clip_duration)
    dur = max(0, end_sec - start_sec)
    if dur <= 0:
        return None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result = subprocess.run(
        [ffmpeg_cmd, "-y", "-ss", f"{start_sec:.2f}", "-i", video_path, "-t", f"{dur:.2f}", "-c", "copy", output_path],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        progress_callback(f"  [WARN] FFmpeg failed for {os.path.basename(output_path)}: {result.stderr[:200]}", -1)
        return None

    return {
        "start": start_sec,
        "end": end_sec,
        "duration": dur,
        "filename": os.path.basename(output_path),
        "path": output_path,
    }
