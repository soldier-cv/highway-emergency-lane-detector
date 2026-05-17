#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
应急车道违章检测系统 - GUI界面
- emergency_lane/traffic_violation_gui.py  GUI界面
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path

# ============ 核心检测逻辑 ============

from models.config import YOLO_MODEL_PATH, YOLO_DEVICE, PROJECT_ROOT


def _get_project_root():
    """获取项目根目录（emergency_lane的上级目录）"""
    return PROJECT_ROOT


def get_model_path():
    """返回YOLOv8s模型路径（统一配置）"""
    return YOLO_MODEL_PATH

def ensure_model():
    """确保模型存在，返回模型路径"""
    if YOLO_MODEL_PATH:
        return YOLO_MODEL_PATH
    return None

def init_hyperlpr3():
    """初始化车牌识别（OpenVINO GPU加速版）"""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from lpr3_openvino import LicensePlateCatcherOV
        catcher = LicensePlateCatcherOV(device="GPU", det_level=1)
        return catcher
    except Exception as e:
        # 回退到原版CPU
        import hyperlpr3 as lpr3
        zip_path = os.path.expanduser("~/.hyperlpr3/20230229.zip")
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except:
                pass
        catcher = lpr3.LicensePlateCatcher(detect_level=lpr3.DETECT_LEVEL_HIGH)
        return catcher

def recognize_plate(catcher, img, bbox):
    """从图像中识别车牌"""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None, 0
    
    plate_img = img[y1:y2, x1:x2]
    if plate_img.size == 0:
        return None, 0
    
    try:
        results = catcher(plate_img)
        if results and len(results) > 0:
            best = max(results, key=lambda r: r[1] if len(r) > 1 else 0)
            plate_text = best[0]
            confidence = best[1] if len(best) > 1 else 0
            if isinstance(plate_text, str) and len(plate_text) >= 7:
                return plate_text, confidence
    except:
        pass
    return None, 0

def enhance_recognize_plate(catcher, img, vehicle_bbox, strategies=None):
    """增强车牌识别：多种裁剪策略"""
    if strategies is None:
        strategies = ['full', 'lower_half', 'lower_third', 'clahe', 'scale2x']
    
    x1, y1, x2, y2 = map(int, vehicle_bbox)
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None, 0
    
    vehicle_img = img[y1:y2, x1:x2]
    if vehicle_img.size == 0:
        return None, 0
    
    vh, vw = vehicle_img.shape[:2]
    candidates = []
    
    for strategy in strategies:
        try:
            if strategy == 'full':
                crop = vehicle_img
            elif strategy == 'lower_half':
                crop = vehicle_img[vh//2:, :]
            elif strategy == 'lower_third':
                crop = vehicle_img[2*vh//3:, :]
            elif strategy == 'clahe':
                gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                crop = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            elif strategy == 'scale2x':
                crop = cv2.resize(vehicle_img, (vw*2, vh*2), interpolation=cv2.INTER_CUBIC)
            else:
                continue
            
            if crop.size == 0:
                continue
            
            results = catcher(crop)
            if results:
                for r in results:
                    plate_text = r[0]
                    conf = r[1] if len(r) > 1 else 0
                    if isinstance(plate_text, str) and len(plate_text) >= 7:
                        candidates.append((plate_text, conf, strategy))
        except:
            continue
    
    if not candidates:
        return None, 0
    
    # 按置信度排序
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0], candidates[0][1]

def multi_frame_recognize(catcher, cap, frame_idx, vehicle_bbox, scan_range=30):
    """多帧扫描识别车牌"""
    plate_votes = {}
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for offset in range(-scan_range, scan_range + 1, 2):
        fi = frame_idx + offset
        if fi < 0 or fi >= total_frames:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue

        plate_text, conf = enhance_recognize_plate(catcher, frame, vehicle_bbox)
        if plate_text and conf > 0.5:
            plate_votes[plate_text] = plate_votes.get(plate_text, 0) + 1

    if plate_votes:
        best_plate = max(plate_votes, key=plate_votes.get)
        return best_plate, plate_votes[best_plate]
    return None, 0


def multi_frame_recognize_with_frame(catcher, cap, frame_idx, vehicle_bbox, scan_range=30):
    """多帧扫描识别车牌，返回最佳识别帧号"""
    plate_votes = {}
    plate_best_frame = {}  # 记录每个车牌最佳识别帧
    plate_best_conf = {}   # 记录每个车牌最高置信度
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for offset in range(-scan_range, scan_range + 1, 2):
        fi = frame_idx + offset
        if fi < 0 or fi >= total_frames:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue

        plate_text, conf = enhance_recognize_plate(catcher, frame, vehicle_bbox)
        if plate_text and conf > 0.5:
            plate_votes[plate_text] = plate_votes.get(plate_text, 0) + 1
            # 记录置信度最高的帧
            if plate_text not in plate_best_conf or conf > plate_best_conf[plate_text]:
                plate_best_conf[plate_text] = conf
                plate_best_frame[plate_text] = fi

    if plate_votes:
        best_plate = max(plate_votes, key=plate_votes.get)
        best_frame = plate_best_frame.get(best_plate, frame_idx)
        return best_plate, best_frame, plate_votes[best_plate]
    return None, frame_idx, 0

def generate_violation_clips(video_path, violation_times, output_dir, progress_callback=None):
    """根据违章时间点生成30秒视频片段（使用FFmpeg直接剪切，不重新编码）"""
    if progress_callback is None:
        progress_callback = lambda msg, pct: None

    clip_duration = 30       # 目标每段30秒
    pre_violation_pad = 3    # 违章前保留3秒
    min_gap_to_cut = 10      # 无违章超过10秒可剪掉
    post_violation_pad = 5   # 违章后多留5秒

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0
    cap.release()

    violation_times = sorted(violation_times)
    clips = []  # (start_sec, end_sec, label)

    if violation_times:
        current_start = max(0, violation_times[0] - pre_violation_pad)
        current_end = violation_times[0]

        for vt in violation_times[1:]:
            gap = vt - current_end
            if gap <= min_gap_to_cut:
                current_end = vt
            else:
                clips.append((current_start, current_end + post_violation_pad, f"clip_{len(clips)+1:02d}"))
                current_start = max(0, vt - pre_violation_pad)
                current_end = vt

        clips.append((current_start, min(current_end + post_violation_pad, video_duration),
                      f"clip_{len(clips)+1:02d}"))

        # 如果某段超过30秒，按违章点拆分
        final_clips = []
        for start, end, label in clips:
            dur = end - start
            if dur <= clip_duration:
                final_clips.append((start, end, label))
            else:
                seg_violations = [vt for vt in violation_times if start <= vt <= end]
                seg_start = start
                for si, vt in enumerate(seg_violations):
                    ideal_end = vt + post_violation_pad
                    next_vt = seg_violations[si + 1] if si + 1 < len(seg_violations) else end
                    if next_vt - vt <= min_gap_to_cut:
                        continue
                    clip_end = min(ideal_end, seg_start + clip_duration)
                    if clip_end <= seg_start:
                        clip_end = seg_start + clip_duration
                    clip_end = min(clip_end, end)
                    final_clips.append((seg_start, clip_end, f"{label}_p{si+1}"))
                    seg_start = max(next_vt - pre_violation_pad, clip_end)

                if seg_start < end:
                    final_clips.append((seg_start, end, f"{label}_p{len(seg_violations)+1}"))

        clips = final_clips

    clips_dir = os.path.join(output_dir, "举报视频片段")
    os.makedirs(clips_dir, exist_ok=True)

    clip_results = []
    for clip_idx, (start_sec, end_sec, label) in enumerate(clips):
        dur = end_sec - start_sec
        clip_filename = f"{label}_{start_sec:.0f}s-{end_sec:.0f}s_{dur:.0f}s.mp4"
        clip_path = os.path.join(clips_dir, clip_filename)

        # 用FFmpeg直接剪切，不重新编码（速度快，保持原始画质）
        ffmpeg_cmd = f'ffmpeg -y -ss {start_sec:.2f} -i "{video_path}" -t {dur:.2f} -c copy "{clip_path}"'
        os.system(ffmpeg_cmd + " >nul 2>&1")

        clip_results.append({
            'index': clip_idx + 1,
            'label': label,
            'start': start_sec,
            'end': end_sec,
            'duration': dur,
            'filename': clip_filename,
            'path': clip_path
        })
        progress_callback(f"  片段 {clip_idx+1}: {start_sec:.0f}s-{end_sec:.0f}s ({dur:.0f}s)", -1)

    return clip_results


def run_detection(video_path, lane_x=0.84, lane_width=0.16, lane_top=0.15,
                  detection_scale=0.5, conf_threshold=0.5, use_gpu=True,
                  progress_callback=None):
    """运行应急车道违章检测，通过progress_callback报告进度"""

    if progress_callback is None:
        progress_callback = lambda msg, pct: None
    
    # 1. 初始化模型
    progress_callback("正在加载YOLOv8模型...", 5)
    model_path = ensure_model()
    if not model_path:
        raise RuntimeError("无法加载YOLOv8模型，请运行 python setup_models.py 下载模型")

    progress_callback(f"  模型路径: {model_path}", -1)

    from ultralytics import YOLO
    device = YOLO_DEVICE if use_gpu else "cpu"
    try:
        model = YOLO(model_path, task="detect")
        progress_callback(f"  YOLOv8加载成功，设备: {device}", -1)
    except Exception as e:
        progress_callback(f"  OpenVINO加载失败({e})，回退CPU...", -1)
        pt_path = os.path.join(_get_project_root(), "yolov8s.pt")
        if os.path.exists(pt_path):
            model = YOLO(pt_path, task="detect")
            device = "cpu"
        else:
            raise

    progress_callback("正在初始化车牌识别引擎...", 10)
    catcher = init_hyperlpr3()
    progress_callback("  车牌识别引擎就绪", -1)
    
    # 2. 打开视频
    progress_callback("正在打开视频...", 15)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    progress_callback(f"视频: {width}x{height} @ {fps:.0f}fps, {duration:.1f}秒, {total_frames}帧", 18)
    
    # 3. 检测循环
    progress_callback("正在检测车辆...", 20)
    
    emergency_lane_x = int(width * lane_x)
    emergency_lane_right = int(width * (lane_x + lane_width))
    emergency_lane_top = int(height * lane_top)
    
    # 跟踪违章车辆
    tracked_violations = {}  # track_id -> {first_frame, bbox, plate, frames}
    active_tracks = {}  # track_id -> last_seen_frame
    violation_id_counter = 0
    detection_interval = 3  # 每3帧检测一次
    
    start_time = time.time()
    
    for frame_idx in range(0, total_frames, detection_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        # 降采样检测
        if detection_scale < 1.0:
            small = cv2.resize(frame, (0, 0), fx=detection_scale, fy=detection_scale)
        else:
            small = frame
        
        # YOLO检测
        results = model.predict(small, conf=conf_threshold, classes=[2, 5, 7], 
                               device=device, verbose=False, iou=0.45)
        
        if not results or len(results) == 0:
            continue
        
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue
        
        # 处理检测框
        boxes = result.boxes
        for i in range(len(boxes)):
            # 获取原始坐标
            xyxy = boxes.xyxy[i].cpu().numpy()
            x1 = int(xyxy[0] / detection_scale)
            y1 = int(xyxy[1] / detection_scale)
            x2 = int(xyxy[2] / detection_scale)
            y2 = int(xyxy[3] / detection_scale)
            conf = float(boxes.conf[i].cpu())
            cls = int(boxes.cls[i].cpu())
            track_id = int(boxes.id[i]) if boxes.id is not None else None
            
            # 检查是否在应急车道区域
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            if cx >= emergency_lane_x and cx <= emergency_lane_right and cy >= emergency_lane_top:
                # 使用简单的IoU跟踪
                best_match = None
                best_iou = 0.3
                
                for tid, info in list(active_tracks.items()):
                    bx1, by1, bx2, by2 = info['bbox']
                    # 计算IoU
                    ix1 = max(x1, bx1)
                    iy1 = max(y1, by1)
                    ix2 = min(x2, bx2)
                    iy2 = min(y2, by2)
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        area1 = (x2 - x1) * (y2 - y1)
                        area2 = (bx2 - bx1) * (by2 - by1)
                        iou = inter / (area1 + area2 - inter)
                        if iou > best_iou:
                            best_iou = iou
                            best_match = tid
                
                if best_match is not None:
                    # 更新现有跟踪
                    active_tracks[best_match]['bbox'] = (x1, y1, x2, y2)
                    active_tracks[best_match]['frame'] = frame_idx
                    if best_match in tracked_violations:
                        tracked_violations[best_match]['last_frame'] = frame_idx
                        tracked_violations[best_match]['bbox'] = (x1, y1, x2, y2)
                else:
                    # 新违章
                    violation_id_counter += 1
                    vid = f"V{violation_id_counter:03d}"
                    tracked_violations[vid] = {
                        'first_frame': frame_idx,
                        'last_frame': frame_idx,
                        'bbox': (x1, y1, x2, y2),
                        'plate': None,
                        'confidence': 0,
                        'time': frame_idx / fps
                    }
                    active_tracks[vid] = {'bbox': (x1, y1, x2, y2), 'frame': frame_idx}
        
        # 清理长时间未见到的跟踪
        stale = [tid for tid, info in active_tracks.items() if frame_idx - info['frame'] > 30]
        for tid in stale:
            del active_tracks[tid]
        
        # 进度回调
        pct = 20 + int(60 * frame_idx / total_frames)
        elapsed = time.time() - start_time
        eta = elapsed / (frame_idx + 1) * (total_frames - frame_idx) / detection_interval
        progress_callback(
            f"检测中: {frame_idx}/{total_frames}帧 ({frame_idx/total_frames*100:.0f}%) | "
            f"已发现{len(tracked_violations)}起违章 | ETA: {eta:.0f}秒",
            pct
        )
    
    cap.release()
    det_elapsed = time.time() - start_time
    progress_callback(f"车辆检测完成，耗时 {det_elapsed:.1f}秒，发现 {len(tracked_violations)} 起违章", -1)

    # 4. 车牌识别阶段
    violation_list = list(tracked_violations.items())
    n_violations = len(violation_list)
    progress_callback(f"检测完成！发现{n_violations}起违章，开始识别车牌...", 82)

    cap = cv2.VideoCapture(video_path)
    t_plate_start = time.time()

    for i, (vid, vinfo) in enumerate(violation_list):
        progress_callback(
            f"[{i+1}/{n_violations}] 识别第{vinfo['first_frame']}帧车牌...",
            82 + int(15 * (i + 1) / max(n_violations, 1))
        )

        # 记录最佳识别帧，默认为违章检测帧
        vinfo['best_frame'] = vinfo['first_frame']

        # 先尝试当前帧直接识别
        cap.set(cv2.CAP_PROP_POS_FRAMES, vinfo['first_frame'])
        ret, frame = cap.read()
        if not ret:
            progress_callback(f"  第{vinfo['first_frame']}帧读取失败，跳过", -1)
            continue

        t0 = time.time()
        plate, conf = enhance_recognize_plate(catcher, frame, vinfo['bbox'])
        dt = time.time() - t0
        if plate and conf > 0.5:
            vinfo['plate'] = plate
            vinfo['confidence'] = conf
            progress_callback(f"  首帧识别成功: {plate} ({conf:.2f}) [{dt:.1f}秒]", -1)
            continue

        progress_callback(f"  首帧未识别 [{dt:.1f}秒]，多帧扫描中...", -1)
        t0 = time.time()
        best_plate, best_frame, votes = multi_frame_recognize_with_frame(catcher, cap, vinfo['first_frame'], vinfo['bbox'], scan_range=30)
        dt = time.time() - t0
        if best_plate:
            vinfo['plate'] = best_plate
            vinfo['confidence'] = min(votes / 10, 1.0)
            vinfo['best_frame'] = best_frame  # 记录最佳识别帧
            progress_callback(f"  多帧扫描识别成功: {best_plate} (票数{votes}) [{dt:.1f}秒]", -1)
        else:
            progress_callback(f"  多帧扫描仍未识别 [{dt:.1f}秒]", -1)

    plate_elapsed = time.time() - t_plate_start
    progress_callback(f"车牌识别完成，耗时 {plate_elapsed:.1f}秒", 82)

    cap.release()

    # 5. 生成30秒违章视频片段
    output_dir = os.path.dirname(video_path)
    video_name = Path(video_path).stem

    violation_times = [vinfo['time'] for vinfo in tracked_violations.values()]
    if violation_times:
        progress_callback(f"正在生成30秒违章视频片段...", 85)
        clips = generate_violation_clips(video_path, violation_times, output_dir, progress_callback)
        progress_callback(f"  生成 {len(clips)} 个视频片段", -1)
    else:
        clips = []

    # 6. 生成结果
    progress_callback("正在生成违章截图...", 95)

    results = []
    for vid, vinfo in tracked_violations.items():
        timestamp = vinfo['time']
        mins = int(timestamp // 60)
        secs = int(timestamp % 60)

        # 计算该违章属于哪个剪辑片段
        clip_index = -1
        clip_label = ""
        clip_relative_time = None
        for ci, clip in enumerate(clips):
            if clip['start'] <= timestamp <= clip['end']:
                clip_index = ci
                clip_label = clip['label']
                clip_relative_time = round(timestamp - clip['start'], 1)
                break

        results.append({
            'id': vid,
            'plate': vinfo['plate'] or '未识别',
            'time': f"{mins:02d}:{secs:02d}",
            'time_seconds': timestamp,
            'frame': vinfo['first_frame'],
            'best_frame': vinfo.get('best_frame', vinfo['first_frame']),  # 车牌识别最佳帧
            'bbox': list(vinfo['bbox']),
            'confidence': vinfo['confidence'],
            'clip_index': clip_index,
            'clip_label': clip_label,
            'clip_relative_time': clip_relative_time
        })
    
    # 保存违章截图
    snapshot_dir = os.path.join(output_dir, f"{video_name}_snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    for r in results:
        # 使用车牌识别最佳帧截图，确保截到车牌
        snap_frame = r.get('best_frame', r['frame'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, snap_frame)
        ret, frame = cap.read()
        if not ret:
            # 回退到违章检测帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, r['frame'])
            ret, frame = cap.read()
        if ret:
            # 绘制标注
            x1, y1, x2, y2 = r['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            label = f"{r['plate']} ({r['confidence']:.0%})"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            # 应急车道区域
            h, w = frame.shape[:2]
            cv2.rectangle(frame,
                         (int(w*lane_x), int(h*lane_top)),
                         (int(w*(lane_x+lane_width)), h),
                         (0, 255, 255), 2)

            snap_path = os.path.join(snapshot_dir, f"violation_{r['id']}.jpg")
            # 用imencode避免中文路径问题
            try:
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                buf.tofile(snap_path)
            except:
                cv2.imwrite(snap_path, frame)
            r['snapshot'] = snap_path
    cap.release()
    progress_callback(f"  截图已保存到 {snapshot_dir}", -1)

    progress_callback("正在生成HTML报告...", 97)

    # 生成HTML报告
    html_path = generate_html_report(results, video_path, output_dir, video_name,
                                      lane_x, lane_width, lane_top, clips)

    total_time = time.time() - start_time
    recognized = sum(1 for r in results if r['plate'] != '未识别')
    progress_callback(f"检测完成！违章{len(results)}起，识别{recognized}个车牌，{len(clips)}个视频片段，总耗时{total_time:.1f}秒", 100)
    progress_callback(f"  HTML: {html_path}", -1)

    return {
        'violations': results,
        'html_path': html_path,
        'snapshot_dir': snapshot_dir,
        'clips': clips,
        'total_time': total_time,
        'total_violations': len(results),
        'recognized_plates': sum(1 for r in results if r['plate'] != '未识别')
    }

def generate_html_report(results, video_path, output_dir, video_name, lane_x, lane_width, lane_top, clips=None):
    """生成HTML报告"""
    html_path = os.path.join(output_dir, f"{video_name}_violation_report.html")

    rows = ""
    for r in results:
        plate_color = "#27ae60" if r['plate'] != '未识别' else "#e74c3c"
        snap = r.get('snapshot', '')
        # 使用相对路径，确保HTML能正确引用截图
        snap_rel = os.path.relpath(snap, output_dir) if snap else ''

        # 片段信息
        clip_info = ""
        if r.get('clip_label') and r.get('clip_relative_time') is not None:
            clip_info = f"片段{r['clip_label']} 第{r['clip_relative_time']:.1f}秒"

        rows += f"""
        <tr>
            <td>{r['id']}</td>
            <td style="color:{plate_color};font-weight:bold;font-size:1.2em">{r['plate']}</td>
            <td>{r['time']}</td>
            <td>{r['confidence']:.0%}</td>
            <td>{clip_info if clip_info else '-'}</td>
            <td>{'<a href="' + snap_rel + '" target="_blank">查看</a>' if snap_rel else '-'}</td>
        </tr>"""

    # 剪辑片段表格
    clips_html = ""
    if clips:
        clips_dir_name = "举报视频片段"
        for clip in clips:
            # 计算该片段包含的违章
            clip_violations = [v for v in results if v.get('clip_label') == clip['label']]
            plates_str = ", ".join(set(v['plate'] for v in clip_violations if v['plate'] != '未识别'))
            clip_file_rel = f"{clips_dir_name}/{clip['filename']}"
            clips_html += f"""
        <tr>
            <td>{clip['index']}</td>
            <td><a href="{clip_file_rel}" target="_blank">{clip['filename']}</a></td>
            <td>{clip['start']:.0f}s - {clip['end']:.0f}s</td>
            <td>{clip['duration']:.0f}s</td>
            <td style="color:#c00;font-weight:bold;">{plates_str or '-'}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>应急车道违章检测报告 - {video_name}</title>
<style>
body {{ font-family: 'Microsoft YaHei', sans-serif; max-width: 1000px; margin: 40px auto; background: #f5f5f5; }}
.card {{ background: white; border-radius: 12px; padding: 30px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 20px; }}
h1 {{ color: #2c3e50; border-bottom: 3px solid #e74c3c; padding-bottom: 10px; }}
h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px; margin-top: 30px; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
th {{ background: #2c3e50; color: white; padding: 12px; text-align: center; }}
td {{ padding: 10px; text-align: center; border-bottom: 1px solid #eee; }}
tr:hover {{ background: #f0f0f0; }}
.stats {{ display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }}
.stat {{ background: #ecf0f1; border-radius: 8px; padding: 15px 25px; text-align: center; }}
.stat .num {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
.stat .label {{ color: #7f8c8d; margin-top: 5px; }}
a {{ color: #3498db; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<div class="card">
<h1>🚨 应急车道违章检测报告</h1>
<p><strong>视频文件：</strong>{os.path.basename(video_path)}</p>
<p><strong>检测时间：</strong>{time.strftime('%Y-%m-%d %H:%M:%S')}</p>
<p><strong>应急车道区域：</strong>X={lane_x:.2f}, 宽度={lane_width:.2f}, 顶部={lane_top:.2f}</p>
<div class="stats">
    <div class="stat"><div class="num">{len(results)}</div><div class="label">违章总数</div></div>
    <div class="stat"><div class="num" style="color:#27ae60">{sum(1 for r in results if r['plate'] != '未识别')}</div><div class="label">车牌识别成功</div></div>
    <div class="stat"><div class="num" style="color:#e74c3c">{sum(1 for r in results if r['plate'] == '未识别')}</div><div class="label">未识别</div></div>
    <div class="stat"><div class="num" style="color:#3498db">{len(clips) if clips else 0}</div><div class="label">视频片段</div></div>
</div>
</div>

<div class="card">
<h2>📋 违章详情</h2>
<table>
<tr><th>编号</th><th>车牌号</th><th>时间</th><th>置信度</th><th>视频片段</th><th>截图</th></tr>
{rows}
</table>
</div>

{"<div class='card'>" if clips else ""}
{"<h2>🎬 举报视频片段（约30秒）</h2>" if clips else ""}
{"<p>片段保存在: <code>举报视频片段</code> 子目录</p>" if clips else ""}
{"<table><tr><th>#</th><th>文件名</th><th>时间范围</th><th>时长</th><th>包含车牌</th></tr>" if clips else ""}
{clips_html}
{"</table></div>" if clips else ""}
</body>
</html>"""

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return html_path


# ============ GUI 界面 ============

class TrafficViolationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("应急车道违章检测系统")
        self.root.geometry("800x780")
        self.root.resizable(True, True)
        self.root.configure(bg="#f0f2f5")
        
        self.is_running = False
        self.result = None
        
        self._build_ui()
    
    def _build_ui(self):
        # 标题栏
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        tk.Label(title_frame, text="🚨 应急车道违章检测系统", 
                font=("Microsoft YaHei", 18, "bold"),
                bg="#2c3e50", fg="white").pack(pady=12)
        
        # 主容器
        main = tk.Frame(self.root, bg="#f0f2f5")
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # ---- 视频选择 ----
        video_frame = tk.LabelFrame(main, text=" 📹 视频文件 ", font=("Microsoft YaHei", 11, "bold"),
                                    bg="white", fg="#2c3e50", padx=10, pady=8)
        video_frame.pack(fill=tk.X, pady=(0, 8))
        
        self.video_var = tk.StringVar()
        video_entry = tk.Entry(video_frame, textvariable=self.video_var, font=("Consolas", 10),
                              bg="#fafafa", relief=tk.FLAT, bd=0)
        video_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8), ipady=6)
        
        btn_browse = tk.Button(video_frame, text="浏览...", font=("Microsoft YaHei", 10),
                              bg="#3498db", fg="white", relief=tk.FLAT, cursor="hand2",
                              command=self._browse_video)
        btn_browse.pack(side=tk.RIGHT, ipady=3, ipadx=10)
        
        # ---- 应急车道参数 ----
        param_frame = tk.LabelFrame(main, text=" ⚙️ 应急车道区域参数 ", font=("Microsoft YaHei", 11, "bold"),
                                    bg="white", fg="#2c3e50", padx=10, pady=8)
        param_frame.pack(fill=tk.X, pady=(0, 8))
        
        params = [
            ("车道X位置", "lane_x", 0.84, 0.0, 1.0, 0.01),
            ("车道宽度", "lane_width", 0.16, 0.01, 0.5, 0.01),
            ("车道顶部Y", "lane_top", 0.15, 0.0, 1.0, 0.01),
        ]
        
        self.param_vars = {}
        for i, (label, key, default, from_, to_, step) in enumerate(params):
            tk.Label(param_frame, text=label, font=("Microsoft YaHei", 10),
                    bg="white", fg="#555").grid(row=i, column=0, sticky="w", pady=2)
            var = tk.DoubleVar(value=default)
            self.param_vars[key] = var
            spin = tk.Spinbox(param_frame, from_=from_, to=to_, increment=step,
                             textvariable=var, width=8, font=("Consolas", 10),
                             bg="#fafafa", relief=tk.FLAT, bd=1)
            spin.grid(row=i, column=1, padx=(10, 20), pady=2)
        
        # 高级选项
        adv_frame = tk.Frame(param_frame, bg="white")
        adv_frame.grid(row=len(params), column=0, columnspan=2, sticky="w", pady=(8, 0))
        
        self.gpu_var = tk.BooleanVar(value=True)
        tk.Checkbutton(adv_frame, text="使用 Intel Arc GPU 加速", variable=self.gpu_var,
                       font=("Microsoft YaHei", 10), bg="white", fg="#555",
                       activebackground="white").pack(side=tk.LEFT)
        
        self.conf_var = tk.DoubleVar(value=0.5)
        tk.Label(adv_frame, text="  置信度阈值:", font=("Microsoft YaHei", 10),
                bg="white", fg="#555").pack(side=tk.LEFT, padx=(20, 5))
        tk.Spinbox(adv_frame, from_=0.1, to=1.0, increment=0.05,
                  textvariable=self.conf_var, width=5, font=("Consolas", 10),
                  bg="#fafafa", relief=tk.FLAT, bd=1).pack(side=tk.LEFT)
        
        self.scale_var = tk.DoubleVar(value=0.5)
        tk.Label(adv_frame, text="  检测缩放:", font=("Microsoft YaHei", 10),
                bg="white", fg="#555").pack(side=tk.LEFT, padx=(15, 5))
        tk.Spinbox(adv_frame, from_=0.25, to=1.0, increment=0.25,
                  textvariable=self.scale_var, width=5, font=("Consolas", 10),
                  bg="#fafafa", relief=tk.FLAT, bd=1).pack(side=tk.LEFT)
        
        # ---- 控制按钮 ----
        ctrl_frame = tk.Frame(main, bg="#f0f2f5")
        ctrl_frame.pack(fill=tk.X, pady=8)
        
        self.btn_start = tk.Button(ctrl_frame, text="▶ 开始检测", font=("Microsoft YaHei", 12, "bold"),
                                   bg="#27ae60", fg="white", relief=tk.FLAT, cursor="hand2",
                                   command=self._start_detection)
        self.btn_start.pack(side=tk.LEFT, ipady=6, ipadx=20)

        self.btn_stop = tk.Button(ctrl_frame, text="⏹ 停止", font=("Microsoft YaHei", 12, "bold"),
                                  bg="#95a5a6", fg="white", relief=tk.FLAT, cursor="hand2",
                                  command=self._stop_detection, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=(10, 0), ipady=6, ipadx=15)

        self.btn_report = tk.Button(ctrl_frame, text="📊 查看报告", font=("Microsoft YaHei", 12, "bold"),
                                    bg="#3498db", fg="white", relief=tk.FLAT, cursor="hand2",
                                    command=self._open_report, state=tk.DISABLED)
        self.btn_report.pack(side=tk.RIGHT, ipady=6, ipadx=15)
        
        # ---- 进度条 ----
        prog_frame = tk.Frame(main, bg="#f0f2f5")
        prog_frame.pack(fill=tk.X, pady=(0, 8))
        
        self.progress = ttk.Progressbar(prog_frame, mode='determinate', length=400)
        self.progress.pack(fill=tk.X, pady=(0, 4))
        
        self.status_var = tk.StringVar(value="就绪 - 请选择视频文件")
        self.status_label = tk.Label(prog_frame, textvariable=self.status_var,
                                     font=("Microsoft YaHei", 10), bg="#f0f2f5", fg="#555",
                                     anchor="w")
        self.status_label.pack(fill=tk.X)
        
        # ---- 结果区域（上半：表格，下半：日志）----
        bottom_area = tk.PanedWindow(main, orient=tk.VERTICAL, bg="#f0f2f5", sashwidth=4)
        bottom_area.pack(fill=tk.BOTH, expand=True)

        # 结果表格
        result_frame = tk.LabelFrame(bottom_area, text=" 📋 检测结果 ", font=("Microsoft YaHei", 11, "bold"),
                                     bg="white", fg="#2c3e50", padx=10, pady=8)
        bottom_area.add(result_frame, stretch="always")

        columns = ("id", "plate", "time", "confidence")
        self.tree = ttk.Treeview(result_frame, columns=columns, show="headings", height=5)
        self.tree.heading("id", text="编号")
        self.tree.heading("plate", text="车牌号")
        self.tree.heading("time", text="时间")
        self.tree.heading("confidence", text="置信度")

        self.tree.column("id", width=80, anchor="center")
        self.tree.column("plate", width=200, anchor="center")
        self.tree.column("time", width=100, anchor="center")
        self.tree.column("confidence", width=100, anchor="center")

        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 运行日志
        log_frame = tk.LabelFrame(bottom_area, text=" 📝 运行日志 ", font=("Microsoft YaHei", 11, "bold"),
                                  bg="white", fg="#2c3e50", padx=10, pady=8)
        bottom_area.add(log_frame, stretch="never")

        self.log_text = tk.Text(log_frame, height=8, font=("Consolas", 9), bg="#1e1e1e",
                                fg="#cccccc", relief=tk.FLAT, wrap=tk.WORD, state=tk.DISABLED)
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 底部状态栏
        bottom = tk.Frame(self.root, bg="#2c3e50", height=30)
        bottom.pack(fill=tk.X, side=tk.BOTTOM)
        bottom.pack_propagate(False)
        self.bottom_label = tk.Label(bottom, text="Intel Arc 140V GPU | YOLOv8s + HyperLPR3 (OpenVINO) | 30s剪辑 + 车牌颜色 + 时间戳",
                                     font=("Microsoft YaHei", 9), bg="#2c3e50", fg="#95a5a6")
        self.bottom_label.pack(pady=4)
    
    def _log(self, msg):
        """向日志区域追加一行（线程安全）"""
        def _append():
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, msg + "\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
        self.root.after(0, _append)

    def _browse_video(self):
        path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mkv *.mov *.flv *.wmv"), ("所有文件", "*.*")]
        )
        if path:
            self.video_var.set(path)
    
    def _start_detection(self):
        video_path = self.video_var.get().strip()
        if not video_path:
            messagebox.showwarning("提示", "请先选择视频文件！")
            return
        if not os.path.exists(video_path):
            messagebox.showerror("错误", f"视频文件不存在：{video_path}")
            return
        
        self.is_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_report.config(state=tk.DISABLED)
        self.progress['value'] = 0
        
        # 清空结果
        for item in self.tree.get_children():
            self.tree.delete(item)
        # 清空日志
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        # 启动检测线程
        thread = threading.Thread(target=self._run_detection_thread, daemon=True)
        thread.start()
    
    def _stop_detection(self):
        self.is_running = False
        self.status_var.set("正在停止...")
    
    def _run_detection_thread(self):
        try:
            result = run_detection(
                video_path=self.video_var.get().strip(),
                lane_x=self.param_vars['lane_x'].get(),
                lane_width=self.param_vars['lane_width'].get(),
                lane_top=self.param_vars['lane_top'].get(),
                detection_scale=self.scale_var.get(),
                conf_threshold=self.conf_var.get(),
                use_gpu=self.gpu_var.get(),
                progress_callback=self._progress_callback
            )
            self.result = result
            self.root.after(0, self._on_detection_done, result)
        except Exception as e:
            self.root.after(0, self._on_detection_error, str(e))
    
    def _progress_callback(self, msg, pct):
        if not self.is_running:
            raise InterruptedError("用户停止了检测")
        self._log(msg)
        if pct >= 0:
            self.root.after(0, self._update_progress, msg, pct)

    def _update_progress(self, msg, pct):
        self.progress['value'] = pct
        self.status_var.set(msg)
    
    def _on_detection_done(self, result):
        self.is_running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_report.config(state=tk.NORMAL)
        
        # 填充结果表格
        for v in result['violations']:
            self.tree.insert("", tk.END, values=(
                v['id'],
                v['plate'],
                v['time'],
                f"{v['confidence']:.0%}"
            ))
        
        recognized = result['recognized_plates']
        total = result['total_violations']
        t = result['total_time']
        
        self.status_var.set(
            f"✅ 检测完成！发现 {total} 起违章，{recognized}/{total} 个车牌识别成功，耗时 {t:.1f}秒"
        )
        
        messagebox.showinfo("检测完成", 
            f"检测完成！\n\n"
            f"违章总数：{total}\n"
            f"车牌识别成功：{recognized}/{total}\n"
            f"总耗时：{t:.1f}秒\n\n"
            f"HTML报告：{result['html_path']}")
    
    def _on_detection_error(self, error_msg):
        self.is_running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.status_var.set(f"❌ 检测失败：{error_msg}")
        messagebox.showerror("检测失败", error_msg)
    
    def _open_report(self):
        if self.result and self.result.get('html_path'):
            import webbrowser
            webbrowser.open(self.result['html_path'])
        else:
            messagebox.showwarning("提示", "暂无报告，请先运行检测")


def main():
    root = tk.Tk()
    app = TrafficViolationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
