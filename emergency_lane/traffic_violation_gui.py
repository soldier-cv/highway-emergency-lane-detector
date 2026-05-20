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
from gpu_backend import get_gpu_backend, resolve_yolo_device
from evidence_utils import cut_video_clip
from report_utils import generate_html_report, dedupe_violations, write_manifest


def _get_project_root():
    """获取项目根目录（emergency_lane的上级目录）"""
    return PROJECT_ROOT


def get_model_path():
    """返回 YOLOv12s 模型路径（统一配置）"""
    return YOLO_MODEL_PATH

def ensure_model():
    """确保模型存在，返回模型路径"""
    if YOLO_MODEL_PATH:
        return YOLO_MODEL_PATH
    return None


def _validate_gpu_model_path(model_path, device):
    """校验模型格式与 GPU 设备是否匹配，避免静默回退到 CPU"""
    model_path_lower = str(model_path).lower()
    is_openvino_model = "_openvino_model" in model_path_lower
    is_pt_model = model_path_lower.endswith(".pt")

    if device == "intel:GPU" and not is_openvino_model:
        raise RuntimeError(
            "GPU加速已开启且当前使用的是 Intel/OpenVINO GPU，但检测模型不是 OpenVINO 导出模型。"
            "请提供 yolo12s_openvino_model 后再开启 GPU 加速，或关闭该开关改用 CPU。"
        )

    if device == "cuda:0" and not is_pt_model:
        raise RuntimeError(
            "GPU加速已开启且当前使用的是 NVIDIA CUDA，但检测模型不是 .pt 模型。"
            "请提供 yolo12s.pt 后再开启 GPU 加速，或关闭该开关改用 CPU。"
        )

def init_hyperlpr3(use_gpu=True):
    """初始化车牌识别
    use_gpu=True: 必须使用GPU，无GPU则报错
    use_gpu=False: 使用CPU
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    gpu = get_gpu_backend()

    if use_gpu:
        # 强制GPU模式：必须有可用GPU
        if gpu.cuda_available:
            from lpr3_ort import LicensePlateCatcherORT
            return LicensePlateCatcherORT(
                providers=["CUDAExecutionProvider"],
                det_level=1,
                allow_cpu_fallback=False,
            )
        elif gpu.openvino_available:
            from lpr3_openvino import LicensePlateCatcherOV
            return LicensePlateCatcherOV(
                device="GPU",
                det_level=1,
                allow_cpu_fallback=False,
            )
        else:
            raise RuntimeError("GPU加速已开启但未检测到可用GPU（需要NVIDIA CUDA或Intel OpenVINO），请关闭GPU加速或安装GPU驱动")
    else:
        # CPU模式
        try:
            import hyperlpr3 as lpr3
            zip_path = os.path.expanduser("~/.hyperlpr3/20230229.zip")
            if os.path.exists(zip_path):
                try:
                    os.remove(zip_path)
                except:
                    pass
            return lpr3.LicensePlateCatcher(detect_level=lpr3.DETECT_LEVEL_HIGH)
        except Exception:
            from lpr3_openvino import LicensePlateCatcherOV
            return LicensePlateCatcherOV(device="CPU", det_level=1)

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

def run_detection(video_path, lane_x=0.84, lane_width=0.16, lane_top=0.15,
                  detection_scale=0.75, conf_threshold=0.5, use_gpu=True,
                  clip_duration=15,
                  progress_callback=None):
    """运行应急车道违章检测，通过progress_callback报告进度
    callback签名: (message, percent, stage_info=None)
    stage_info: dict with keys like 'stage', 'model', 'gpu', etc.
    """

    if progress_callback is None:
        progress_callback = lambda msg, pct, **kw: None

    def _cb(msg, pct, **kwargs):
        progress_callback(msg, pct, **kwargs)

    start_time = time.time()

    # ====== 阶段1: 加载模型 (0%-8%) ======
    _cb("正在加载YOLOv12s模型...", 2, stage="加载模型")
    model_path = ensure_model()
    if not model_path:
        raise RuntimeError("无法加载模型，请确保 yolo12s.pt 或 yolo12s_openvino_model 存在")

    model_name = os.path.basename(model_path).replace('.pt', '').replace('_openvino_model', '')

    from ultralytics import YOLO

    gpu_backend = get_gpu_backend()
    gpu_info = gpu_backend.backend_name
    if gpu_backend.cuda_available:
        gpu_info = f"CUDA ({gpu_backend.gpu_name})"
    elif gpu_backend.openvino_available:
        gpu_info = "OpenVINO GPU"

    # 确定设备
    if use_gpu:
        if gpu_backend.cuda_available:
            device = "cuda:0"
        elif gpu_backend.openvino_available:
            device = "intel:GPU"
        else:
            raise RuntimeError("GPU加速已开启但未检测到可用GPU（需要NVIDIA CUDA或Intel OpenVINO），请关闭GPU加速或安装GPU驱动")
    else:
        device = "cpu"

    if use_gpu:
        _validate_gpu_model_path(model_path, device)

    try:
        model = YOLO(model_path, task="detect")
        _cb(f"模型加载完成: {model_name} @ {device}", 5, stage="加载模型",
            model=model_name, device=device, gpu=gpu_info)
    except Exception as e:
        if use_gpu:
            raise RuntimeError(f"GPU模型加载失败: {e}。请检查GPU驱动是否正常，或关闭GPU加速使用CPU模式")
        _cb(f"模型加载失败({e})，尝试CPU回退...", 4, stage="加载模型")
        pt_path = os.path.join(_get_project_root(), "yolo12s.pt")
        if os.path.exists(pt_path):
            model = YOLO(pt_path, task="detect")
            device = "cpu"
            model_name = os.path.basename(pt_path).replace('.pt', '')
        else:
            raise

    _cb("正在初始化车牌识别引擎...", 6, stage="加载模型")
    catcher = init_hyperlpr3(use_gpu=use_gpu)
    if use_gpu:
        if gpu_backend.cuda_available:
            lpr_name = "HyperLPR3 ONNXRuntime CUDA"
        else:
            lpr_name = "HyperLPR3 OpenVINO GPU"
    else:
        lpr_name = "HyperLPR3 CPU"
    _cb(f"车牌引擎就绪: {lpr_name}", 8, stage="加载模型", lpr=lpr_name)

    # ====== 阶段2: 打开视频 (8%-10%) ======
    _cb("正在打开视频...", 9, stage="打开视频")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    _cb(f"视频: {width}x{height} @ {fps:.0f}fps, {duration:.1f}秒, {total_frames}帧",
        10, stage="打开视频")

    # ====== 阶段3: 车辆检测 (10%-60%) ======
    _cb("正在检测违章车辆...", 10, stage="车辆检测")

    emergency_lane_x = int(width * lane_x)
    emergency_lane_right = int(width * (lane_x + lane_width))
    emergency_lane_top = int(height * lane_top)

    tracked_violations = {}
    active_tracks = {}
    violation_id_counter = 0
    detection_interval = 3

    last_progress_time = time.time()

    for frame_idx in range(0, total_frames, detection_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        if detection_scale < 1.0:
            small = cv2.resize(frame, (0, 0), fx=detection_scale, fy=detection_scale)
        else:
            small = frame

        results = model.predict(small, conf=conf_threshold, classes=[2, 5, 7],
                               device=device, verbose=False, iou=0.45)

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1 = int(xyxy[0] / detection_scale)
                y1 = int(xyxy[1] / detection_scale)
                x2 = int(xyxy[2] / detection_scale)
                y2 = int(xyxy[3] / detection_scale)
                conf = float(boxes.conf[i].cpu())

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                if cx >= emergency_lane_x and cx <= emergency_lane_right and cy >= emergency_lane_top:
                    best_match = None
                    best_iou = 0.3

                    for tid, info in list(active_tracks.items()):
                        bx1, by1, bx2, by2 = info['bbox']
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
                        active_tracks[best_match]['bbox'] = (x1, y1, x2, y2)
                        active_tracks[best_match]['frame'] = frame_idx
                        if best_match in tracked_violations:
                            tracked_violations[best_match]['last_frame'] = frame_idx
                            tracked_violations[best_match]['bbox'] = (x1, y1, x2, y2)
                    else:
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

        stale = [tid for tid, info in active_tracks.items() if frame_idx - info['frame'] > 30]
        for tid in stale:
            del active_tracks[tid]

        # 每秒最多更新2次进度（避免SSE风暴）
        now = time.time()
        if now - last_progress_time >= 0.5 or frame_idx + detection_interval >= total_frames:
            last_progress_time = now
            pct = 10 + int(50 * (frame_idx + 1) / total_frames)
            elapsed = now - start_time
            speed = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            eta = (total_frames - frame_idx - 1) / speed / 60 if speed > 0 else 0
            _cb(
                f"检测中: {frame_idx+1}/{total_frames}帧 ({(frame_idx+1)/total_frames*100:.0f}%) "
                f"| 速度:{speed:.1f}fps | 已发现{len(tracked_violations)}起违章 "
                f"| ETA:{eta:.1f}分钟",
                pct,
                stage="车辆检测",
                frame=frame_idx+1,
                total_frames=total_frames,
                speed=round(speed, 1),
                violations=len(tracked_violations),
                eta_min=round(eta, 1)
            )

    cap.release()
    det_elapsed = time.time() - start_time
    _cb(f"车辆检测完成: {det_elapsed:.1f}秒, 发现{len(tracked_violations)}起违章",
        60, stage="车辆检测")

    # ====== 阶段4: 车牌识别 (60%-90%) ======
    violation_list = list(tracked_violations.items())
    n_violations = len(violation_list)
    _cb(f"开始识别车牌 ({n_violations}起违章)...", 62, stage="车牌识别")

    cap = cv2.VideoCapture(video_path)
    t_plate_start = time.time()

    for i, (vid, vinfo) in enumerate(violation_list):
        pct = 62 + int(28 * (i + 1) / max(n_violations, 1))
        _cb(
            f"[{i+1}/{n_violations}] 识别第{vinfo['first_frame']}帧车牌...",
            pct,
            stage="车牌识别",
            plate_idx=i+1,
            plate_total=n_violations
        )

        vinfo['best_frame'] = vinfo['first_frame']

        cap.set(cv2.CAP_PROP_POS_FRAMES, vinfo['first_frame'])
        ret, frame = cap.read()
        if not ret:
            _cb(f"  第{vinfo['first_frame']}帧读取失败，跳过", -1, stage="车牌识别")
            continue

        t0 = time.time()
        plate, conf = enhance_recognize_plate(catcher, frame, vinfo['bbox'])
        dt = time.time() - t0
        if plate and conf > 0.5:
            vinfo['plate'] = plate
            vinfo['confidence'] = conf
            _cb(f"  首帧识别成功: {plate} ({conf:.2f}) [{dt:.1f}秒]", -1, stage="车牌识别")
            continue

        _cb(f"  首帧未识别 [{dt:.1f}秒]，多帧扫描中...", -1, stage="车牌识别")
        t0 = time.time()
        best_plate, best_frame, votes = multi_frame_recognize_with_frame(catcher, cap, vinfo['first_frame'], vinfo['bbox'], scan_range=30)
        dt = time.time() - t0
        if best_plate:
            vinfo['plate'] = best_plate
            vinfo['confidence'] = min(votes / 10, 1.0)
            vinfo['best_frame'] = best_frame
            _cb(f"  多帧扫描识别成功: {best_plate} (票数{votes}) [{dt:.1f}秒]", -1, stage="车牌识别")
        else:
            _cb(f"  多帧扫描仍未识别 [{dt:.1f}秒]", -1, stage="车牌识别")

    plate_elapsed = time.time() - t_plate_start
    _cb(f"车牌识别完成: {plate_elapsed:.1f}秒", 90, stage="车牌识别")

    cap.release()

    # ====== 阶段5: 生成证据和报告 (90%-100%) ======
    output_dir = os.path.join(os.path.dirname(video_path), f"{Path(video_path).stem}_检测结果")
    video_name = Path(video_path).stem
    evidence_dir = os.path.join(output_dir, "evidence")

    os.makedirs(output_dir, exist_ok=True)
    if os.path.isdir(evidence_dir):
        import shutil
        shutil.rmtree(evidence_dir)
    os.makedirs(evidence_dir, exist_ok=True)

    import shutil
    shutil.copy2(video_path, os.path.join(output_dir, os.path.basename(video_path)))

    _cb("正在生成证据文件...", 91, stage="生成报告")

    results = []
    for vid, vinfo in tracked_violations.items():
        timestamp = vinfo['time']
        mins = int(timestamp // 60)
        secs = int(timestamp % 60)
        results.append({
            'id': vid,
            'plate': vinfo['plate'] or '未识别',
            'time': f"{mins:02d}:{secs:02d}",
            'time_seconds': timestamp,
            'frame': vinfo['first_frame'],
            'best_frame': vinfo.get('best_frame', vinfo['first_frame']),
            'bbox': list(vinfo['bbox']),
            'confidence': vinfo['confidence'],
        })

    results = dedupe_violations(results, time_window=5.0)

    cap = cv2.VideoCapture(video_path)
    clip_results = []
    for index, result in enumerate(results, start=1):
        plate_key = result['plate_key']
        evidence_path = os.path.join(evidence_dir, plate_key)
        os.makedirs(evidence_path, exist_ok=True)

        _cb(f"  截图+视频: [{index}/{len(results)}] {plate_key}...",
            91 + int(6 * index / max(len(results), 1)),
            stage="生成报告")

        snap_frame = result.get('best_frame', result['frame'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, snap_frame)
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, result['frame'])
            ret, frame = cap.read()
        if ret:
            x1, y1, x2, y2 = result['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            label = f"{result['plate']} ({result['confidence']:.0%})"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            h, w = frame.shape[:2]
            cv2.rectangle(frame,
                         (int(w*lane_x), int(h*lane_top)),
                         (int(w*(lane_x+lane_width)), h),
                         (0, 255, 255), 2)
            snapshot_path = os.path.join(evidence_path, f'{plate_key}.jpg')
            try:
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                buf.tofile(snapshot_path)
            except Exception:
                cv2.imwrite(snapshot_path, frame)
            result['snapshot'] = snapshot_path

        clip_path = os.path.join(evidence_path, f'{plate_key}.mp4')
        clip = cut_video_clip(
            video_path,
            clip_path,
            result['time_seconds'],
            clip_duration=clip_duration,
            progress_callback=progress_callback,
        )
        if clip is not None:
            result['clip_path'] = clip['path']
            result['clip_label'] = plate_key
            result['clip_relative_time'] = round(result['time_seconds'] - clip['start'], 1)
            clip_results.append({'index': index, 'label': plate_key, **clip})

        meta_path = os.path.join(evidence_path, 'meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            import json
            json.dump({
                'plate': result['plate'],
                'plate_key': plate_key,
                'time_seconds': result['time_seconds'],
                'time': result['time'],
                'confidence': result['confidence'],
            }, f, ensure_ascii=False, indent=2)

    cap.release()

    _cb("正在生成HTML报告...", 98, stage="生成报告")
    write_manifest(results, output_dir, video_path)
    html_path = generate_html_report(results, video_path, output_dir, video_name,
                                      lane_x, lane_width, lane_top, clip_results)

    total_time = time.time() - start_time
    recognized = sum(1 for r in results if r['plate'] != '未识别')
    _cb(f"检测完成! 共{len(results)}组证据, {recognized}个车牌, {len(clip_results)}个视频, 耗时{total_time:.0f}秒",
        100, stage="完成")

    return {
        'violations': results,
        'html_path': html_path,
        'evidence_dir': evidence_dir,
        'clips': clip_results,
        'total_time': total_time,
        'total_violations': len(results),
        'recognized_plates': recognized,
        'model_info': {
            'model': model_name,
            'device': device,
            'gpu': gpu_info,
            'lpr': lpr_name,
            'video_info': f"{width}x{height}@{fps:.0f}fps",
            'duration': round(duration, 1),
        }
    }

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
        tk.Checkbutton(adv_frame, text="严格 GPU 模式", variable=self.gpu_var,
                       font=("Microsoft YaHei", 10), bg="white", fg="#555",
                       activebackground="white").pack(side=tk.LEFT)
        
        self.conf_var = tk.DoubleVar(value=0.5)
        tk.Label(adv_frame, text="  置信度阈值:", font=("Microsoft YaHei", 10),
                bg="white", fg="#555").pack(side=tk.LEFT, padx=(20, 5))
        tk.Spinbox(adv_frame, from_=0.1, to=1.0, increment=0.05,
                  textvariable=self.conf_var, width=5, font=("Consolas", 10),
                  bg="#fafafa", relief=tk.FLAT, bd=1).pack(side=tk.LEFT)
        
        self.scale_var = tk.DoubleVar(value=0.75)
        tk.Label(adv_frame, text="  检测缩放:", font=("Microsoft YaHei", 10),
                bg="white", fg="#555").pack(side=tk.LEFT, padx=(15, 5))
        tk.Spinbox(adv_frame, from_=0.25, to=1.0, increment=0.25,
                  textvariable=self.scale_var, width=5, font=("Consolas", 10),
                  bg="#fafafa", relief=tk.FLAT, bd=1).pack(side=tk.LEFT)

        self.clip_var = tk.IntVar(value=15)
        tk.Label(adv_frame, text="  证据视频时长:", font=("Microsoft YaHei", 10),
                bg="white", fg="#555").pack(side=tk.LEFT, padx=(15, 5))
        tk.Spinbox(adv_frame, from_=5, to=60, increment=5,
                  textvariable=self.clip_var, width=5, font=("Consolas", 10),
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
        self.bottom_label = tk.Label(bottom, text="YOLOv12s | 严格 GPU 模式 | 15s剪辑 + 车牌颜色 + 时间戳",
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
                clip_duration=self.clip_var.get(),
                progress_callback=self._progress_callback
            )
            self.result = result
            self.root.after(0, self._on_detection_done, result)
        except Exception as e:
            self.root.after(0, self._on_detection_error, str(e))
    
    def _progress_callback(self, msg, pct, **kwargs):
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
