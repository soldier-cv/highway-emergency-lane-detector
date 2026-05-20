"""
高速公路应急车道违章检测系统 v8.1
====================================
流程：检测违章 → 剪裁15s片段 → 在剪辑上识别车牌+颜色+时间戳 → 生成报告

功能:
1. YOLOv12s 目标检测
2. 15s违章视频片段剪辑（保留违章前3s）
3. 剪辑片段保存在原视频所在文件夹的子目录
4. 报告中违章时间按片段时间标记（片段内第X秒）
5. 视频右上角时间戳OCR提取
6. 车牌颜色精准识别（蓝/黄/绿/白/黑5种）
7. OpenVINO / CUDA GPU全链路加速
"""
import cv2
import numpy as np
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import compute_overlap, compute_iou, format_tc
from models.config import YOLO_MODEL_PATH
from gpu_backend import get_gpu_backend, resolve_yolo_device
from evidence_utils import cut_video_clip
from report_utils import generate_html_report, dedupe_violations, write_manifest

# =============================================
# 配置
# =============================================
# 支持命令行传入视频路径
if len(sys.argv) > 1:
    VIDEO_PATH = sys.argv[1]
else:
    VIDEO_PATH = input("请输入视频文件路径: ").strip().strip('"')
VIDEO_DIR = os.path.dirname(VIDEO_PATH)
VIDEO_STEM = Path(VIDEO_PATH).stem

# 输出目录：原视频所在文件夹的子目录
OUTPUT_DIR = os.path.join(VIDEO_DIR, f"{VIDEO_STEM}_检测结果")
EVIDENCE_DIR = os.path.join(OUTPUT_DIR, "evidence")
REPORT_HTML = os.path.join(OUTPUT_DIR, f"{VIDEO_STEM}_violation_report.html")

LANE_START_X = 0.84
MIN_OVERLAP = 0.25
CONF_THRESH = 0.25
DET_SCALE = 0.5
CONFIRM_FRAMES = 3
COOLDOWN_FRAMES = 150
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# 中国车牌颜色分类
PLATE_COLORS = {
    0: {"name": "蓝牌", "desc": "小型汽车号牌（蓝底白字）", "bg": "#0052CC"},
    1: {"name": "黄牌", "desc": "大型汽车/挂车/教练车（黄底黑字）", "bg": "#F5A623"},
    2: {"name": "白牌", "desc": "军车/警车/武警（白底黑/红字）", "bg": "#CCCCCC"},
    3: {"name": "绿牌", "desc": "新能源车辆（绿底黑字）", "bg": "#00B140"},
    4: {"name": "黑牌", "desc": "港澳/外资企业（黑底白字）", "bg": "#333333"},
}

print("=" * 60)
print("  应急车道违章检测 v8.1")
print("  YOLOv12s | GPU加速 | 15s剪辑 | 时间戳 | 车牌颜色")
print("=" * 60)


def recognize_plate_color(plate_img, plate_text=""):
    """识别车牌颜色（纯HSV色彩空间分析，不依赖车牌号码）
    Returns: (color_id, color_name, confidence)
    """
    if plate_img is None or plate_img.size == 0:
        return -1, "未知", 0.0
    
    hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
    h, w = plate_img.shape[:2]
    
    # 采样策略：取车牌中间区域，排除字符（白色高亮度像素）
    mx = int(w * 0.1)
    # 上下两条窄带（避开中间大字符区域）
    top_band = hsv[0:h//5, mx:w-mx]
    bot_band = hsv[h*4//5:h, mx:w-mx]
    # 也取中间区域但过滤掉白色（字符）
    mid_region = hsv[h//5:h*4//5, mx:w-mx]
    
    # 合并上下带
    sample_regions = []
    if top_band.size > 0:
        sample_regions.append(top_band)
    if bot_band.size > 0:
        sample_regions.append(bot_band)
    
    # 中间区域去掉白色像素（字符）
    if mid_region.size > 0:
        non_white_mask = ~((mid_region[:,:,1] < 50) & (mid_region[:,:,2] > 180))
        mid_no_white = mid_region[non_white_mask]
        if mid_no_white.size > 0:
            mid_no_white = mid_no_white.reshape(-1, 3)
            sample_regions.append(mid_no_white)
    
    if not sample_regions:
        return -1, "未知", 0.0
    
    # 合并所有采样
    samples = np.vstack([s.reshape(-1, 3) for s in sample_regions])
    total = samples.shape[0]
    if total == 0:
        return -1, "未知", 0.0
    
    # HSV颜色范围
    # 蓝牌底色: H 100-130, S 80-255, V 60-255
    blue_mask = ((samples[:,0] >= 100) & (samples[:,0] <= 130) & 
                 (samples[:,1] >= 80) & (samples[:,2] >= 60))
    blue_r = np.sum(blue_mask) / total
    
    # 黄牌底色: H 15-35, S 80-255, V 80-255
    yellow_mask = ((samples[:,0] >= 15) & (samples[:,0] <= 35) & 
                   (samples[:,1] >= 80) & (samples[:,2] >= 80))
    yellow_r = np.sum(yellow_mask) / total
    
    # 绿牌底色: H 35-85, S 40-255, V 40-255
    green_mask = ((samples[:,0] >= 35) & (samples[:,0] <= 85) & 
                  (samples[:,1] >= 40) & (samples[:,2] >= 40))
    green_r = np.sum(green_mask) / total
    
    # 白牌底色: S 0-40, V 200-255
    white_mask = ((samples[:,1] <= 40) & (samples[:,2] >= 200))
    white_r = np.sum(white_mask) / total
    
    # 黑牌底色: V 0-50（极暗）
    black_mask = (samples[:,2] <= 50)
    black_r = np.sum(black_mask) / total
    
    scores = {0: ("蓝牌", blue_r), 1: ("黄牌", yellow_r), 3: ("绿牌", green_r),
              2: ("白牌", white_r), 4: ("黑牌", black_r)}
    
    best_id = max(scores, key=lambda k: scores[k][1])
    best_name, best_ratio = scores[best_id]
    
    if best_ratio < 0.03:
        # 默认蓝牌（最常见）
        return 0, "蓝牌", 0.3
    
    second = sorted(scores.values(), key=lambda x: x[1], reverse=True)[1][1]
    confidence = min(best_ratio / (best_ratio + second + 1e-6), 1.0)
    
    return best_id, best_name, float(confidence)


def extract_timestamp_ocr(frame):
    """提取视频右上角时间戳（OCR）
    行车记录仪时间戳通常在右上角，白色文字
    """
    h, w = frame.shape[:2]
    # 裁剪右上角区域
    tw, th = int(w * 0.35), int(h * 0.08)
    ts_region = frame[5:5+th, w-tw-5:w-5]
    
    # 预处理：灰度+二值化
    gray = cv2.cvtColor(ts_region, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    try:
        import pytesseract
        text = pytesseract.image_to_string(
            binary,
            config='--psm 7 -c tessedit_char_whitelist=0123456789:/-._ AMPP '
        )
        text = text.strip()
        if text and len(text) > 5:
            return text
    except Exception:
        pass
    
    return None


def extract_timestamp_from_video(frame, frame_idx, fps):
    """综合提取时间戳：优先OCR，否则用视频帧时间"""
    ocr_result = extract_timestamp_ocr(frame)
    video_time = format_tc(frame_idx / fps)
    return {
        "dvr_time": ocr_result,
        "video_time": video_time,
    }


# =============================================
# 1. 加载模型
# =============================================
print("\n[1/5] 加载模型...")

if YOLO_MODEL_PATH is None:
    print("  ❌ 未找到模型文件！请确保项目根目录下有 yolo12s.pt 或 yolo12s_openvino_model/")
    print("     或运行: python setup_models.py")
    sys.exit(1)
gpu = get_gpu_backend()
yolo = YOLO(YOLO_MODEL_PATH, task="detect")
_yolo_device = resolve_yolo_device(YOLO_MODEL_PATH, cuda_available=gpu.cuda_available, openvino_available=gpu.openvino_available)
dummy = np.zeros((848, 1920, 3), dtype=np.uint8)
for _ in range(5):
    yolo.predict(dummy, conf=0.25, device=_yolo_device, classes=[2,3,5,7], verbose=False)
print(f"  YOLOv12s {_yolo_device} ready")

if gpu.cuda_available:
    from lpr3_ort import LicensePlateCatcherORT
    plate_catcher = LicensePlateCatcherORT(det_level=1)
    print("  HyperLPR3 ONNXRuntime CUDA ready")
else:
    from lpr3_openvino import LicensePlateCatcherOV
    plate_catcher = LicensePlateCatcherOV(device="GPU", det_level=1)
    print("  HyperLPR3 OpenVINO GPU ready")

# =============================================
# 2. 检测违章车辆（只检测，不识别）
# =============================================
print("\n[2/5] 检测违章车辆...")
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_duration = total_frames / fps

det_w = int(width * DET_SCALE)
det_h = int(height * DET_SCALE)
sx = width / det_w
sy = height / det_h
lane_x1 = int(width * LANE_START_X)
lane_y1 = int(height * 0.15)
lane_region = (lane_x1, lane_y1, width, height)

print(f"  {width}x{height} @ {fps:.0f}fps, {total_frames}帧, {video_duration:.1f}s")

tracks = []
next_tid = 0
violations_raw = []
confirmed = set()
last_rec = {}
fc = 0
t_start = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    fc += 1

    small = cv2.resize(frame, (det_w, det_h))
    results = yolo.predict(small, conf=CONF_THRESH, device=_yolo_device,
                           classes=list(VEHICLE_CLASSES.keys()), verbose=False)

    dets = []
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cid = int(box.cls[0])
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
            dets.append((
                (int(bx1*sx), int(by1*sy), int(bx2*sx), int(by2*sy)),
                cid, float(box.conf[0])
            ))

    matched = set()
    for bbox, cid, conf in dets:
        best_iou, best_idx = 0.3, -1
        for ti, trk in enumerate(tracks):
            if ti in matched or fc - trk['lf'] > 30:
                continue
            iou = compute_iou(bbox, trk['bbox'])
            if iou > best_iou:
                best_iou, best_idx = iou, ti

        in_lane = compute_overlap(bbox, lane_region) >= MIN_OVERLAP

        if best_idx >= 0:
            trk = tracks[best_idx]
            trk['bbox'] = bbox
            trk['lf'] = fc
            trk['cid'] = cid
            trk['conf'] = conf
            trk['ilc'] = trk['ilc'] + 1 if in_lane else max(0, trk['ilc'] - 1)
            matched.add(best_idx)
        else:
            tracks.append({
                'tid': next_tid, 'bbox': bbox, 'cid': cid, 'conf': conf,
                'ilc': 1 if in_lane else 0, 'lf': fc, 'ff': fc,
            })
            next_tid += 1

    tracks = [t for t in tracks if fc - t['lf'] <= 60]

    for trk in tracks:
        if trk['ilc'] < CONFIRM_FRAMES:
            continue
        tid = trk['tid']
        if tid in confirmed:
            if fc - last_rec.get(f"t_{tid}", -9999) < COOLDOWN_FRAMES:
                continue

        violations_raw.append({
            'track_id': tid,
            'bbox': list(trk['bbox']),
            'frame': fc,
            'timestamp_seconds': round(fc / fps, 2),
        })
        last_rec[f"t_{tid}"] = fc
        confirmed.add(tid)

    if fc % 600 == 0:
        elapsed = time.time() - t_start
        speed = fc / elapsed
        eta = (total_frames - fc) / speed / 60
        print(f"  帧 {fc}/{total_frames} ({fc/total_frames*100:.0f}%) "
              f"速度:{speed:.1f}fps ETA:{eta:.1f}min 违章:{len(violations_raw)}")

cap.release()
elapsed_det = time.time() - t_start

print(f"\n检测完成! 耗时: {elapsed_det:.1f}s ({elapsed_det/60:.1f}min)")
print(f"检测到 {len(violations_raw)} 起违章")

# =============================================
# 3. 15s违章视频片段剪辑
# =============================================
print(f"\n[3/5] 生成15s违章视频片段...")

os.makedirs(OUTPUT_DIR, exist_ok=True)
if os.path.isdir(EVIDENCE_DIR):
    shutil.rmtree(EVIDENCE_DIR)
os.makedirs(EVIDENCE_DIR, exist_ok=True)
shutil.copy2(VIDEO_PATH, os.path.join(OUTPUT_DIR, os.path.basename(VIDEO_PATH)))

# =============================================
# 4. 在剪辑片段上识别车牌+颜色+时间戳
# =============================================
print(f"\n[4/5] 识别车牌+颜色+时间戳 ({len(violations_raw)} 起)...")

cap = cv2.VideoCapture(VIDEO_PATH)

def recognize_plate(frame, bbox):
    """多策略车牌识别，返回 (plate_text, confidence, plate_type, plate_bbox_in_frame)"""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    results = []
    px = int((x2 - x1) * 0.1)
    py = int((y2 - y1) * 0.15)
    cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
    cx2, cy2 = min(w, x2 + px), min(h, y2 + py)

    crops_info = [
        (frame[cy1:cy2, cx1:cx2], (cx1, cy1)),                               # 全车
        (frame[(y1+y2)//2:cy2, cx1:cx2], (cx1, (y1+y2)//2)),                  # 下半
        (frame[y1+(y2-y1)*2//3:cy2, cx1:cx2], (cx1, y1+(y2-y1)*2//3)),       # 下1/3
    ]

    for crop, (offset_x, offset_y) in crops_info:
        if crop.size == 0:
            continue
        ch, cw = crop.shape[:2]
        
        # 原图
        try:
            for r in plate_catcher(crop):
                p_no, conf, p_type, p_bbox = r
                if len(p_no) >= 7 and conf > 0.25:
                    # 转换车牌bbox到原始帧坐标
                    pb = p_bbox if p_bbox is not None else [0,0,cw,ch]
                    abs_bbox = [pb[0]+offset_x, pb[1]+offset_y, pb[2]+offset_x, pb[3]+offset_y]
                    results.append((p_no, float(conf), int(p_type), abs_bbox))
        except:
            pass
        
        # CLAHE增强
        try:
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            l_ch = cv2.createCLAHE(3.0, (8,8)).apply(cv2.split(lab)[0])
            enh = cv2.cvtColor(cv2.merge([l_ch, cv2.split(lab)[1], cv2.split(lab)[2]]), cv2.COLOR_LAB2BGR)
            for r in plate_catcher(enh):
                p_no, conf, p_type, p_bbox = r
                if len(p_no) >= 7 and conf > 0.25:
                    pb = p_bbox if p_bbox is not None else [0,0,cw,ch]
                    abs_bbox = [pb[0]+offset_x, pb[1]+offset_y, pb[2]+offset_x, pb[3]+offset_y]
                    results.append((p_no, float(conf), int(p_type), abs_bbox))
        except:
            pass
        
        # 2x放大
        if min(ch, cw) < 400:
            big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            try:
                for r in plate_catcher(big):
                    p_no, conf, p_type, p_bbox = r
                    if len(p_no) >= 7 and conf > 0.25:
                        pb = p_bbox if p_bbox is not None else [0,0,cw*2,ch*2]
                        # 缩放回原始crop坐标
                        abs_bbox = [pb[0]//2+offset_x, pb[1]//2+offset_y, pb[2]//2+offset_x, pb[3]//2+offset_y]
                        results.append((p_no, float(conf), int(p_type), abs_bbox))
            except:
                pass

    if not results:
        return None
    # 投票选最佳
    stats = defaultdict(lambda: {"count": 0, "max_conf": 0, "best_bbox": None})
    for p_no, conf, pt, pb in results:
        stats[p_no]["count"] += 1
        if conf > stats[p_no]["max_conf"]:
            stats[p_no]["max_conf"] = conf
            stats[p_no]["best_bbox"] = pb
        stats[p_no]["type"] = pt
    best, bs = None, 0
    for p_no, s in stats.items():
        score = s["count"] * 0.3 + s["max_conf"] * 0.7
        if score > bs:
            bs = score
            best = (p_no, s["max_conf"], s["type"], s["best_bbox"])
    return best

t_p2 = time.time()
violations = []

for i, vr in enumerate(violations_raw):
    fi = vr["frame"]
    bbox = tuple(vr["bbox"])
    print(f"  [{i+1}/{len(violations_raw)}] 帧{fi}...", end="", flush=True)
    
    # 多帧扫描识别车牌，记录最佳识别帧和车牌位置
    best_plate, best_conf = None, 0
    best_frame_no = fi  # 默认用违章帧
    best_plate_bbox_in_frame = None  # 车牌在原始帧上的精确位置
    for offset in range(-30, 31, 2):
        target = fi + offset
        if target < 1 or target > total_frames:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, f = cap.read()
        if not ret:
            continue
        h, w = f.shape[:2]
        px = int((bbox[2]-bbox[0])*0.1)
        py = int((bbox[3]-bbox[1])*0.15)
        cx1, cy1 = max(0, bbox[0]-px), max(0, bbox[1]-py)
        cx2, cy2 = min(w, bbox[2]+px), min(h, bbox[3]+py)
        crop = f[cy1:cy2, cx1:cx2]
        
        if crop.size == 0:
            continue
        
        # 直接在crop上运行车牌检测
        try:
            for r in plate_catcher(crop):
                p_no, conf, p_type, p_bbox = r
                if len(p_no) >= 7 and conf > best_conf and conf > 0.25:
                    best_plate = (p_no, float(conf), int(p_type))
                    best_conf = float(conf)
                    best_frame_no = target
                    # 记录车牌在原始帧上的位置
                    if p_bbox is not None:
                        best_plate_bbox_in_frame = [p_bbox[0]+cx1, p_bbox[1]+cy1, 
                                                     p_bbox[2]+cx1, p_bbox[3]+cy1]
                    else:
                        best_plate_bbox_in_frame = None
        except:
            pass
        
        # CLAHE增强
        try:
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            l_ch = cv2.createCLAHE(3.0, (8,8)).apply(cv2.split(lab)[0])
            enh = cv2.cvtColor(cv2.merge([l_ch, cv2.split(lab)[1], cv2.split(lab)[2]]), cv2.COLOR_LAB2BGR)
            for r in plate_catcher(enh):
                p_no, conf, p_type, p_bbox = r
                if len(p_no) >= 7 and conf > best_conf and conf > 0.25:
                    best_plate = (p_no, float(conf), int(p_type))
                    best_conf = float(conf)
                    best_frame_no = target
                    if p_bbox is not None:
                        best_plate_bbox_in_frame = [p_bbox[0]+cx1, p_bbox[1]+cy1, 
                                                     p_bbox[2]+cx1, p_bbox[3]+cy1]
                    else:
                        best_plate_bbox_in_frame = None
        except:
            pass
    
    # 获取违章帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame_at = cap.read()
    
    # 提取时间戳
    ts_info = {"dvr_time": None, "video_time": format_tc(fi / fps)}
    if ret:
        ts_info = extract_timestamp_from_video(frame_at, fi, fps)
    
    # 识别车牌颜色（用最佳识别帧上的精确车牌位置）
    plate_color_id, plate_color_name, plate_color_conf = 0, "蓝牌", 0.0
    if best_plate:
        # 用最佳识别帧做颜色识别（车牌位置最精确）
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_no)
        ret_color, frame_color = cap.read()
        if ret_color:
            h, w = frame_color.shape[:2]
            if best_plate_bbox_in_frame is not None:
                px1, py1, px2, py2 = best_plate_bbox_in_frame
                pad_x = max(5, int((px2-px1)*0.08))
                pad_y = max(3, int((py2-py1)*0.15))
                plate_crop = frame_color[max(0,py1-pad_y):min(h,py2+pad_y), 
                                         max(0,px1-pad_x):min(w,px2+pad_x)]
            else:
                # 回退：用车辆区域的下半部分
                vx1, vy1, vx2, vy2 = bbox
                plate_crop = frame_color[max(0,(vy1+vy2)//2):min(h,vy2), 
                                         max(0,vx1):min(w,vx2)]
            
            if plate_crop.size > 0:
                plate_text_for_color = best_plate[0] if best_plate else ""
                plate_color_id, plate_color_name, plate_color_conf = recognize_plate_color(plate_crop, plate_text_for_color)
    
    # 填充结果
    if best_plate:
        vr["plate_number"] = best_plate[0]
        vr["plate_confidence"] = best_plate[1]
    else:
        vr["plate_number"] = "未识别"
        vr["plate_confidence"] = 0.0
    
    vr["plate_color_id"] = plate_color_id
    vr["plate_color_name"] = plate_color_name
    vr["plate_color_confidence"] = plate_color_conf
    vr["dvr_time"] = ts_info.get("dvr_time")
    vr["video_time"] = ts_info["video_time"]
    
    # 片段内相对时间
    clip_rel = vr.get("clip_relative_time")
    clip_label = vr.get("clip_label", "")
    clip_time_str = f"片段内第{clip_rel:.1f}秒" if clip_rel is not None else "未归属片段"
    
    print(f" {vr['plate_number']} ({vr['plate_color_name']}) {vr['plate_confidence']:.2f} "
          f"视频:{vr['video_time']} DVR:{vr['dvr_time']} {clip_time_str}")
    
    violations.append(vr)

cap.release()
violations = dedupe_violations(violations, time_window=5.0)
for violation in violations:
    violation.pop("snapshot", None)
    violation.pop("clip_path", None)

cap = cv2.VideoCapture(VIDEO_PATH)
clip_results = []
for index, violation in enumerate(violations, start=1):
    plate_key = violation["plate_key"]
    evidence_path = os.path.join(EVIDENCE_DIR, plate_key)
    os.makedirs(evidence_path, exist_ok=True)

    snap_frame = violation.get("best_frame", violation["frame"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, snap_frame)
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, violation["frame"])
        ret, frame = cap.read()
    if ret:
        x1, y1, x2, y2 = map(int, violation["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = f'{violation["plate_number"]} ({violation["plate_confidence"]:.0%})'
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        snapshot_path = os.path.join(evidence_path, f'{plate_key}.jpg')
        try:
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            buf.tofile(snapshot_path)
        except Exception:
            cv2.imwrite(snapshot_path, frame)
        violation["snapshot"] = snapshot_path

    clip_path = os.path.join(evidence_path, f'{plate_key}.mp4')
    clip = cut_video_clip(VIDEO_PATH, clip_path, violation["timestamp_seconds"])
    if clip is not None:
        violation["clip_path"] = clip["path"]
        violation["clip_label"] = plate_key
        violation["clip_relative_time"] = round(violation["timestamp_seconds"] - clip["start"], 1)
        clip_results.append({"index": index, "label": plate_key, **clip})

    meta_path = os.path.join(evidence_path, 'meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        import json
        json.dump({
            'plate': violation['plate_number'],
            'plate_key': plate_key,
            'time_seconds': violation['timestamp_seconds'],
            'video_time': violation['video_time'],
            'dvr_time': violation.get('dvr_time'),
            'confidence': violation['plate_confidence'],
            'plate_color_name': violation.get('plate_color_name'),
        }, f, ensure_ascii=False, indent=2)

cap.release()
elapsed_p2 = time.time() - t_p2

# =============================================
# 5. 生成报告
# =============================================
print(f"\n[5/5] 生成报告...")

total_time = time.time() - t_start

pv = defaultdict(list)
for v in violations:
    pv[v["plate_number"]].append(v)

plates_found = len([p for p in pv if p != "未识别"])
write_manifest(violations, OUTPUT_DIR, VIDEO_PATH)
html_path = generate_html_report(
    violations,
    VIDEO_PATH,
    OUTPUT_DIR,
    VIDEO_STEM,
    LANE_START_X,
    1.0 - LANE_START_X,
    0.15,
    clips=clip_results,
    embed_snapshots=True,
)
REPORT_HTML = html_path

print("\n" + "=" * 60)
print(f"  检测: {elapsed_det:.0f}s | 识别: {elapsed_p2:.0f}s | 总: {total_time:.0f}s ({total_time/60:.1f}min)")
print(f"  违章: {len(violations)} | 车牌: {plates_found} | 视频: {len(clip_results)}个")
print(f"\n  输出目录: {OUTPUT_DIR}")
for p in sorted(p for p in pv if p != "未识别"):
    vs = pv[p]
    bc = max(v["plate_confidence"] for v in vs)
    pc = vs[0].get("plate_color_name", "?")
    dvr = vs[0].get("dvr_time", "")
    vt = vs[0]["video_time"]
    print(f"  {p} ({pc}) 视频:{vt} DVR:{dvr} 置信度:{bc:.2f}")
print("=" * 60)
