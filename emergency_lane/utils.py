"""共享工具函数：重叠计算、IoU、时间码格式化、JSON序列化修复"""

import numpy as np


def compute_overlap(bbox, region):
    """计算bbox与区域的重叠面积占bbox面积的比例"""
    bx1, by1, bx2, by2 = bbox
    rx1, ry1, rx2, ry2 = region
    ix1, iy1 = max(bx1, rx1), max(by1, ry1)
    ix2, iy2 = min(bx2, rx2), min(by2, ry2)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1) / max((bx2 - bx1) * (by2 - by1), 1)


def compute_iou(b1, b2):
    """计算两个框的IoU (Intersection over Union)"""
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if x1 >= x2 or y1 >= y2:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area_b = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / max(area_a + area_b - inter, 1)


def format_tc(seconds):
    """将秒数格式化为 HH:MM:SS 时间码"""
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def fix_json_types(obj):
    """递归修复numpy类型，使其可被JSON序列化"""
    if isinstance(obj, dict):
        return {k: fix_json_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [fix_json_types(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
