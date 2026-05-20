#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型下载脚本 - 一键下载所有需要的模型到本地 models/ 目录

使用方法:
    python setup_models.py          # 下载所有模型
    python setup_models.py --yolo   # 只下载YOLO模型
    python setup_models.py --lpr    # 只下载HyperLPR3模型
    python setup_models.py --check  # 只检查模型状态
"""
import os
import sys
import argparse
import shutil
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
YOLO_DIR = MODELS_DIR / "yolo12"
LPR_DIR = MODELS_DIR / "hyperlpr3"

# YOLO模型 (从ultralytics自动下载)
YOLO_MODELS = {
    "yolo12s.pt": "yolo12s.pt",
}

# HyperLPR3 ONNX模型
LPR_MODELS = {
    "y5fu_640x_sim.onnx": "检测模型(640x)",
    "y5fu_320x_sim.onnx": "检测模型(320x,轻量)",
    "rpv3_mdict_160_r3.onnx": "识别模型",
    "litemodel_cls_96x_r1.onnx": "分类模型",
}


def check_yolo_model():
    """检查YOLO模型状态"""
    print("\n[1/2] YOLO 模型检查")
    print("-" * 40)

    # 检查项目本地
    local_path = MODELS_DIR / "yolo12s.pt"
    if local_path.exists():
        size_mb = local_path.stat().st_size / 1024 / 1024
        print(f"  [OK] 本地: {local_path} ({size_mb:.1f}MB)")
        return True

    # 检查ultralytics缓存
    cache_path = Path.home() / ".cache" / "ultralytics" / "yolo12s.pt"
    if cache_path.exists():
        size_mb = cache_path.stat().st_size / 1024 / 1024
        print(f"  [OK] 缓存: {cache_path} ({size_mb:.1f}MB)")
        return True

    # 检查项目根目录
    root_path = PROJECT_ROOT / "yolo12s.pt"
    if root_path.exists():
        size_mb = root_path.stat().st_size / 1024 / 1024
        print(f"  [OK] 项目根: {root_path} ({size_mb:.1f}MB)")
        return True

    print("  [!!] 未找到 yolo12s.pt")
    print("  首次运行时 ultralytics 会自动从网络下载")
    print("  或手动下载: https://github.com/ultralytics/assets/releases")
    return False


def check_lpr_models():
    """检查HyperLPR3模型状态"""
    print("\n[2/2] HyperLPR3 模型检查")
    print("-" * 40)

    # 多个可能的路径
    search_paths = [
        LPR_DIR,
        Path.home() / ".hyperlpr3" / "20230229" / "onnx",
    ]

    found = {}
    for sp in search_paths:
        if sp.exists():
            for model_file, desc in LPR_MODELS.items():
                fp = sp / model_file
                if fp.exists() and model_file not in found:
                    size_kb = fp.stat().st_size / 1024
                    found[model_file] = (fp, size_kb)

    if len(found) == len(LPR_MODELS):
        print(f"  [OK] 全部 {len(found)}/{len(LPR_MODELS)} 个模型就绪")
        for name, (path, size) in found.items():
            print(f"       {name} ({size:.0f}KB) @ {path.parent}")
        return True
    else:
        print(f"  [!!] 找到 {len(found)}/{len(LPR_MODELS)} 个模型")
        for model_file, desc in LPR_MODELS.items():
            if model_file in found:
                print(f"       [OK] {model_file} - {desc}")
            else:
                print(f"       [!!] {model_file} - {desc} (缺失)")

        print(f"\n  HyperLPR3模型通常随 pip install hyperlpr3 自动安装")
        print(f"  或手动放置到: {LPR_DIR}")
        return False


def copy_yolo_to_local():
    """将YOLO模型复制到本地models/目录"""
    cache_path = Path.home() / ".cache" / "ultralytics" / "yolo12s.pt"
    root_path = PROJECT_ROOT / "yolo12s.pt"
    local_path = MODELS_DIR / "yolo12s.pt"

    if local_path.exists():
        return True

    src = None
    if root_path.exists():
        src = root_path
    elif cache_path.exists():
        src = cache_path

    if src:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local_path)
        print(f"  [COPY] {src} -> {local_path}")
        return True
    return False


def copy_lpr_to_local():
    """将HyperLPR3模型复制到本地models/目录"""
    default_dir = Path.home() / ".hyperlpr3" / "20230229" / "onnx"

    if not default_dir.exists():
        return False

    LPR_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for model_file in LPR_MODELS:
        src = default_dir / model_file
        dst = LPR_DIR / model_file
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"  [COPY] {src.name}")
            copied += 1

    if copied > 0:
        print(f"  已复制 {copied} 个模型到 {LPR_DIR}")
    return copied > 0


def main():
    parser = argparse.ArgumentParser(description="模型下载/检查脚本")
    parser.add_argument("--yolo", action="store_true", help="只处理YOLO模型")
    parser.add_argument("--lpr", action="store_true", help="只处理HyperLPR3模型")
    parser.add_argument("--check", action="store_true", help="只检查状态，不复制")
    parser.add_argument("--copy", action="store_true", help="复制模型到本地models/目录")
    args = parser.parse_args()

    print("=" * 50)
    print("  高速公路应急车道违章检测 - 模型管理工具")
    print("=" * 50)

    if not args.yolo and not args.lpr:
        args.yolo = args.lpr = True  # 默认处理全部

    results = {}

    if args.yolo:
        results["YOLO"] = check_yolo_model()
        if args.copy and not results["YOLO"]:
            results["YOLO"] = copy_yolo_to_local()

    if args.lpr:
        results["LPR"] = check_lpr_models()
        if args.copy and not results["LPR"]:
            results["LPR"] = copy_lpr_to_local()

    # 总结
    print("\n" + "=" * 50)
    if all(results.values()):
        print("  所有模型就绪！可以运行程序了。")
        print("  启动: python emergency_lane/run_gpu_v8.py --video <视频路径>")
    else:
        print("  部分模型缺失，请按提示下载。")
        print("  详细说明见 README.md")
    print("=" * 50)

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
