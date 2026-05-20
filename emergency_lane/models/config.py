"""
统一模型路径配置
仅支持 YOLOv12 模型，优先使用项目本地 models/ 目录。
自动检测 CUDA > OpenVINO > CPU
"""
import os

from gpu_backend import detect_cuda, detect_openvino_gpu, resolve_yolo_device

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


_CUDA_AVAILABLE, _, _ = detect_cuda()
_OPENVINO_AVAILABLE = detect_openvino_gpu()


if _CUDA_AVAILABLE:
    _YOLO_CANDIDATES = [
        os.path.join(MODELS_DIR, "yolo12s.pt"),
        os.path.join(PROJECT_ROOT, "yolo12s.pt"),
        os.path.join(MODELS_DIR, "yolo12s_openvino_model"),
        os.path.join(PROJECT_ROOT, "yolo12s_openvino_model"),
    ]
else:
    _YOLO_CANDIDATES = [
        os.path.join(MODELS_DIR, "yolo12s_openvino_model"),
        os.path.join(PROJECT_ROOT, "yolo12s_openvino_model"),
        os.path.join(MODELS_DIR, "yolo12s.pt"),
        os.path.join(PROJECT_ROOT, "yolo12s.pt"),
    ]

YOLO_MODEL_PATH = None

for _p in _YOLO_CANDIDATES:
    if os.path.exists(_p):
        YOLO_MODEL_PATH = _p
        break

YOLO_DEVICE = resolve_yolo_device(YOLO_MODEL_PATH, cuda_available=_CUDA_AVAILABLE, openvino_available=_OPENVINO_AVAILABLE)


_HYPERLPR3_LOCAL_DIR = os.path.join(MODELS_DIR, "onnx")
_HYPERLPR3_DEFAULT_DIR = os.path.expanduser("~/.hyperlpr3/20230229/onnx")

def _find_model(filename):
    local = os.path.join(_HYPERLPR3_LOCAL_DIR, filename)
    if os.path.exists(local):
        return local
    default = os.path.join(_HYPERLPR3_DEFAULT_DIR, filename)
    if os.path.exists(default):
        return default
    return local

HYPERLPR3_ONNX_DIR = _HYPERLPR3_LOCAL_DIR if os.path.isdir(_HYPERLPR3_LOCAL_DIR) else _HYPERLPR3_DEFAULT_DIR

HYPERLPR3_DET_MODEL = _find_model("y5fu_640x_sim.onnx")
HYPERLPR3_DET_MODEL_LOW = _find_model("y5fu_320x_sim.onnx")
HYPERLPR3_REC_MODEL = _find_model("rpv3_mdict_160_r3.onnx")
HYPERLPR3_CLS_MODEL = _find_model("litemodel_cls_96x_r1.onnx")
