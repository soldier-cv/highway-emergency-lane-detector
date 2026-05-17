"""
统一模型路径配置
优先使用项目本地 models/ 目录，回退到原有路径
自动检测 CUDA > OpenVINO > CPU
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def _detect_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


_CUDA_AVAILABLE = _detect_cuda()


if _CUDA_AVAILABLE:
    _YOLO_CANDIDATES = [
        os.path.join(MODELS_DIR, "yolov8s.pt"),
        os.path.join(PROJECT_ROOT, "yolov8s.pt"),
        os.path.join(MODELS_DIR, "yolov8s_openvino_model"),
        os.path.join(PROJECT_ROOT, "yolov8s_openvino_model"),
        os.path.expanduser("~/.cache/modelscope/models/AI-ModelScope/YOLOv8/yolov8s.pt"),
        os.path.expanduser("~/.cache/modelscope/models/AI-ModelScope/YOLOv8/yolov8s_openvino_model"),
    ]
else:
    _YOLO_CANDIDATES = [
        os.path.join(MODELS_DIR, "yolov8s_openvino_model"),
        os.path.join(PROJECT_ROOT, "yolov8s_openvino_model"),
        os.path.join(MODELS_DIR, "yolov8s.pt"),
        os.path.join(PROJECT_ROOT, "yolov8s.pt"),
        os.path.expanduser("~/.cache/modelscope/models/AI-ModelScope/YOLOv8/yolov8s_openvino_model"),
        os.path.expanduser("~/.cache/modelscope/models/AI-ModelScope/YOLOv8/yolov8s.pt"),
    ]

YOLO_MODEL_PATH = None
YOLO_DEVICE = "cuda:0" if _CUDA_AVAILABLE else "cpu"

for _p in _YOLO_CANDIDATES:
    if os.path.exists(_p):
        YOLO_MODEL_PATH = _p
        if _CUDA_AVAILABLE:
            YOLO_DEVICE = "cuda:0"
        elif "openvino" in _p:
            YOLO_DEVICE = "intel:GPU"
        else:
            YOLO_DEVICE = "cpu"
        break


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
