"""
统一GPU后端管理模块
自动检测可用GPU加速后端：CUDA > OpenVINO > CPU
为YOLO推理和ONNX推理提供统一的设备选择接口
"""

import os
import logging

logger = logging.getLogger(__name__)


def detect_cuda():
    """检测NVIDIA CUDA是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            logger.info(f"CUDA可用: {gpu_name} ({gpu_mem:.1f}GB)")
            return True, gpu_name, gpu_mem
    except ImportError:
        pass
    return False, None, 0


def detect_openvino_gpu():
    """检测OpenVINO GPU是否可用"""
    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        if "GPU" in devices:
            logger.info(f"OpenVINO GPU可用: {devices}")
            return True
    except ImportError:
        pass
    return False


def detect_onnxruntime_providers():
    """检测onnxruntime可用的执行提供者"""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        logger.info(f"ONNXRuntime可用providers: {providers}")
        return providers
    except ImportError:
        return []


class GPUBackend:
    """统一GPU后端管理器"""

    def __init__(self, force_device=None):
        self.cuda_available = False
        self.openvino_available = False
        self.gpu_name = None
        self.gpu_memory_gb = 0
        self.onnx_providers = []

        self.cuda_available, self.gpu_name, self.gpu_memory_gb = detect_cuda()
        self.openvino_available = detect_openvino_gpu()
        self.onnx_providers = detect_onnxruntime_providers()

        if force_device:
            self.yolo_device = self._resolve_force_device(force_device)
        elif self.cuda_available:
            self.yolo_device = "cuda:0"
        elif self.openvino_available:
            self.yolo_device = "intel:GPU"
        else:
            self.yolo_device = "cpu"

        if force_device == "cuda" or (not force_device and "CUDAExecutionProvider" in self.onnx_providers):
            self.onnx_provider = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif force_device == "openvino" or (not force_device and self.openvino_available):
            self.onnx_provider = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
        else:
            self.onnx_provider = ["CPUExecutionProvider"]

        if "CUDAExecutionProvider" in self.onnx_provider:
            self.lpr_device = "cuda"
        elif self.openvino_available:
            self.lpr_device = "GPU"
        else:
            self.lpr_device = "CPU"

        self._print_status()

    def _resolve_force_device(self, device):
        device = device.lower()
        if device in ("cuda", "cuda:0", "gpu"):
            if self.cuda_available:
                return "cuda:0"
            logger.warning("CUDA不可用，回退到CPU")
            return "cpu"
        elif device in ("openvino", "intel:gpu"):
            if self.openvino_available:
                return "intel:GPU"
            logger.warning("OpenVINO GPU不可用，回退到CPU")
            return "cpu"
        elif device == "cpu":
            return "cpu"
        return "cpu"

    def _print_status(self):
        print(f"  GPU后端状态:")
        print(f"    CUDA: {'可用' if self.cuda_available else '不可用'}"
              + (f" ({self.gpu_name}, {self.gpu_memory_gb:.1f}GB)" if self.cuda_available else ""))
        print(f"    OpenVINO GPU: {'可用' if self.openvino_available else '不可用'}")
        print(f"    YOLO设备: {self.yolo_device}")
        print(f"    ONNX提供者: {self.onnx_provider}")
        print(f"    车牌识别设备: {self.lpr_device}")

    @property
    def is_gpu_accelerated(self):
        return self.cuda_available or self.openvino_available

    @property
    def backend_name(self):
        if "cuda" in str(self.yolo_device).lower():
            return "CUDA"
        elif "intel" in str(self.yolo_device).lower() or "openvino" in str(self.yolo_device).lower():
            return "OpenVINO"
        return "CPU"

    def get_yolo_model_path_candidates(self, project_root):
        models_dir = os.path.join(project_root, "models")
        if self.cuda_available:
            return [
                os.path.join(models_dir, "yolov8s.pt"),
                os.path.join(project_root, "yolov8s.pt"),
                os.path.join(models_dir, "yolov8s_openvino_model"),
                os.path.join(project_root, "yolov8s_openvino_model"),
            ]
        elif self.openvino_available:
            return [
                os.path.join(models_dir, "yolov8s_openvino_model"),
                os.path.join(project_root, "yolov8s_openvino_model"),
                os.path.join(models_dir, "yolov8s.pt"),
                os.path.join(project_root, "yolov8s.pt"),
            ]
        else:
            return [
                os.path.join(project_root, "yolov8s.pt"),
                os.path.join(models_dir, "yolov8s.pt"),
                os.path.join(models_dir, "yolov8s_openvino_model"),
                os.path.join(project_root, "yolov8s_openvino_model"),
            ]


_backend_instance = None


def get_gpu_backend(force_device=None):
    global _backend_instance
    if _backend_instance is None:
        _backend_instance = GPUBackend(force_device=force_device)
    return _backend_instance
