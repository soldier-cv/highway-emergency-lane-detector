"""
HyperLPR3 车牌识别 - ONNXRuntime CUDA 加速版
直接用 OpenVINO API 加载 HyperLPR3 的 ONNX 模型
用正确的字符集和后处理逻辑
"""

import os
import cv2
import numpy as np


from models.config import (
    HYPERLPR3_DET_MODEL,
    HYPERLPR3_DET_MODEL_LOW,
    HYPERLPR3_REC_MODEL,
    HYPERLPR3_CLS_MODEL,

)

# HyperLPR3 的完整字符集（77个，与源码一致）
TOKEN = [
    'blank', "'", '0', '1', '2', '3', '4', '5', '6', '7',
    '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
    'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '云', '京', '冀',
    '吉', '学', '宁', '川', '挂', '新', '晋', '桂', '民', '沪',
    '津', '浙', '渝', '港', '湘', '琼', '甘', '皖', '粤', '航',
    '苏', '蒙', '藏', '警', '豫', '贵', '赣', '辽', '鄂', '闽',
    '陕', '青', '鲁', '黑', '领', '使', '澳',
]

# Ignored tokens（blank=0 和 '=1 不参与解码）
IGNORED_TOKENS = {0, 1}


def encode_images(image, wh_ratio, input_size):
    """与HyperLPR3一致的预处理"""
    h, w = input_size
    if wh_ratio > (w / h):
        resized = cv2.resize(image, (w, int(w / wh_ratio)))
    else:
        resized = cv2.resize(image, (int(h * wh_ratio), h))
    
    # Pad to input size
    rh, rw = resized.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    canvas[:rh, :rw] = resized.astype(np.float32)
    canvas = canvas / 255.0
    canvas = canvas.transpose(2, 0, 1)  # (3, H, W)
    return canvas



def _get_providers(allow_cpu_fallback=True):
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available:
            if allow_cpu_fallback:
                return ['CUDAExecutionProvider', 'CPUExecutionProvider']
            return ['CUDAExecutionProvider']
        return ['CPUExecutionProvider']
    except ImportError:
        return ['CPUExecutionProvider']


class LPR3DetectorORT:
    """车牌检测器 - ONNXRuntime CUDA 加速 (YOLOv5格式)"""
    
    # 车牌类型：10类
    PLATE_CLASSES = [
        "蓝牌单层", "黄牌单层", "白牌单层", "绿牌新能源", "黑牌港澳",
        "香港单层", "香港双层", "澳门单层", "澳门双层", "黄牌双层"
    ]
    
    def __init__(self, model_path=None, providers=None):
        if model_path is None:
            model_path = HYPERLPR3_DET_MODEL
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"检测模型不存在: {model_path}")
        
        import onnxruntime as ort
        if providers is None:
            providers = _get_providers()
        self.session = ort.InferenceSession(model_path, providers=providers)
        input_shape = self.session.get_inputs()[0].shape
        self.input_size = (input_shape[2], input_shape[3])
        self.input_name = self.session.get_inputs()[0].name
        print(f"  LPR3Detector ORT: {self.session.get_providers()}")
    def detect(self, img, conf_threshold=0.3, nms_threshold=0.5):
        """检测车牌区域，返回 [(x1, y1, x2, y2, score, class_id), ...]"""
        h, w = img.shape[:2]
        input_h, input_w = self.input_size
        
        # 预处理
        resized = cv2.resize(img, (input_w, input_h))
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)
        
        # 推理
        output = self.session.run(None, {self.input_name: blob})[0]
        
        # 后处理 - YOLOv5 输出: (1, 25200, 15) = (batch, anchors, 4+1+10)
        predictions = output[0]  # (25200, 15)
        
        boxes = []
        for pred in predictions:
            obj_conf = pred[4]
            if obj_conf < conf_threshold:
                continue
            
            # 取10类中最大的
            class_confs = pred[5:]  # 10 classes
            class_id = np.argmax(class_confs)
            class_conf = class_confs[class_id]
            
            score = obj_conf * class_conf
            if score < conf_threshold:
                continue
            
            cx, cy, bw, bh = pred[0], pred[1], pred[2], pred[3]
            # 映射回原图
            x1 = int((cx - bw/2) / input_w * w)
            y1 = int((cy - bh/2) / input_h * h)
            x2 = int((cx + bw/2) / input_w * w)
            y2 = int((cy + bh/2) / input_h * h)
            
            boxes.append([x1, y1, x2, y2, float(score), int(class_id)])
        
        # NMS
        if len(boxes) > 0:
            boxes = self._nms(boxes, nms_threshold)
        
        return boxes
    
    @staticmethod
    def _nms(boxes, threshold):
        if len(boxes) == 0:
            return []
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        keep = []
        while boxes:
            best = boxes.pop(0)
            keep.append(best)
            remaining = []
            for b in boxes:
                ix1 = max(best[0], b[0])
                iy1 = max(best[1], b[1])
                ix2 = min(best[2], b[2])
                iy2 = min(best[3], b[3])
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    area_a = (best[2] - best[0]) * (best[3] - best[1])
                    area_b = (b[2] - b[0]) * (b[3] - b[1])
                    iou = inter / (area_a + area_b - inter + 1e-6)
                    if iou >= threshold:
                        continue
                remaining.append(b)
            boxes = remaining
        return keep


class LPR3RecognizerORT:
    """车牌识别器 - OpenVINO 加速 (CTC解码)"""
    
    def __init__(self, model_path=None, providers=None):
        if model_path is None:
            model_path = HYPERLPR3_REC_MODEL
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"识别模型不存在: {model_path}")
        
        import onnxruntime as ort
        if providers is None:
            providers = _get_providers()
        self.session = ort.InferenceSession(model_path, providers=providers)
        input_shape = self.session.get_inputs()[0].shape
        self.input_size = (input_shape[2], input_shape[3])
        self.input_name = self.session.get_inputs()[0].name
        print(f"  LPR3Recognizer ORT: {self.session.get_providers()}")
    def recognize(self, plate_img):
        """识别车牌图像，返回 (车牌号, 置信度)"""
        if plate_img is None or plate_img.size == 0:
            return None, 0
        
        h, w = plate_img.shape[:2]
        wh_ratio = w * 1.0 / h
        
        # 预处理（与HyperLPR3一致）
        data = encode_images(plate_img, wh_ratio, self.input_size)
        data = np.expand_dims(data, 0)  # (1, 3, 48, 160)
        
        # 推理
        output = self.session.run(None, {self.input_name: data})[0]
        
        # 后处理 - CTC解码
        prod = output  # probability distribution
        argmax = np.argmax(prod, axis=2)  # (1, 20)
        rmax = np.max(prod, axis=2)  # (1, 20)
        
        # CTC解码：去重 + 去blank
        char_list = []
        conf_list = []
        prev_idx = -1
        
        for i in range(argmax.shape[1]):
            idx = int(argmax[0, i])
            conf = float(rmax[0, i])
            
            if idx in IGNORED_TOKENS:
                prev_idx = idx
                continue
            # 去重
            if idx == prev_idx:
                prev_idx = idx
                continue
            
            if idx < len(TOKEN):
                char_list.append(TOKEN[idx])
                conf_list.append(conf)
            
            prev_idx = idx
        
        if not char_list:
            return None, 0
        
        plate_text = ''.join(char_list)
        avg_conf = float(np.mean(conf_list)) if conf_list else 0
        
        if len(plate_text) >= 7:
            return plate_text, float(avg_conf)
        
        return None, 0


class LPR3ClassifierORT:
    """车牌分类器 - 判断车牌类型"""
    
    PLATE_TYPES = {
        0: "蓝牌", 1: "黄牌单层", 2: "白牌单层", 3: "绿牌新能源",
        4: "黑牌港澳", 5: "香港单层", 6: "香港双层", 7: "澳门单层",
        8: "澳门双层", 9: "黄牌双层",
    }
    
    def __init__(self, model_path=None, providers=None):
        if model_path is None:
            model_path = HYPERLPR3_CLS_MODEL
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"分类模型不存在: {model_path}")
        
        import onnxruntime as ort
        if providers is None:
            providers = _get_providers()
        self.session = ort.InferenceSession(model_path, providers=providers)
        input_shape = self.session.get_inputs()[0].shape
        self.input_size = (input_shape[2], input_shape[3])
        self.input_name = self.session.get_inputs()[0].name
    def classify(self, plate_img):
        if plate_img is None or plate_img.size == 0:
            return -1, 0
        input_h, input_w = self.input_size
        resized = cv2.resize(plate_img, (input_w, input_h))
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]
        output = self.session.run(None, {self.input_name: blob})[0]
        idx = np.argmax(output[0])
        conf = output[0][idx]
        return int(idx), float(conf)


class LicensePlateCatcherORT:
    """HyperLPR3 替代品 - ONNXRuntime CUDA 加速版
    
    完全兼容 HyperLPR3 的接口，但用 ONNXRuntime CUDA 推理
    """
    
    def __init__(self, providers=None, det_level=1, allow_cpu_fallback=True):
        """
        Args:
            providers: ONNXRuntime providers (None=auto-detect)
            det_level: 0=低精度(320x), 1=高精度(640x)
            allow_cpu_fallback: 是否允许回退到 CPU
        """
        if det_level == 0:
            det_path = HYPERLPR3_DET_MODEL_LOW
        else:
            det_path = HYPERLPR3_DET_MODEL

        if providers is None:
            providers = _get_providers(allow_cpu_fallback=allow_cpu_fallback)
        
        self.detector = LPR3DetectorORT(det_path, providers=providers)
        self.recognizer = LPR3RecognizerORT(providers=providers)
        # 当前流程直接使用检测模型输出的类别，不再额外初始化 CPU 分类模型，
        # 这样在严格 GPU 模式下可避免任何 CPU 回退。
        self.classifier = None
        self.allow_cpu_fallback = allow_cpu_fallback
        self._fallback_catcher = None
    
    def _get_fallback(self):
        """懒加载 HyperLPR3 原版作为 fallback"""
        if self._fallback_catcher is None:
            try:
                import hyperlpr3 as lpr3
                zip_path = os.path.expanduser("~/.hyperlpr3/20230229.zip")
                if os.path.exists(zip_path):
                    try:
                        os.remove(zip_path)
                    except:
                        pass
                self._fallback_catcher = lpr3.LicensePlateCatcher(detect_level=lpr3.DETECT_LEVEL_HIGH)
            except:
                pass
        return self._fallback_catcher
    
    def __call__(self, img):
        """
        识别图像中的车牌
        兼容 HyperLPR3 接口
        Returns:
            list of [plate_text, confidence, plate_type, [x1,y1,x2,y2]]
        """
        # 1. 检测车牌区域
        detections = self.detector.detect(img, conf_threshold=0.3)
        
        results = []
        for x1, y1, x2, y2, det_conf, class_id in detections:
            # 裁剪车牌区域
            h, w = img.shape[:2]
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(w, x2), min(h, y2)
            plate_img = img[cy1:cy2, cx1:cx2]
            
            if plate_img.size == 0:
                continue
            
            # 2. 识别车牌号
            plate_text, rec_conf = self.recognizer.recognize(plate_img)
            
            # 3. 分类车牌类型（用检测模型的类别或分类器）
            ptype = class_id  # 检测模型直接给出类别
            
            if plate_text and len(plate_text) >= 7:
                conf = rec_conf  # 用识别置信度为主
                results.append([plate_text, np.float32(conf), int(ptype), [x1, y1, x2, y2]])
        
        # Fallback: 如果没识别出来，用原版
        if not results and self.allow_cpu_fallback:
            fallback = self._get_fallback()
            if fallback:
                try:
                    fb_results = fallback(img)
                    if fb_results:
                        for r in fb_results:
                            if len(r) > 0 and len(str(r[0])) >= 7:
                                results.append(r)
                except:
                    pass
        
        return results
