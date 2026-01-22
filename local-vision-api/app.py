"""
本地视觉工具 API 服务
提供 YOLO-World 检测和 SAM2/SAM3 分割功能

SAM3 支持:
- 原生文本概念分割 (需要 sam3.pt 权重)
- 如果 SAM3 不可用，自动回退到 YOLO-World + SAM2 组合
"""

import base64
import io
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# 延迟加载模型
_yolo_model = None
_sam_model = None  # SAM2 (bbox-based)
_sam3_predictor = None  # SAM3 (text-based)
_sam3_available = None  # 缓存 SAM3 可用性检查

# 新增 CV 模型
_yolo_pose_model = None  # 姿态估计
# _yolo_face_model 已移除 - 人脸检测复用 pose 模型
_yolo_cls_model = None   # 图像分类
_clip_model = None       # CLIP 图像嵌入
_clip_preprocess = None

# 默认设备配置
def get_device():
    """获取推理设备 (优先 GPU)"""
    device = os.environ.get("DEVICE", "auto")
    if device == "auto":
        if torch.cuda.is_available():
            return 0  # CUDA GPU
        elif torch.backends.mps.is_available():
            return "mps"  # Apple Silicon GPU
        else:
            return "cpu"
    return device

DEVICE = get_device()
print(f"Using device: {DEVICE}")

app = FastAPI(title="Local Vision API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_yolo_model():
    """延迟加载 YOLO-World 模型"""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO

        model_path = os.environ.get("YOLO_MODEL_PATH", "yolov8x-worldv2.pt")
        print(f"Loading YOLO-World model: {model_path} on {DEVICE}")
        _yolo_model = YOLO(model_path)
        _yolo_model.to(DEVICE)
    return _yolo_model


def get_sam_model():
    """延迟加载 SAM2 模型 (bbox-based 分割)"""
    global _sam_model
    if _sam_model is None:
        from ultralytics import SAM

        model_path = os.environ.get("SAM_MODEL_PATH", "sam2_l.pt")
        print(f"Loading SAM2 model: {model_path} on {DEVICE}")
        _sam_model = SAM(model_path)
        _sam_model.to(DEVICE)
    return _sam_model


def is_sam3_available() -> bool:
    """检查 SAM3 是否可用"""
    global _sam3_available
    if _sam3_available is not None:
        return _sam3_available

    sam3_path = os.environ.get("SAM3_MODEL_PATH", "sam3.pt")
    _sam3_available = os.path.exists(sam3_path)
    if _sam3_available:
        print(f"SAM3 available: {sam3_path}")
    else:
        print(f"SAM3 not found at {sam3_path}, will use YOLO+SAM2 fallback")
    return _sam3_available


def get_sam3_predictor():
    """延迟加载 SAM3 文本分割预测器"""
    global _sam3_predictor
    if _sam3_predictor is None:
        if not is_sam3_available():
            return None

        try:
            from ultralytics.models.sam import SAM3SemanticPredictor

            sam3_path = os.environ.get("SAM3_MODEL_PATH", "sam3.pt")
            print(f"Loading SAM3 model: {sam3_path} on {DEVICE}")

            # 设置设备
            device_str = str(DEVICE) if DEVICE != 0 else "0"

            overrides = dict(
                conf=0.25,
                task="segment",
                mode="predict",
                model=sam3_path,
                device=device_str,
                half=True,  # FP16 加速
                save=False,
            )
            _sam3_predictor = SAM3SemanticPredictor(overrides=overrides)
        except ImportError:
            print("SAM3SemanticPredictor not available in current Ultralytics version")
            return None
        except Exception as e:
            print(f"Failed to load SAM3: {e}")
            return None

    return _sam3_predictor


def get_yolo_pose_model():
    """延迟加载 YOLO-Pose 姿态估计模型"""
    global _yolo_pose_model
    if _yolo_pose_model is None:
        from ultralytics import YOLO

        model_path = os.environ.get("YOLO_POSE_PATH", "yolo11x-pose.pt")
        print(f"Loading YOLO-Pose model: {model_path} on {DEVICE}")
        _yolo_pose_model = YOLO(model_path)
        _yolo_pose_model.to(DEVICE)
    return _yolo_pose_model


def get_yolo_face_model():
    """人脸检测使用 pose 模型提取人脸区域"""
    # 复用 pose 模型 - 从头部关键点提取人脸
    return get_yolo_pose_model()


def get_yolo_cls_model():
    """延迟加载 YOLO 图像分类模型"""
    global _yolo_cls_model
    if _yolo_cls_model is None:
        from ultralytics import YOLO

        model_path = os.environ.get("YOLO_CLS_PATH", "yolo11x-cls.pt")
        print(f"Loading YOLO-Cls model: {model_path} on {DEVICE}")
        _yolo_cls_model = YOLO(model_path)
        _yolo_cls_model.to(DEVICE)
    return _yolo_cls_model


def get_clip_model():
    """延迟加载 CLIP 模型用于图像嵌入"""
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        import clip

        model_name = os.environ.get("CLIP_MODEL", "ViT-B/32")
        device = "cuda" if DEVICE == 0 else ("mps" if DEVICE == "mps" else "cpu")
        print(f"Loading CLIP model: {model_name} on {device}")
        _clip_model, _clip_preprocess = clip.load(model_name, device=device)
    return _clip_model, _clip_preprocess


def image_bytes_to_numpy(image_bytes: bytes) -> np.ndarray:
    """将图像字节转换为 numpy 数组"""
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


def rle_encode(mask: np.ndarray) -> Dict[str, Any]:
    """将二值掩码编码为 RLE 格式"""
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return {
        "counts": runs.tolist(),
        "size": list(mask.shape),
    }


def rle_decode(counts: List[int], size: List[int]) -> np.ndarray:
    """将 RLE 解码为二值掩码"""
    h, w = size
    mask = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    val = 0
    for count in counts:
        mask[pos:pos + count] = val
        pos += count
        val = 1 - val
    return mask.reshape((h, w), order='F')


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok"}


@app.get("/status")
async def model_status():
    """查看模型加载状态"""
    sam3_path = os.environ.get("SAM3_MODEL_PATH", "sam3.pt")
    sam2_path = os.environ.get("SAM_MODEL_PATH", "sam2_l.pt")
    yolo_path = os.environ.get("YOLO_MODEL_PATH", "yolov8x-worldv2.pt")

    # 设备信息
    device_name = str(DEVICE)
    if DEVICE == 0:
        device_name = f"cuda:0 ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "cuda:0"
    elif DEVICE == "mps":
        device_name = "mps (Apple Silicon GPU)"

    return {
        "device": device_name,
        "models": {
            "yolo_world": {
                "path": yolo_path,
                "available": os.path.exists(yolo_path),
                "loaded": _yolo_model is not None,
            },
            "sam2": {
                "path": sam2_path,
                "available": os.path.exists(sam2_path),
                "loaded": _sam_model is not None,
            },
            "sam3": {
                "path": sam3_path,
                "available": os.path.exists(sam3_path),
                "loaded": _sam3_predictor is not None,
                "note": "原生文本概念分割" if os.path.exists(sam3_path) else "使用 YOLO+SAM2 回退",
            },
        }
    }


@app.post("/v1/tools/text-to-object-detection")
async def object_detection(
    image: UploadFile = File(...),
    prompts: str = Form(...),
    confidence: float = Form(0.3),
    model: str = Form("yolo-world"),
):
    """
    YOLO-World 开放词汇对象检测

    输入:
    - image: 图像文件
    - prompts: JSON 格式的提示词列表，如 '["person", "car"]'
    - confidence: 置信度阈值
    - model: 模型名称 (yolo-world)

    输出格式与 LandingAI API 一致
    """
    try:
        # 解析 prompts
        prompt_list = json.loads(prompts)
        if isinstance(prompt_list, str):
            prompt_list = [prompt_list]

        # 读取图像
        image_bytes = await image.read()
        img_array = image_bytes_to_numpy(image_bytes)
        height, width = img_array.shape[:2]

        # 加载模型并设置类别
        yolo_model = get_yolo_model()
        yolo_model.set_classes(prompt_list)

        # 推理
        results = yolo_model.predict(img_array, conf=confidence, verbose=False)

        # 格式化结果
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    label = prompt_list[cls_id] if cls_id < len(prompt_list) else f"class_{cls_id}"

                    detections.append({
                        "label": label,
                        "score": round(conf, 3),
                        "bounding_box": [
                            int(box[0]),  # x1
                            int(box[1]),  # y1
                            int(box[2]),  # x2
                            int(box[3]),  # y2
                        ],
                    })

        # 返回与 LandingAI 一致的格式
        return {"data": [detections]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/tools/sam2")
async def sam_segmentation(
    image: UploadFile = File(...),
    bboxes: str = Form(...),
    model: str = Form("sam2"),
):
    """
    SAM3 实例分割 (兼容 sam2 端点名)

    输入:
    - image: 图像文件
    - bboxes: JSON 格式的边界框列表，如 '[{"labels": ["person"], "bboxes": [[x1,y1,x2,y2]]}]'
    - model: 模型名称

    输出格式与 LandingAI API 一致，mask 为 RLE 编码
    """
    try:
        # 解析 bboxes
        bbox_data = json.loads(bboxes)
        if isinstance(bbox_data, list) and len(bbox_data) > 0:
            bbox_info = bbox_data[0]
        else:
            bbox_info = bbox_data

        labels = bbox_info.get("labels", [])
        boxes = bbox_info.get("bboxes", [])

        if not boxes:
            return {"data": [[]]}

        # 读取图像
        image_bytes = await image.read()
        img_array = image_bytes_to_numpy(image_bytes)

        # 加载 SAM3 模型
        sam_model = get_sam_model()

        # 使用边界框进行分割
        results = sam_model(img_array, bboxes=boxes, verbose=False)

        # 格式化结果
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.masks is not None:
                for i, mask in enumerate(result.masks.data):
                    label = labels[i] if i < len(labels) else f"object_{i}"
                    box = boxes[i] if i < len(boxes) else [0, 0, 0, 0]

                    mask_np = mask.cpu().numpy().astype(bool)

                    detections.append({
                        "label": label,
                        "score": 1.0,  # SAM3 通过 bbox 分割通常没有置信度
                        "bounding_box": box,
                        "mask": rle_encode(mask_np),
                    })

        # 返回与 LandingAI 一致的格式
        return {"data": [[detections]]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _sam3_native_segment(img_array: np.ndarray, prompt_list: List[str], confidence: float) -> List[Dict]:
    """使用原生 SAM3 进行文本概念分割"""
    predictor = get_sam3_predictor()
    if predictor is None:
        return None

    try:
        # 设置图像
        predictor.set_image(img_array)

        # 使用文本提示分割
        results = predictor(text=prompt_list)

        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.masks is not None:
                for i, mask in enumerate(result.masks.data):
                    mask_np = mask.cpu().numpy().astype(bool)

                    # SAM3 返回的标签和置信度
                    label = prompt_list[i % len(prompt_list)] if prompt_list else f"object_{i}"
                    score = float(result.boxes.conf[i].cpu().numpy()) if result.boxes is not None else 1.0

                    # 从 mask 计算 bbox
                    ys, xs = np.where(mask_np)
                    if len(xs) > 0 and len(ys) > 0:
                        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                    else:
                        bbox = [0, 0, 0, 0]

                    detections.append({
                        "label": label,
                        "score": round(score, 3),
                        "bounding_box": bbox,
                        "mask": rle_encode(mask_np),
                    })

        return detections
    except Exception as e:
        print(f"SAM3 native segmentation failed: {e}")
        return None


async def _yolo_sam2_segment(img_array: np.ndarray, prompt_list: List[str], confidence: float) -> List[Dict]:
    """使用 YOLO-World + SAM2 组合进行分割 (SAM3 回退方案)"""
    # Step 1: 用 YOLO-World 检测
    yolo_model = get_yolo_model()
    yolo_model.set_classes(prompt_list)
    yolo_results = yolo_model.predict(img_array, conf=confidence, verbose=False)

    # 收集检测到的边界框
    bboxes = []
    labels = []
    scores = []

    for result in yolo_results:
        boxes = result.boxes
        if boxes is not None:
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                label = prompt_list[cls_id] if cls_id < len(prompt_list) else f"class_{cls_id}"

                bboxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
                labels.append(label)
                scores.append(conf)

    # 如果没有检测到任何对象
    if not bboxes:
        return []

    # Step 2: 用 SAM2 分割
    sam_model = get_sam_model()
    sam_results = sam_model(img_array, bboxes=bboxes, verbose=False)

    # 格式化结果
    detections = []
    if sam_results and len(sam_results) > 0:
        result = sam_results[0]
        if result.masks is not None:
            for i, mask in enumerate(result.masks.data):
                mask_np = mask.cpu().numpy().astype(bool)

                label = labels[i] if i < len(labels) else f"object_{i}"
                score = scores[i] if i < len(scores) else 1.0
                bbox = bboxes[i] if i < len(bboxes) else [0, 0, 0, 0]

                detections.append({
                    "label": label,
                    "score": round(score, 3),
                    "bounding_box": bbox,
                    "mask": rle_encode(mask_np),
                })

    return detections


@app.post("/v1/tools/sam3-concept")
async def sam3_concept_segmentation(
    image: UploadFile = File(...),
    prompts: str = Form(...),
    confidence: float = Form(0.3),
):
    """
    文本概念分割

    优先使用原生 SAM3，如果 SAM3 不可用则回退到 YOLO-World + SAM2 组合。

    输入:
    - image: 图像文件
    - prompts: JSON 格式的文本提示词，如 '["person", "car"]'
    - confidence: 置信度阈值

    输出:
    - 分割结果列表，每个包含 label, score, bounding_box, mask (RLE 编码)
    """
    try:
        # 解析 prompts
        prompt_list = json.loads(prompts)
        if isinstance(prompt_list, str):
            prompt_list = [prompt_list]

        # 读取图像
        image_bytes = await image.read()
        img_array = image_bytes_to_numpy(image_bytes)

        # 优先尝试原生 SAM3
        detections = await _sam3_native_segment(img_array, prompt_list, confidence)

        # 如果 SAM3 不可用或失败，回退到 YOLO+SAM2
        if detections is None:
            detections = await _yolo_sam2_segment(img_array, prompt_list, confidence)

        return {"data": [[detections]]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 新增 CV 工具端点
# ============================================================

@app.post("/v1/tools/pose-estimation")
async def pose_estimation(
    image: UploadFile = File(...),
    confidence: float = Form(0.5),
):
    """
    人体姿态估计 - 检测人体关键点

    输入:
    - image: 图像文件
    - confidence: 置信度阈值

    输出:
    - 检测结果列表，每个包含 bbox, keypoints (17个关键点), score
    """
    try:
        image_bytes = await image.read()
        img_array = image_bytes_to_numpy(image_bytes)

        model = get_yolo_pose_model()
        results = model.predict(img_array, conf=confidence, verbose=False)

        detections = []
        for result in results:
            if result.keypoints is not None:
                for i in range(len(result.keypoints)):
                    kpts = result.keypoints[i].data.cpu().numpy()[0]  # [17, 3]
                    box = result.boxes.xyxy[i].cpu().numpy() if result.boxes is not None else [0, 0, 0, 0]
                    conf = float(result.boxes.conf[i].cpu().numpy()) if result.boxes is not None else 1.0

                    # 格式化关键点: [[x, y, conf], ...]
                    keypoints = []
                    for kpt in kpts:
                        keypoints.append({
                            "x": float(kpt[0]),
                            "y": float(kpt[1]),
                            "confidence": float(kpt[2]) if len(kpt) > 2 else 1.0,
                        })

                    detections.append({
                        "label": "person",
                        "score": round(conf, 3),
                        "bounding_box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                        "keypoints": keypoints,
                    })

        return {"data": detections}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/tools/face-detection")
async def face_detection(
    image: UploadFile = File(...),
    confidence: float = Form(0.5),
):
    """
    人脸检测 - 使用 pose 模型提取人脸区域

    输入:
    - image: 图像文件
    - confidence: 置信度阈值

    输出:
    - 检测结果列表，每个包含 bbox, landmarks (5个关键点: 左眼、右眼、鼻子、左嘴角、右嘴角), score
    """
    try:
        image_bytes = await image.read()
        img_array = image_bytes_to_numpy(image_bytes)

        model = get_yolo_face_model()  # 使用 pose 模型
        results = model.predict(img_array, conf=confidence, verbose=False)

        detections = []
        for result in results:
            if result.keypoints is None:
                continue

            for i in range(len(result.keypoints)):
                kpts = result.keypoints[i].data.cpu().numpy()[0]  # (17, 3) - x, y, conf

                # 头部关键点: nose(0), left_eye(1), right_eye(2), left_ear(3), right_ear(4)
                head_indices = [0, 1, 2, 3, 4]
                head_kpts = kpts[head_indices]

                # 过滤低置信度关键点
                valid_kpts = head_kpts[head_kpts[:, 2] > 0.3]
                if len(valid_kpts) < 2:
                    continue

                # 计算人脸边界框
                x_coords = valid_kpts[:, 0]
                y_coords = valid_kpts[:, 1]

                # 扩展边界框以包含完整人脸
                width = x_coords.max() - x_coords.min()
                height = y_coords.max() - y_coords.min()
                padding_x = width * 0.5
                padding_y = height * 0.8

                x1 = max(0, int(x_coords.min() - padding_x))
                y1 = max(0, int(y_coords.min() - padding_y))
                x2 = int(x_coords.max() + padding_x)
                y2 = int(y_coords.max() + padding_y * 1.5)

                # 构建5点人脸关键点 (眼、鼻、嘴角)
                # COCO pose 没有嘴角，用鼻子两侧估算
                landmarks = []
                # 左眼
                if kpts[1, 2] > 0.3:
                    landmarks.append({"x": float(kpts[1, 0]), "y": float(kpts[1, 1])})
                else:
                    landmarks.append(None)
                # 右眼
                if kpts[2, 2] > 0.3:
                    landmarks.append({"x": float(kpts[2, 0]), "y": float(kpts[2, 1])})
                else:
                    landmarks.append(None)
                # 鼻子
                if kpts[0, 2] > 0.3:
                    landmarks.append({"x": float(kpts[0, 0]), "y": float(kpts[0, 1])})
                else:
                    landmarks.append(None)
                # 左嘴角、右嘴角 (用鼻子位置估算)
                if kpts[0, 2] > 0.3:
                    nose_x, nose_y = kpts[0, 0], kpts[0, 1]
                    face_width = width if width > 0 else 30
                    landmarks.append({"x": float(nose_x - face_width * 0.3), "y": float(nose_y + face_width * 0.3)})
                    landmarks.append({"x": float(nose_x + face_width * 0.3), "y": float(nose_y + face_width * 0.3)})
                else:
                    landmarks.extend([None, None])

                # 使用人体检测置信度
                person_conf = float(result.boxes.conf[i].cpu().numpy()) if result.boxes is not None else 0.8
                avg_kpt_conf = float(valid_kpts[:, 2].mean())
                face_conf = (person_conf + avg_kpt_conf) / 2

                detections.append({
                    "label": "face",
                    "score": round(face_conf, 3),
                    "bounding_box": [x1, y1, x2, y2],
                    "landmarks": landmarks,
                })

        return {"data": detections}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/tools/image-classification")
async def image_classification(
    image: UploadFile = File(...),
    top_k: int = Form(5),
):
    """
    图像分类

    输入:
    - image: 图像文件
    - top_k: 返回前 k 个预测结果

    输出:
    - 分类结果列表，每个包含 label, score
    """
    try:
        image_bytes = await image.read()
        img_array = image_bytes_to_numpy(image_bytes)

        model = get_yolo_cls_model()
        results = model.predict(img_array, verbose=False)

        classifications = []
        for result in results:
            probs = result.probs
            if probs is not None:
                top_indices = probs.top5[:top_k]
                top_confs = probs.top5conf[:top_k].cpu().numpy()

                for idx, conf in zip(top_indices, top_confs):
                    label = result.names[idx]
                    classifications.append({
                        "label": label,
                        "score": round(float(conf), 3),
                    })

        return {"data": classifications}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/tools/image-embedding")
async def image_embedding(
    image: UploadFile = File(...),
):
    """
    图像嵌入 - 使用 CLIP 提取图像特征向量

    输入:
    - image: 图像文件

    输出:
    - embedding: 512维特征向量
    """
    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        model, preprocess = get_clip_model()
        device = next(model.parameters()).device

        image_input = preprocess(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            # 归一化
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        embedding = image_features.cpu().numpy()[0].tolist()

        return {"data": {"embedding": embedding, "dimension": len(embedding)}}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/tools/image-similarity")
async def image_similarity(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
):
    """
    图像相似度比较 - 使用 CLIP 计算两张图像的余弦相似度

    输入:
    - image1: 第一张图像
    - image2: 第二张图像

    输出:
    - similarity: 相似度分数 (0-1)
    """
    try:
        image1_bytes = await image1.read()
        image2_bytes = await image2.read()

        pil_image1 = Image.open(io.BytesIO(image1_bytes)).convert("RGB")
        pil_image2 = Image.open(io.BytesIO(image2_bytes)).convert("RGB")

        model, preprocess = get_clip_model()
        device = next(model.parameters()).device

        image1_input = preprocess(pil_image1).unsqueeze(0).to(device)
        image2_input = preprocess(pil_image2).unsqueeze(0).to(device)

        with torch.no_grad():
            features1 = model.encode_image(image1_input)
            features2 = model.encode_image(image2_input)

            # 归一化
            features1 = features1 / features1.norm(dim=-1, keepdim=True)
            features2 = features2 / features2.norm(dim=-1, keepdim=True)

            # 余弦相似度
            similarity = (features1 @ features2.T).item()

        return {"data": {"similarity": round(similarity, 4)}}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 图像增强
# ============================================================

@app.post("/v1/tools/remove-background")
async def remove_background(
    image: UploadFile = File(...),
    foreground: str = Form("foreground object"),
):
    """
    背景移除 - 使用 SAM3 分割前景

    输入:
    - image: 图像文件
    - foreground: 前景描述 (如 "person", "product")

    输出:
    - image_base64: 带透明背景的 PNG 图像
    - mask_base64: 前景掩码
    """
    try:
        image_bytes = await image.read()
        img_array = image_bytes_to_numpy(image_bytes)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        width, height = pil_image.size

        # 使用 SAM3 分割前景
        detections = await _sam3_native_segment(img_array, [foreground], 0.3)

        if detections is None:
            # 回退到 YOLO + SAM2
            detections = await _yolo_sam2_segment(img_array, [foreground], 0.3)

        # 合并所有前景掩码
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        for det in detections:
            mask_data = det.get("mask", {})
            if isinstance(mask_data, dict) and "counts" in mask_data:
                # RLE 解码
                mask_np = rle_decode(mask_data["counts"], mask_data.get("size", [height, width]))
                combined_mask = np.maximum(combined_mask, (mask_np > 0).astype(np.uint8) * 255)

        # 创建透明背景图像
        rgba_array = np.array(pil_image)
        rgba_array[:, :, 3] = combined_mask  # 设置 alpha 通道

        # 编码为 base64
        result_image = Image.fromarray(rgba_array, mode="RGBA")
        buffer = io.BytesIO()
        result_image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        mask_image = Image.fromarray(combined_mask)
        mask_buffer = io.BytesIO()
        mask_image.save(mask_buffer, format="PNG")
        mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode("utf-8")

        return {
            "data": {
                "image_base64": image_base64,
                "mask_base64": mask_base64,
                "width": width,
                "height": height,
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/tools/upscale-image")
async def upscale_image(
    image: UploadFile = File(...),
    scale: int = Form(2),
):
    """
    图像超分辨率 - 使用 Lanczos 算法放大图像

    输入:
    - image: 图像文件
    - scale: 放大倍数 (2-4)

    输出:
    - image_base64: 放大后的图像
    - width: 输出宽度
    - height: 输出高度
    """
    try:
        if scale < 1 or scale > 4:
            raise HTTPException(status_code=400, detail="Scale must be between 1 and 4")

        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        width, height = pil_image.size
        new_width = width * scale
        new_height = height * scale

        # 使用 Lanczos 算法进行高质量放大
        upscaled = pil_image.resize((new_width, new_height), Image.LANCZOS)

        # 编码为 base64
        buffer = io.BytesIO()
        upscaled.save(buffer, format="JPEG", quality=95)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "data": {
                "image_base64": image_base64,
                "width": new_width,
                "height": new_height,
                "scale": scale,
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
