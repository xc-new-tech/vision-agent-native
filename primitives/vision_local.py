"""
本地视觉模型原语

包含基于本地 API 的检测、分割、特征与增强能力。
"""

import base64
import io
import json
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image
import requests

from vision_agent_native.config import DEFAULT_CONFIG


def detect_objects(
    image: Union[str, np.ndarray],
    prompt: str,
    threshold: float = 0.3,
    api_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """对象检测 - 使用本地 YOLO-World 模型

    Args:
        image: 图像路径或 numpy 数组
        prompt: 要检测的对象（如 "person", "car,dog"）
        threshold: 置信度阈值
        api_url: 本地 API 地址（默认 http://localhost:8001）

    Returns:
        检测结果 [{"label": str, "bbox": [x1,y1,x2,y2], "score": float}]
        bbox 为归一化坐标 (0-1)
    """
    api_url = api_url or DEFAULT_CONFIG.vision_api_url

    if isinstance(image, str):
        with open(image, "rb") as f:
            image_bytes = f.read()
        img = Image.open(image)
        width, height = img.size
    else:
        img = Image.fromarray(image)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
        height, width = image.shape[:2]

    prompts = [p.strip() for p in prompt.split(",")]

    try:
        response = requests.post(
            f"{api_url}/v1/tools/text-to-object-detection",
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            data={
                "prompts": json.dumps(prompts),
                "confidence": threshold,
            },
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()

        detections = []
        for det in result.get("data", [[]])[0]:
            bbox = det.get("bounding_box", [])
            if bbox:
                norm_bbox = [
                    bbox[0] / width,
                    bbox[1] / height,
                    bbox[2] / width,
                    bbox[3] / height,
                ]
            else:
                norm_bbox = []

            detections.append({
                "label": det.get("label", ""),
                "bbox": norm_bbox,
                "score": det.get("score", 0.0),
            })

        return detections

    except requests.exceptions.ConnectionError:
        print(f"[警告] 本地视觉 API 未运行: {api_url}")
        print("请启动: cd local-vision-api && python app.py")
        return []
    except Exception as e:
        print(f"[错误] detect_objects 失败: {e}")
        return []


def segment_objects(
    image: Union[str, np.ndarray],
    detections: List[Dict[str, Any]],
    api_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """实例分割 - 使用本地 SAM 模型

    基于检测结果中的 bbox 进行分割。

    Args:
        image: 图像路径或 numpy 数组
        detections: 检测结果列表，需包含 bbox 字段（归一化或像素坐标）
        api_url: 本地 API 地址

    Returns:
        分割结果 [{"label": str, "bbox": [...], "mask": np.ndarray}]
    """
    api_url = api_url or DEFAULT_CONFIG.vision_api_url

    if isinstance(image, str):
        with open(image, "rb") as f:
            image_bytes = f.read()
        img = Image.open(image)
        height, width = img.size
    else:
        img = Image.fromarray(image)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
        height, width = image.shape[:2]

    bboxes = []
    labels = []
    for det in detections:
        bbox = det.get("bbox", [])
        if len(bbox) == 4 and max(bbox) <= 1:
            bbox = [bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height]
        bboxes.append(bbox)
        labels.append(det.get("label", "object"))

    if not bboxes:
        return []

    try:
        response = requests.post(
            f"{api_url}/v1/tools/sam2",
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            data={
                "bboxes": json.dumps([{"labels": labels, "bboxes": bboxes}]),
            },
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()

        segments = []
        data = result.get("data", [[[]]])
        if data and data[0] and data[0][0]:
            for seg in data[0][0]:
                mask_data = seg.get("mask", {})

                if isinstance(mask_data, dict) and "counts" in mask_data:
                    mask = _rle_decode(mask_data["counts"], mask_data.get("size", [height, width]))
                else:
                    mask = None

                segments.append({
                    "label": seg.get("label", ""),
                    "bbox": seg.get("bounding_box", []),
                    "score": seg.get("score", 1.0),
                    "mask": mask,
                })

        return segments

    except requests.exceptions.ConnectionError:
        print(f"[警告] 本地视觉 API 未运行: {api_url}")
        print("请启动: cd local-vision-api && python app.py")
        return []
    except Exception as e:
        print(f"[错误] segment_objects 失败: {e}")
        return []


def segment_by_text(
    image: Union[str, np.ndarray],
    prompt: str,
    threshold: float = 0.3,
    api_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """概念分割 - 使用 SAM3 直接用文本分割

    SAM3 新特性：无需边界框，直接用文本概念分割。

    Args:
        image: 图像路径或 numpy 数组
        prompt: 要分割的对象（如 "person", "car,dog"）
        threshold: 置信度阈值
        api_url: 本地 API 地址

    Returns:
        分割结果 [{"label": str, "bbox": [...], "mask": np.ndarray}]
    """
    api_url = api_url or DEFAULT_CONFIG.vision_api_url

    if isinstance(image, str):
        with open(image, "rb") as f:
            image_bytes = f.read()
        img = Image.open(image)
        height, width = img.size
    else:
        img = Image.fromarray(image)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
        height, width = image.shape[:2]

    prompts = [p.strip() for p in prompt.split(",")]

    try:
        response = requests.post(
            f"{api_url}/v1/tools/sam3-concept",
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            data={
                "prompts": json.dumps(prompts),
                "confidence": threshold,
            },
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()

        segments = []
        data = result.get("data", [[[]]])
        if data and data[0] and data[0][0]:
            for seg in data[0][0]:
                mask_data = seg.get("mask", {})

                if isinstance(mask_data, dict) and "counts" in mask_data:
                    mask = _rle_decode(mask_data["counts"], mask_data.get("size", [height, width]))
                else:
                    mask = None

                segments.append({
                    "label": seg.get("label", ""),
                    "bbox": seg.get("bounding_box", []),
                    "score": seg.get("score", 1.0),
                    "mask": mask,
                })

        return segments

    except requests.exceptions.ConnectionError:
        print(f"[警告] 本地视觉 API 未运行: {api_url}")
        print("请启动: cd local-vision-api && python app.py")
        return []
    except Exception as e:
        print(f"[错误] segment_by_text 失败: {e}")
        return []


def estimate_pose(
    image: Union[str, np.ndarray],
    threshold: float = 0.5,
    api_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """人体姿态估计 - 检测人体关键点"""
    api_url = api_url or DEFAULT_CONFIG.vision_api_url

    if isinstance(image, str):
        with open(image, "rb") as f:
            image_bytes = f.read()
    else:
        img = Image.fromarray(image)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

    try:
        response = requests.post(
            f"{api_url}/v1/tools/pose-estimation",
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            data={"confidence": threshold},
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()

        detections = []
        for det in result.get("data", []):
            detections.append({
                "label": det.get("label", "person"),
                "score": det.get("score", 0.0),
                "bbox": det.get("bounding_box", []),
                "keypoints": det.get("keypoints", []),
            })

        return detections

    except requests.exceptions.ConnectionError:
        print(f"[警告] 本地视觉 API 未运行: {api_url}")
        print("请启动: cd local-vision-api && python app.py")
        return []
    except Exception as e:
        print(f"[错误] estimate_pose 失败: {e}")
        return []


def detect_faces(
    image: Union[str, np.ndarray],
    threshold: float = 0.5,
    api_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """人脸检测"""
    api_url = api_url or DEFAULT_CONFIG.vision_api_url

    if isinstance(image, str):
        with open(image, "rb") as f:
            image_bytes = f.read()
    else:
        img = Image.fromarray(image)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

    try:
        response = requests.post(
            f"{api_url}/v1/tools/face-detection",
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            data={"confidence": threshold},
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()

        detections = []
        for det in result.get("data", []):
            detections.append({
                "label": det.get("label", "face"),
                "score": det.get("score", 0.0),
                "bbox": det.get("bounding_box", []),
                "landmarks": det.get("landmarks"),
            })

        return detections

    except requests.exceptions.ConnectionError:
        print(f"[警告] 本地视觉 API 未运行: {api_url}")
        print("请启动: cd local-vision-api && python app.py")
        return []
    except Exception as e:
        print(f"[错误] detect_faces 失败: {e}")
        return []


def classify_image(
    image: Union[str, np.ndarray],
    top_k: int = 5,
    api_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """图像分类"""
    api_url = api_url or DEFAULT_CONFIG.vision_api_url

    if isinstance(image, str):
        with open(image, "rb") as f:
            image_bytes = f.read()
    else:
        img = Image.fromarray(image)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

    try:
        response = requests.post(
            f"{api_url}/v1/tools/image-classification",
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            data={"top_k": top_k},
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()

        return result.get("data", [])

    except requests.exceptions.ConnectionError:
        print(f"[警告] 本地视觉 API 未运行: {api_url}")
        print("请启动: cd local-vision-api && python app.py")
        return []
    except Exception as e:
        print(f"[错误] classify_image 失败: {e}")
        return []


def get_embedding(
    image: Union[str, np.ndarray],
    api_url: Optional[str] = None,
) -> List[float]:
    """获取图像嵌入向量 (CLIP)"""
    api_url = api_url or DEFAULT_CONFIG.vision_api_url

    if isinstance(image, str):
        with open(image, "rb") as f:
            image_bytes = f.read()
    else:
        img = Image.fromarray(image)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

    try:
        response = requests.post(
            f"{api_url}/v1/tools/image-embedding",
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()

        return result.get("data", {}).get("embedding", [])

    except requests.exceptions.ConnectionError:
        print(f"[警告] 本地视觉 API 未运行: {api_url}")
        print("请启动: cd local-vision-api && python app.py")
        return []
    except Exception as e:
        print(f"[错误] get_embedding 失败: {e}")
        return []


def compare_images(
    image1: Union[str, np.ndarray],
    image2: Union[str, np.ndarray],
    api_url: Optional[str] = None,
) -> float:
    """比较两张图像的相似度 (使用 CLIP)"""
    api_url = api_url or DEFAULT_CONFIG.vision_api_url

    if isinstance(image1, str):
        with open(image1, "rb") as f:
            image1_bytes = f.read()
    else:
        img = Image.fromarray(image1)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image1_bytes = buffer.getvalue()

    if isinstance(image2, str):
        with open(image2, "rb") as f:
            image2_bytes = f.read()
    else:
        img = Image.fromarray(image2)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image2_bytes = buffer.getvalue()

    try:
        response = requests.post(
            f"{api_url}/v1/tools/image-similarity",
            files={
                "image1": ("image1.jpg", image1_bytes, "image/jpeg"),
                "image2": ("image2.jpg", image2_bytes, "image/jpeg"),
            },
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()

        return result.get("data", {}).get("similarity", 0.0)

    except requests.exceptions.ConnectionError:
        print(f"[警告] 本地视觉 API 未运行: {api_url}")
        print("请启动: cd local-vision-api && python app.py")
        return 0.0
    except Exception as e:
        print(f"[错误] compare_images 失败: {e}")
        return 0.0


def remove_background(
    image: Union[str, np.ndarray],
    foreground: str = "foreground object",
    api_url: Optional[str] = None,
) -> Dict[str, Any]:
    """背景移除 - 使用 SAM3 分割前景"""
    api_url = api_url or DEFAULT_CONFIG.vision_api_url

    if isinstance(image, str):
        with open(image, "rb") as f:
            image_bytes = f.read()
    else:
        img = Image.fromarray(image)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

    try:
        response = requests.post(
            f"{api_url}/v1/tools/remove-background",
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            data={"foreground": foreground},
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()

        data = result.get("data", {})

        image_data = base64.b64decode(data.get("image_base64", ""))
        result_image = np.array(Image.open(io.BytesIO(image_data)))

        mask_data = base64.b64decode(data.get("mask_base64", ""))
        mask = np.array(Image.open(io.BytesIO(mask_data)))

        return {
            "image": result_image,
            "mask": mask,
            "width": data.get("width", 0),
            "height": data.get("height", 0),
        }

    except requests.exceptions.ConnectionError:
        print(f"[警告] 本地视觉 API 未运行: {api_url}")
        print("请启动: cd local-vision-api && python app.py")
        return {"image": None, "mask": None, "width": 0, "height": 0}
    except Exception as e:
        print(f"[错误] remove_background 失败: {e}")
        return {"image": None, "mask": None, "width": 0, "height": 0}


def upscale_image(
    image: Union[str, np.ndarray],
    scale: int = 2,
    api_url: Optional[str] = None,
) -> Dict[str, Any]:
    """图像超分辨率"""
    api_url = api_url or DEFAULT_CONFIG.vision_api_url

    if isinstance(image, str):
        with open(image, "rb") as f:
            image_bytes = f.read()
    else:
        img = Image.fromarray(image)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

    try:
        response = requests.post(
            f"{api_url}/v1/tools/upscale-image",
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            data={"scale": scale},
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()

        data = result.get("data", {})

        image_data = base64.b64decode(data.get("image_base64", ""))
        result_image = np.array(Image.open(io.BytesIO(image_data)))

        return {
            "image": result_image,
            "width": data.get("width", 0),
            "height": data.get("height", 0),
            "scale": data.get("scale", scale),
        }

    except requests.exceptions.ConnectionError:
        print(f"[警告] 本地视觉 API 未运行: {api_url}")
        print("请启动: cd local-vision-api && python app.py")
        return {"image": None, "width": 0, "height": 0, "scale": 0}
    except Exception as e:
        print(f"[错误] upscale_image 失败: {e}")
        return {"image": None, "width": 0, "height": 0, "scale": 0}


def filter_by_score(
    detections: List[Dict[str, Any]],
    threshold: float,
) -> List[Dict[str, Any]]:
    """按置信度过滤检测结果"""
    return [d for d in detections if d.get("score", 0) >= threshold]


def filter_by_label(
    detections: List[Dict[str, Any]],
    labels: List[str],
) -> List[Dict[str, Any]]:
    """按标签过滤检测结果"""
    return [d for d in detections if d.get("label") in labels]


def _rle_decode(counts: List[int], size: List[int]) -> np.ndarray:
    """解码 RLE 掩码"""
    h, w = size
    mask = np.zeros(h * w, dtype=np.uint8)

    pos = 0
    val = 0
    for count in counts:
        mask[pos:pos + count] = val
        pos += count
        val = 1 - val

    return mask.reshape((h, w), order="F")
