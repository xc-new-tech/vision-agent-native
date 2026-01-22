"""
图像操作原语

原子操作: 加载、保存、基本变换
"""

import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    """加载图像为 numpy 数组

    Args:
        path: 图像路径

    Returns:
        RGB 格式的 numpy 数组 (H, W, C)
    """
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def save_image(image: np.ndarray, path: str, quality: int = 95) -> str:
    """保存图像

    Args:
        image: numpy 数组 (H, W, C)
        path: 保存路径
        quality: JPEG 质量 (1-100)

    Returns:
        保存的文件路径
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(image)

    # 根据扩展名选择格式
    ext = Path(path).suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        img.save(path, "JPEG", quality=quality)
    elif ext == ".png":
        img.save(path, "PNG")
    else:
        img.save(path)

    return path


def get_image_info(path: str) -> Dict[str, Any]:
    """获取图像信息

    Args:
        path: 图像路径

    Returns:
        图像信息 {"width": int, "height": int, "mode": str, "format": str}
    """
    img = Image.open(path)
    return {
        "width": img.width,
        "height": img.height,
        "mode": img.mode,
        "format": img.format,
        "path": path,
    }


def resize_image(
    image: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    max_size: Optional[int] = None,
) -> np.ndarray:
    """调整图像大小

    Args:
        image: 输入图像
        width: 目标宽度 (可选)
        height: 目标高度 (可选)
        max_size: 最大边长 (可选，保持比例)

    Returns:
        调整后的图像
    """
    img = Image.fromarray(image)
    orig_w, orig_h = img.size

    if max_size is not None:
        # 保持比例，最大边不超过 max_size
        ratio = min(max_size / orig_w, max_size / orig_h)
        if ratio < 1:
            width = int(orig_w * ratio)
            height = int(orig_h * ratio)
        else:
            return image
    elif width is not None and height is None:
        # 按宽度等比缩放
        ratio = width / orig_w
        height = int(orig_h * ratio)
    elif height is not None and width is None:
        # 按高度等比缩放
        ratio = height / orig_h
        width = int(orig_w * ratio)
    elif width is None and height is None:
        return image

    img = img.resize((width, height), Image.Resampling.LANCZOS)
    return np.array(img)


def crop_image(
    image: np.ndarray,
    bbox: List[float],
    normalized: bool = True,
) -> np.ndarray:
    """裁剪图像

    Args:
        image: 输入图像
        bbox: 边界框 [x1, y1, x2, y2]
        normalized: bbox 是否为归一化坐标 (0-1)

    Returns:
        裁剪后的图像
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    if normalized:
        x1, x2 = int(x1 * w), int(x2 * w)
        y1, y2 = int(y1 * h), int(y2 * h)
    else:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # 确保在有效范围内
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    return image[y1:y2, x1:x2]


def image_to_base64(image: np.ndarray, format: str = "PNG") -> str:
    """将图像转换为 base64 字符串

    Args:
        image: numpy 数组
        format: 图像格式 (PNG, JPEG)

    Returns:
        base64 编码的字符串
    """
    img = Image.fromarray(image)
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(b64_string: str) -> np.ndarray:
    """将 base64 字符串转换为图像

    Args:
        b64_string: base64 编码的字符串

    Returns:
        numpy 数组
    """
    image_data = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(image_data))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)
