"""
LLM 视觉原语

核心原则:
- vision_chat 是最基础的原语
- ocr/vqa 基于 vision_chat 组合
"""

import base64
import io
import json
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from vision_agent_native.config import Config
from vision_agent_native.llm import LLM


def vision_chat(
    images: List[Union[str, np.ndarray, Dict]],
    prompt: str,
    system: Optional[str] = None,
) -> str:
    """视觉对话 - 核心原语

    用 LLM 视觉能力分析一张或多张图像。
    这是最原子的操作，Agent 可以用任意 prompt 来分析图像。

    Args:
        images: 图像列表，支持:
            - 文件路径 (str)
            - numpy 数组
            - dict: {"data": base64, "media_type": "image/jpeg"}
        prompt: 用户提示词
        system: 系统提示词（可选）

    Returns:
        LLM 的响应文本
    """
    formatted_images = []
    for img in images:
        formatted_images.append(_format_image(img))

    config = Config()
    llm = LLM.create(config)

    messages = [
        {
            "role": "user",
            "content": prompt,
            "images": formatted_images,
        }
    ]

    return llm.chat(messages, system=system)


def ocr(
    image: Union[str, np.ndarray],
    language: str = "auto",
) -> List[Dict[str, Any]]:
    """文字识别 - 使用 LLM 能力

    Args:
        image: 图像路径或 numpy 数组
        language: 语言提示（auto/zh/en/...）

    Returns:
        识别结果 [{"text": str, "bbox": [x1,y1,x2,y2], "score": float}]
    """
    lang_hint = ""
    if language == "zh":
        lang_hint = "主要是中文文字。"
    elif language == "en":
        lang_hint = "主要是英文文字。"

    system = "你是 OCR 专家，精确识别图像中的所有文字。"

    prompt = f"""识别图像中的所有文字。{lang_hint}

输出 JSON 格式:
[
  {{"text": "识别的文字", "position": "位置描述(如:左上/中间/右下)"}}
]

只输出 JSON，不要其他内容。"""

    response = vision_chat([image], prompt, system)

    results = []
    try:
        start = response.find("[")
        end = response.rfind("]") + 1
        if start != -1 and end > start:
            parsed = json.loads(response[start:end])
            for item in parsed:
                results.append({
                    "text": item.get("text", ""),
                    "position": item.get("position", ""),
                    "score": 1.0,
                })
    except Exception:
        results.append({"text": response, "position": "", "score": 1.0})

    return results


def vqa(
    image: Union[str, np.ndarray],
    question: str,
) -> str:
    """视觉问答 - 使用 LLM 能力

    Args:
        image: 图像路径或 numpy 数组
        question: 问题

    Returns:
        答案字符串
    """
    return vision_chat([image], question)


def _format_image(image: Union[str, np.ndarray, Dict]) -> Dict:
    """将图像转换为 LLM 接受的格式"""
    if isinstance(image, dict):
        return image
    if isinstance(image, str):
        return _load_image_for_llm(image)
    if isinstance(image, np.ndarray):
        return _array_to_llm_format(image)
    raise ValueError(f"Unsupported image type: {type(image)}")


def _load_image_for_llm(path: str) -> Dict:
    """加载图像文件为 LLM 格式"""
    ext = os.path.splitext(path)[1].lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(ext, "image/jpeg")

    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    return {"data": data, "media_type": media_type}


def _array_to_llm_format(image: np.ndarray, format: str = "PNG") -> Dict:
    """将 numpy 数组转换为 LLM 格式"""
    img = Image.fromarray(image)
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    media_type = "image/png" if format == "PNG" else "image/jpeg"
    return {"data": data, "media_type": media_type}
