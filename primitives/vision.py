"""
视觉原语聚合

保留原有导入路径，内部按职责拆分。
"""

from vision_agent_native.primitives.vision_llm import (
    vision_chat,
    ocr,
    vqa,
)
from vision_agent_native.primitives.vision_local import (
    detect_objects,
    segment_objects,
    segment_by_text,
    estimate_pose,
    detect_faces,
    classify_image,
    get_embedding,
    compare_images,
    remove_background,
    upscale_image,
    filter_by_score,
    filter_by_label,
)

__all__ = [
    "vision_chat",
    "ocr",
    "vqa",
    "detect_objects",
    "segment_objects",
    "segment_by_text",
    "estimate_pose",
    "detect_faces",
    "classify_image",
    "get_embedding",
    "compare_images",
    "remove_background",
    "upscale_image",
    "filter_by_score",
    "filter_by_label",
]
