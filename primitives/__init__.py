"""
原子原语工具

设计原则 (Agent-native):
- 每个工具只做一件事
- 判断逻辑属于 Agent，不属于工具
- 工具可自由组合
- vision_chat 是核心原语，其他视觉操作可通过它实现
"""

from vision_agent_native.primitives.files import (
    read_file,
    write_file,
    list_dir,
    delete_file,
    file_exists,
    read_json,
    write_json,
    append_file,
)
from vision_agent_native.primitives.images import (
    load_image,
    save_image,
    get_image_info,
    resize_image,
    crop_image,
    image_to_base64,
    base64_to_image,
)
from vision_agent_native.primitives.vision import (
    # LLM 视觉能力
    vision_chat,
    ocr,
    vqa,
    # 本地模型 (YOLO + SAM)
    detect_objects,
    segment_objects,
    segment_by_text,
    # 人体/姿态
    estimate_pose,
    detect_faces,
    # 特征提取
    classify_image,
    get_embedding,
    compare_images,
    # 图像增强
    remove_background,
    upscale_image,
    # 辅助
    filter_by_score,
    filter_by_label,
)

__all__ = [
    # Files
    "read_file",
    "write_file",
    "list_dir",
    "delete_file",
    "file_exists",
    "read_json",
    "write_json",
    "append_file",
    # Images
    "load_image",
    "save_image",
    "get_image_info",
    "resize_image",
    "crop_image",
    "image_to_base64",
    "base64_to_image",
    # Vision - LLM 能力
    "vision_chat",      # 核心：任意 prompt 分析图像
    "ocr",              # 文字识别 (LLM)
    "vqa",              # 视觉问答 (LLM)
    # Vision - 本地模型
    "detect_objects",   # YOLO-World 检测
    "segment_objects",  # SAM 分割 (基于 bbox)
    "segment_by_text",  # SAM3 概念分割 (基于文本)
    # Vision - 人体/姿态
    "estimate_pose",    # 人体姿态估计 (17关键点)
    "detect_faces",     # 人脸检测 (5关键点)
    # Vision - 特征提取
    "classify_image",   # 图像分类
    "get_embedding",    # CLIP 图像嵌入
    "compare_images",   # 图像相似度
    # Vision - 图像增强
    "remove_background",  # 背景移除 (SAM3)
    "upscale_image",      # 图像超分辨率
    # 辅助
    "filter_by_score",
    "filter_by_label",
]
