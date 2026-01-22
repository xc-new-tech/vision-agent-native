"""
Vision Agent MCP Server

提供图像检测、分割、OCR等视觉工具，供 Claude Code 调用。
"""

import json
import os
import sys
from typing import Optional

# 添加正确的路径以便导入 vision_agent_native
# 项目结构: /path/to/vision_agent_native/ 既是项目根也是包目录
# 需要把 /path/to/ 加入 sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_parent_of_project = os.path.dirname(_project_root)
if _parent_of_project not in sys.path:
    sys.path.insert(0, _parent_of_project)

from fastmcp import FastMCP

# 延迟导入 primitives，避免启动时加载模型
_primitives_loaded = False


def _ensure_primitives():
    """确保 primitives 模块已导入"""
    global _primitives_loaded
    if not _primitives_loaded:
        global vision, images, files
        from vision_agent_native.primitives import vision, images, files
        _primitives_loaded = True


# 创建 MCP Server
mcp = FastMCP(
    "vision-agent",
    instructions="Vision Agent - 提供图像检测、分割、OCR等视觉AI工具。需要本地 Vision API 运行。",
)


# ============================================================
# 核心视觉工具
# ============================================================

@mcp.tool()
def detect_objects(
    image_path: str,
    prompt: str,
    threshold: float = 0.3,
) -> dict:
    """
    检测图像中的对象

    使用 YOLO-World 模型进行开放词汇对象检测。

    Args:
        image_path: 图像文件的绝对路径
        prompt: 要检测的对象，多个用逗号分隔，如 "person,car" 或 "螺母,螺栓"
        threshold: 置信度阈值 (0-1)，默认 0.3

    Returns:
        dict: {
            "count": 检测到的对象数量,
            "detections": [{
                "label": 标签,
                "bbox": [x1, y1, x2, y2] 归一化坐标 (0-1),
                "score": 置信度
            }, ...]
        }

    Note:
        需要本地 Vision API 运行在 VISION_API_URL (默认 http://localhost:8001)
    """
    _ensure_primitives()

    if not os.path.exists(image_path):
        return {"error": f"图像文件不存在: {image_path}"}

    try:
        detections = vision.detect_objects(image_path, prompt, threshold)
        return {
            "count": len(detections),
            "detections": detections,
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def segment_objects(
    image_path: str,
    detections_json: str,
) -> dict:
    """
    基于检测结果分割对象

    使用 SAM2 模型，根据边界框生成精确的分割掩码。

    Args:
        image_path: 图像文件的绝对路径
        detections_json: 检测结果的 JSON 字符串，格式为 detect_objects 的输出
                        或 [{"label": "x", "bbox": [x1,y1,x2,y2], "score": 0.9}, ...]

    Returns:
        dict: {
            "count": 分割数量,
            "segments": [{
                "label": 标签,
                "bbox": 边界框,
                "score": 置信度,
                "mask_rle": RLE 编码的掩码 (可用于后续处理)
            }, ...]
        }
    """
    _ensure_primitives()

    if not os.path.exists(image_path):
        return {"error": f"图像文件不存在: {image_path}"}

    try:
        # 解析检测结果
        data = json.loads(detections_json)
        if isinstance(data, dict) and "detections" in data:
            detections = data["detections"]
        else:
            detections = data

        segments = vision.segment_objects(image_path, detections)

        # 转换结果，移除不可序列化的 mask numpy 数组
        result_segments = []
        for seg in segments:
            result_seg = {
                "label": seg.get("label", ""),
                "bbox": seg.get("bbox", []),
                "score": seg.get("score", 1.0),
            }
            # 如果有 mask，转为 RLE 简化信息
            if seg.get("mask") is not None:
                mask = seg["mask"]
                result_seg["mask_shape"] = list(mask.shape)
                result_seg["mask_area"] = int(mask.sum())
            result_segments.append(result_seg)

        return {
            "count": len(result_segments),
            "segments": result_segments,
        }
    except json.JSONDecodeError as e:
        return {"error": f"JSON 解析错误: {e}"}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def segment_by_text(
    image_path: str,
    prompt: str,
    threshold: float = 0.3,
) -> dict:
    """
    使用文本直接分割图像中的对象

    SAM3 新特性：无需先检测，直接用文本概念进行分割。
    如果 SAM3 不可用，会自动回退到 YOLO-World + SAM2 组合。

    Args:
        image_path: 图像文件的绝对路径
        prompt: 要分割的对象，多个用逗号分隔，如 "person,car"
        threshold: 置信度阈值 (0-1)

    Returns:
        dict: {
            "count": 分割数量,
            "segments": [{
                "label": 标签,
                "bbox": 边界框,
                "score": 置信度,
                "mask_shape": 掩码尺寸,
                "mask_area": 掩码像素面积
            }, ...]
        }
    """
    _ensure_primitives()

    if not os.path.exists(image_path):
        return {"error": f"图像文件不存在: {image_path}"}

    try:
        segments = vision.segment_by_text(image_path, prompt, threshold)

        result_segments = []
        for seg in segments:
            result_seg = {
                "label": seg.get("label", ""),
                "bbox": seg.get("bbox", []),
                "score": seg.get("score", 1.0),
            }
            if seg.get("mask") is not None:
                mask = seg["mask"]
                result_seg["mask_shape"] = list(mask.shape)
                result_seg["mask_area"] = int(mask.sum())
            result_segments.append(result_seg)

        return {
            "count": len(result_segments),
            "segments": result_segments,
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def ocr(image_path: str) -> dict:
    """
    识别图像中的文字

    使用 LLM 视觉能力进行 OCR 文字识别。

    Args:
        image_path: 图像文件的绝对路径

    Returns:
        dict: {
            "count": 识别到的文字块数量,
            "texts": [{
                "text": 识别的文字,
                "bbox": 文字位置 (如果有)
            }, ...]
        }
    """
    _ensure_primitives()

    if not os.path.exists(image_path):
        return {"error": f"图像文件不存在: {image_path}"}

    try:
        img = images.load_image(image_path)
        texts = vision.ocr(img)
        return {
            "count": len(texts),
            "texts": texts,
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def vqa(image_path: str, question: str) -> dict:
    """
    视觉问答 - 回答关于图像的问题

    使用 LLM 视觉能力分析图像并回答问题。

    Args:
        image_path: 图像文件的绝对路径
        question: 关于图像的问题

    Returns:
        dict: {
            "answer": 回答内容
        }
    """
    _ensure_primitives()

    if not os.path.exists(image_path):
        return {"error": f"图像文件不存在: {image_path}"}

    try:
        img = images.load_image(image_path)
        answer = vision.vqa(img, question)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 扩展视觉工具
# ============================================================

@mcp.tool()
def estimate_pose(
    image_path: str,
    threshold: float = 0.5,
) -> dict:
    """
    人体姿态估计 - 检测人体关键点

    检测图像中的人体，返回 17 个 COCO 格式关键点。

    Args:
        image_path: 图像文件的绝对路径
        threshold: 置信度阈值 (0-1)

    Returns:
        dict: {
            "count": 检测到的人数,
            "poses": [{
                "label": "person",
                "bbox": 边界框,
                "score": 置信度,
                "keypoints": [{x, y, confidence}, ...] 17个关键点
            }, ...]
        }
    """
    _ensure_primitives()

    if not os.path.exists(image_path):
        return {"error": f"图像文件不存在: {image_path}"}

    try:
        poses = vision.estimate_pose(image_path, threshold)
        return {
            "count": len(poses),
            "poses": poses,
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def detect_faces(
    image_path: str,
    threshold: float = 0.5,
) -> dict:
    """
    人脸检测

    检测图像中的人脸，返回边界框和关键点。

    Args:
        image_path: 图像文件的绝对路径
        threshold: 置信度阈值 (0-1)

    Returns:
        dict: {
            "count": 检测到的人脸数量,
            "faces": [{
                "label": "face",
                "bbox": 边界框,
                "score": 置信度,
                "landmarks": 5个人脸关键点 (眼、鼻、嘴角)
            }, ...]
        }
    """
    _ensure_primitives()

    if not os.path.exists(image_path):
        return {"error": f"图像文件不存在: {image_path}"}

    try:
        faces = vision.detect_faces(image_path, threshold)
        return {
            "count": len(faces),
            "faces": faces,
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def classify_image(
    image_path: str,
    top_k: int = 5,
) -> dict:
    """
    图像分类

    对图像进行分类，返回 top-k 预测结果。

    Args:
        image_path: 图像文件的绝对路径
        top_k: 返回前 k 个预测结果

    Returns:
        dict: {
            "classifications": [{
                "label": 类别标签,
                "score": 置信度
            }, ...]
        }
    """
    _ensure_primitives()

    if not os.path.exists(image_path):
        return {"error": f"图像文件不存在: {image_path}"}

    try:
        classifications = vision.classify_image(image_path, top_k)
        return {"classifications": classifications}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def compare_images(
    image1_path: str,
    image2_path: str,
) -> dict:
    """
    比较两张图像的相似度

    使用 CLIP 模型计算图像特征，返回余弦相似度。

    Args:
        image1_path: 第一张图像的绝对路径
        image2_path: 第二张图像的绝对路径

    Returns:
        dict: {
            "similarity": 相似度分数 (0-1，越高越相似)
        }
    """
    _ensure_primitives()

    if not os.path.exists(image1_path):
        return {"error": f"图像文件不存在: {image1_path}"}
    if not os.path.exists(image2_path):
        return {"error": f"图像文件不存在: {image2_path}"}

    try:
        similarity = vision.compare_images(image1_path, image2_path)
        return {"similarity": similarity}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def remove_background(
    image_path: str,
    output_path: str,
    foreground: str = "foreground object",
) -> dict:
    """
    移除图像背景

    使用 SAM 分割前景对象，生成透明背景的 PNG 图像。

    Args:
        image_path: 输入图像的绝对路径
        output_path: 输出 PNG 图像的绝对路径
        foreground: 前景对象描述，如 "person", "product"

    Returns:
        dict: {
            "output_path": 输出文件路径,
            "width": 图像宽度,
            "height": 图像高度
        }
    """
    _ensure_primitives()

    if not os.path.exists(image_path):
        return {"error": f"图像文件不存在: {image_path}"}

    try:
        result = vision.remove_background(image_path, foreground)

        if result.get("image") is not None:
            # 保存结果图像
            from PIL import Image
            img = Image.fromarray(result["image"])
            img.save(output_path, "PNG")

            return {
                "output_path": output_path,
                "width": result.get("width", 0),
                "height": result.get("height", 0),
            }
        else:
            return {"error": "背景移除失败，未生成结果图像"}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 图像辅助工具
# ============================================================

@mcp.tool()
def get_image_info(image_path: str) -> dict:
    """
    获取图像基本信息

    Args:
        image_path: 图像文件的绝对路径

    Returns:
        dict: {
            "width": 宽度,
            "height": 高度,
            "channels": 通道数,
            "format": 格式,
            "file_size": 文件大小 (字节)
        }
    """
    _ensure_primitives()

    if not os.path.exists(image_path):
        return {"error": f"图像文件不存在: {image_path}"}

    try:
        info = images.get_image_info(image_path)
        info["file_size"] = os.path.getsize(image_path)
        return info
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def filter_detections(
    detections_json: str,
    min_score: Optional[float] = None,
    labels: Optional[str] = None,
) -> dict:
    """
    过滤检测结果

    Args:
        detections_json: 检测结果的 JSON 字符串
        min_score: 最小置信度阈值，低于此值的结果会被过滤
        labels: 要保留的标签，多个用逗号分隔，如 "person,car"

    Returns:
        dict: {
            "count": 过滤后数量,
            "detections": 过滤后的检测结果
        }
    """
    _ensure_primitives()

    try:
        data = json.loads(detections_json)
        if isinstance(data, dict) and "detections" in data:
            detections = data["detections"]
        else:
            detections = data

        # 按分数过滤
        if min_score is not None:
            detections = vision.filter_by_score(detections, min_score)

        # 按标签过滤
        if labels is not None:
            label_list = [l.strip() for l in labels.split(",")]
            detections = vision.filter_by_label(detections, label_list)

        return {
            "count": len(detections),
            "detections": detections,
        }
    except json.JSONDecodeError as e:
        return {"error": f"JSON 解析错误: {e}"}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 入口
# ============================================================

def main():
    """MCP Server 入口"""
    mcp.run()


if __name__ == "__main__":
    main()
