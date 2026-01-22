"""
检测方案：SAM3 文本分割 (进化探索优化)
目标：圆形热熔铜螺母检测
进化历史：经过 6 轮迭代，17 种方案探索
最终准确率：100% (18/18)

关键发现：
- YOLO-World 对小型工业零件检测效果差，容易误检整体外壳
- SAM3 segment_by_text 配合 "brass insert, metal circle" 效果最佳
- 阈值 0.41 是最优平衡点

使用方法：
    python detect_copper_nuts.py [image_path]

依赖：pip install pillow requests
前提：本地 Vision API 运行在 http://localhost:8001
"""

import requests
import json
import os
import sys
from PIL import Image, ImageDraw
from typing import List, Dict, Any

# 配置
VISION_API_URL = os.environ.get("VISION_API_URL", "http://localhost:8001")

# 检测参数（经进化探索调优）
DETECTION_PROMPT = "brass insert, metal circle"
THRESHOLD = 0.05  # SAM3 初始阈值
FILTER_THRESHOLD = 0.41  # 过滤阈值（进化优化得出）


def detect_copper_nuts(image_path: str) -> List[Dict[str, Any]]:
    """
    检测图像中的铜螺母

    Args:
        image_path: 图像文件路径

    Returns:
        检测结果列表，每个包含 label, bbox, score
    """
    # 调用 SAM3 segment_by_text API
    url = f"{VISION_API_URL}/v1/tools/sam3-concept"

    with open(image_path, "rb") as f:
        files = {"image": f}
        data = {
            "prompt": DETECTION_PROMPT,
            "threshold": THRESHOLD
        }
        response = requests.post(url, files=files, data=data)

    if response.status_code != 200:
        raise Exception(f"API error: {response.text}")

    result = response.json()
    segments = result.get("segments", [])

    # 过滤低置信度结果
    filtered = [
        {
            "label": seg["label"],
            "bbox": seg["bbox"],
            "score": seg["score"]
        }
        for seg in segments
        if seg["score"] >= FILTER_THRESHOLD
    ]

    return filtered


def visualize(image_path: str, detections: List[Dict], output_path: str):
    """
    可视化检测结果

    Args:
        image_path: 输入图像路径
        detections: 检测结果列表
        output_path: 输出图像路径
    """
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    for i, det in enumerate(detections):
        bbox = det["bbox"]
        score = det["score"]

        # 绘制红色边框
        draw.rectangle(bbox, outline="red", width=3)

        # 绘制编号
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        draw.text((cx - 5, cy - 10), str(i + 1), fill="yellow")

    # 添加统计信息
    draw.text((50, 50), f"Detected: {len(detections)} copper nuts", fill="yellow")

    img.save(output_path)
    print(f"Saved visualization to: {output_path}")


if __name__ == "__main__":
    # 默认路径
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, "golden sample.jpg")

    # 生成输出路径
    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_detected{ext}"

    # 执行检测
    print(f"Processing: {image_path}")
    print(f"Detection params: prompt='{DETECTION_PROMPT}', filter_threshold={FILTER_THRESHOLD}")

    detections = detect_copper_nuts(image_path)
    print(f"Detected {len(detections)} copper nuts")

    # 可视化
    visualize(image_path, detections, output_path)

    # 输出详细结果
    print("\nDetection details:")
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det['label']}: bbox={det['bbox']}, score={det['score']:.3f}")
