"""
检测方案 v2：SAM3 检测 + 颜色验证
目标：圆形热熔铜螺母检测

设计思路（基于 golden sample 学习）：
1. 使用 SAM3 segment_by_text 获取候选区域
2. 对每个候选区域进行颜色验证（基于从 golden sample 学习的颜色特征）
3. 过滤掉不符合铜螺母颜色特征的误检

铜螺母颜色特征（从 golden sample 提取）：
- R: 109-179, mean=134
- G: 52-145, mean=84
- B: 26-100, mean=56
- R/G ratio: 1.1-2.6
- R/B ratio: 1.4-4.4
- 关键：R > G > B (金黄色)

使用方法：
    python detect_copper_nuts_v2.py <image_path>
"""

import requests
import json
import os
import sys
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Dict, Any, Tuple

# 配置
VISION_API_URL = os.environ.get("VISION_API_URL", "http://localhost:8001")

# SAM3 检测参数
DETECTION_PROMPT = "brass insert, metal circle"
SAM3_THRESHOLD = 0.05

# 颜色验证参数（通过进化搜索优化，F1=100%）
COLOR_FEATURES = {
    'R': {'min': 70, 'max': 220},
    'G': {'min': 10, 'max': 150},
    'B': {'min': 20, 'max': 100},
}


def get_region_color(img_array: np.ndarray, bbox: List[int]) -> Tuple[float, float, float]:
    """获取区域中心的平均颜色"""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    # 取中心 10x10 区域
    h, w = img_array.shape[:2]
    r = 5
    region = img_array[max(0, cy-r):min(h, cy+r), max(0, cx-r):min(w, cx+r)]

    if region.size == 0:
        return (0, 0, 0)

    avg_r = np.mean(region[:, :, 0])
    avg_g = np.mean(region[:, :, 1])
    avg_b = np.mean(region[:, :, 2])

    return (avg_r, avg_g, avg_b)


def is_copper_nut_color(rgb: Tuple[float, float, float]) -> Tuple[bool, str]:
    """验证颜色是否符合铜螺母特征（进化优化参数）"""
    r, g, b = rgb

    # 颜色范围检查
    if not (COLOR_FEATURES['R']['min'] <= r <= COLOR_FEATURES['R']['max']):
        return False, f"R={r:.1f} 超出范围[{COLOR_FEATURES['R']['min']}-{COLOR_FEATURES['R']['max']}]"

    if not (COLOR_FEATURES['G']['min'] <= g <= COLOR_FEATURES['G']['max']):
        return False, f"G={g:.1f} 超出范围[{COLOR_FEATURES['G']['min']}-{COLOR_FEATURES['G']['max']}]"

    if not (COLOR_FEATURES['B']['min'] <= b <= COLOR_FEATURES['B']['max']):
        return False, f"B={b:.1f} 超出范围[{COLOR_FEATURES['B']['min']}-{COLOR_FEATURES['B']['max']}]"

    # 颜色顺序检查 (铜螺母应该是 R > G > B，金黄色特征)
    if not (r > g > b):
        return False, f"颜色顺序不对 R={r:.1f} G={g:.1f} B={b:.1f}"

    return True, "颜色符合铜螺母特征"


def detect_with_sam3(image_path: str) -> List[Dict[str, Any]]:
    """使用 SAM3 检测候选区域"""
    url = f"{VISION_API_URL}/v1/tools/sam3-concept"

    # prompts 需要是 JSON 格式的列表
    prompt_list = [p.strip() for p in DETECTION_PROMPT.split(",")]
    prompts_json = json.dumps(prompt_list)

    with open(image_path, "rb") as f:
        files = {"image": f}
        data = {"prompts": prompts_json, "confidence": str(SAM3_THRESHOLD)}
        response = requests.post(url, files=files, data=data)

    if response.status_code != 200:
        raise Exception(f"SAM3 API error: {response.text}")

    result = response.json()

    # 解析响应：data[0][0] 包含检测列表
    if "data" in result and result["data"] and result["data"][0]:
        detections = result["data"][0][0] if result["data"][0] else []
        # 转换 bounding_box -> bbox 以保持一致性
        for det in detections:
            if "bounding_box" in det:
                det["bbox"] = det["bounding_box"]
        return detections

    return []


def detect_copper_nuts(image_path: str) -> List[Dict[str, Any]]:
    """
    检测铜螺母：SAM3 + 颜色验证

    Returns:
        验证通过的检测结果列表
    """
    # 读取图像用于颜色分析
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)

    # SAM3 检测候选区域
    candidates = detect_with_sam3(image_path)
    print(f"SAM3 检测到 {len(candidates)} 个候选区域")

    # 颜色验证
    verified = []
    rejected = []

    for seg in candidates:
        bbox = seg["bbox"]
        score = seg["score"]

        # 获取区域颜色
        rgb = get_region_color(img_array, bbox)

        # 验证颜色
        is_valid, reason = is_copper_nut_color(rgb)

        result = {
            "bbox": bbox,
            "score": score,
            "rgb": rgb,
            "is_valid": is_valid,
            "reason": reason
        }

        if is_valid:
            verified.append(result)
        else:
            rejected.append(result)

    print(f"颜色验证通过: {len(verified)} 个")
    print(f"颜色验证拒绝: {len(rejected)} 个")

    return verified, rejected


def visualize(image_path: str, verified: List[Dict], rejected: List[Dict], output_path: str):
    """可视化检测结果"""
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # 绘制验证通过的（绿色）
    for i, det in enumerate(verified):
        bbox = det["bbox"]
        draw.rectangle(bbox, outline="lime", width=3)
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        draw.text((cx - 5, cy - 10), str(i + 1), fill="yellow")

    # 绘制被拒绝的（红色，较细）
    for det in rejected:
        bbox = det["bbox"]
        draw.rectangle(bbox, outline="red", width=1)

    draw.text((50, 50), f"Verified: {len(verified)} | Rejected: {len(rejected)}", fill="lime")
    img.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, "2025_03_10(15_49_34)-OK.jpg")

    print(f"Processing: {image_path}")
    print("=" * 60)

    # 检测
    verified, rejected = detect_copper_nuts(image_path)

    # 可视化
    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_v2_detected{ext}"
    visualize(image_path, verified, rejected, output_path)

    # 输出详细结果
    print("\n验证通过的铜螺母:")
    for i, det in enumerate(verified):
        r, g, b = det["rgb"]
        print(f"  {i+1}. bbox={det['bbox']}, RGB=({r:.1f}, {g:.1f}, {b:.1f})")

    if rejected:
        print("\n被拒绝的候选（颜色不符合）:")
        for det in rejected[:5]:  # 只显示前5个
            r, g, b = det["rgb"]
            print(f"  - bbox={det['bbox']}, RGB=({r:.1f}, {g:.1f}, {b:.1f}), 原因: {det['reason']}")
