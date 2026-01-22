"""
铜螺母检测方案 v3 (最终版)
进化优化后的检测方案，包含自动去重和数量评估

检测流程：
1. SAM3 segment_by_text 获取候选区域
2. 颜色验证过滤（进化优化参数）
3. 去重（距离 < 30 像素视为重复）
4. 数量评估（与 golden sample 的 19 个对比）

使用方法：
    python detect_copper_nuts_final.py <image_path>
"""

import requests
import json
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Tuple

# 配置
VISION_API_URL = os.environ.get("VISION_API_URL", "http://localhost:8001")

# 检测参数（进化优化）
DETECTION_PROMPT = "brass insert, metal circle"
SAM3_THRESHOLD = 0.05

# 颜色验证参数（进化优化，F1=100%）
COLOR_R_RANGE = (70, 220)
COLOR_G_RANGE = (10, 150)
COLOR_B_RANGE = (20, 100)

# 去重距离阈值
DEDUP_DISTANCE = 30

# Golden sample 铜螺母数量
EXPECTED_COUNT = 19


def bbox_center(bbox: List[int]) -> Tuple[int, int]:
    """获取 bbox 中心点"""
    return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)


def distance(c1: Tuple[int, int], c2: Tuple[int, int]) -> float:
    """计算两点距离"""
    return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5


def get_region_color(img_array: np.ndarray, bbox: List[int]) -> Tuple[float, float, float]:
    """获取区域中心颜色"""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    h, w = img_array.shape[:2]
    r = 5
    region = img_array[max(0, cy-r):min(h, cy+r), max(0, cx-r):min(w, cx+r)]
    if region.size == 0:
        return (0, 0, 0)
    return (np.mean(region[:, :, 0]), np.mean(region[:, :, 1]), np.mean(region[:, :, 2]))


def is_copper_nut_color(rgb: Tuple[float, float, float]) -> Tuple[bool, str]:
    """验证颜色是否符合铜螺母特征"""
    r, g, b = rgb

    if not (COLOR_R_RANGE[0] <= r <= COLOR_R_RANGE[1]):
        return False, f"R={r:.1f} 超出范围"
    if not (COLOR_G_RANGE[0] <= g <= COLOR_G_RANGE[1]):
        return False, f"G={g:.1f} 超出范围"
    if not (COLOR_B_RANGE[0] <= b <= COLOR_B_RANGE[1]):
        return False, f"B={b:.1f} 超出范围"
    if not (r > g > b):
        return False, f"颜色顺序不对"

    return True, "OK"


def detect_with_sam3(image_path: str) -> List[Dict[str, Any]]:
    """使用 SAM3 检测候选区域"""
    url = f"{VISION_API_URL}/v1/tools/sam3-concept"

    prompt_list = [p.strip() for p in DETECTION_PROMPT.split(",")]
    prompts_json = json.dumps(prompt_list)

    with open(image_path, "rb") as f:
        files = {"image": f}
        data = {"prompts": prompts_json, "confidence": str(SAM3_THRESHOLD)}
        response = requests.post(url, files=files, data=data)

    if response.status_code != 200:
        raise Exception(f"SAM3 API error: {response.text}")

    result = response.json()

    if "data" in result and result["data"] and result["data"][0]:
        detections = result["data"][0][0] if result["data"][0] else []
        for det in detections:
            if "bounding_box" in det:
                det["bbox"] = det["bounding_box"]
        return detections

    return []


def deduplicate(detections: List[Dict], threshold: float = DEDUP_DISTANCE) -> List[Dict]:
    """去除重复检测"""
    unique = []
    for det in detections:
        center = bbox_center(det["bbox"])
        is_dup = False
        for u in unique:
            u_center = bbox_center(u["bbox"])
            if distance(center, u_center) < threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(det)
    return unique


def detect_copper_nuts(image_path: str) -> Dict:
    """
    检测铜螺母（完整流程）

    Returns:
        检测结果字典
    """
    # 读取图像
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)

    # SAM3 检测
    candidates = detect_with_sam3(image_path)

    # 颜色验证
    verified = []
    rejected = []

    for seg in candidates:
        bbox = seg["bbox"]
        rgb = get_region_color(img_array, bbox)
        is_valid, reason = is_copper_nut_color(rgb)

        result = {
            "bbox": bbox,
            "score": seg.get("score", 0),
            "rgb": rgb,
            "is_valid": is_valid,
            "reason": reason
        }

        if is_valid:
            verified.append(result)
        else:
            rejected.append(result)

    # 去重
    unique = deduplicate(verified)
    duplicates_removed = len(verified) - len(unique)

    # 评估
    count = len(unique)
    status = "OK" if count == EXPECTED_COUNT else "NG"
    diff = count - EXPECTED_COUNT

    return {
        "candidates": len(candidates),
        "verified": len(verified),
        "duplicates_removed": duplicates_removed,
        "unique_count": count,
        "expected_count": EXPECTED_COUNT,
        "difference": diff,
        "status": status,
        "detections": unique,
        "rejected": rejected
    }


def visualize(image_path: str, result: Dict, output_path: str):
    """可视化检测结果"""
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # 绘制检测结果（绿色）
    for i, det in enumerate(result["detections"]):
        bbox = det["bbox"]
        draw.rectangle(bbox, outline="lime", width=3)
        cx, cy = bbox_center(bbox)
        draw.text((cx - 5, cy - 10), str(i + 1), fill="yellow")

    # 绘制被拒绝的（红色）
    for det in result["rejected"]:
        bbox = det["bbox"]
        draw.rectangle(bbox, outline="red", width=1)

    # 状态信息
    status = result["status"]
    count = result["unique_count"]
    expected = result["expected_count"]

    if status == "OK":
        status_text = f"OK: {count}/{expected} copper nuts"
        color = "lime"
    else:
        diff = result["difference"]
        if diff < 0:
            status_text = f"NG: {count}/{expected} (缺少 {-diff} 个)"
        else:
            status_text = f"NG: {count}/{expected} (多出 {diff} 个)"
        color = "red"

    draw.text((50, 50), status_text, fill=color)

    img.save(output_path)
    return output_path


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, "golden sample.jpg")

    print(f"Processing: {image_path}")
    print("=" * 60)

    # 检测
    result = detect_copper_nuts(image_path)

    # 输出结果
    print(f"SAM3 候选: {result['candidates']} 个")
    print(f"颜色验证通过: {result['verified']} 个")
    print(f"去重删除: {result['duplicates_removed']} 个")
    print(f"最终检测: {result['unique_count']} 个")
    print(f"期望数量: {result['expected_count']} 个")
    print("=" * 60)

    if result["status"] == "OK":
        print(f"✓ 检测结果: OK")
    else:
        diff = result["difference"]
        if diff < 0:
            print(f"✗ 检测结果: NG (缺少 {-diff} 个铜螺母)")
        else:
            print(f"✗ 检测结果: NG (多出 {diff} 个)")

    # 可视化
    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_final_detected{ext}"
    visualize(image_path, result, output_path)
    print(f"\nSaved: {output_path}")
