"""
进化式检测方案优化器
自动探索最优检测参数，无需人工介入

进化策略：
1. 定义参数搜索空间
2. 运行检测 + 自动评估
3. 记录所有方案结果
4. 基于评估结果生成改进方案
5. 迭代直到找到最优方案
"""

import requests
import json
import os
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from auto_evaluator import evaluate_detections, print_evaluation, GROUND_TRUTH

# 配置
VISION_API_URL = os.environ.get("VISION_API_URL", "http://localhost:8001")
GOLDEN_SAMPLE = "golden sample.jpg"


@dataclass
class DetectionConfig:
    """检测配置"""
    prompt: str
    sam3_threshold: float
    use_color_filter: bool = True
    color_r_min: int = 100
    color_r_max: int = 200
    color_g_min: int = 40
    color_g_max: int = 160
    color_b_min: int = 20
    color_b_max: int = 110
    require_r_gt_g_gt_b: bool = True


def detect_with_sam3(image_path: str, prompt: str, threshold: float) -> List[Dict]:
    """调用 SAM3 检测"""
    url = f"{VISION_API_URL}/v1/tools/sam3-concept"

    # prompts 需要是 JSON 格式的列表
    prompt_list = [p.strip() for p in prompt.split(",")]
    prompts_json = json.dumps(prompt_list)

    with open(image_path, "rb") as f:
        files = {"image": f}
        data = {"prompts": prompts_json, "confidence": str(threshold)}
        response = requests.post(url, files=files, data=data)

    if response.status_code != 200:
        print(f"SAM3 API error: {response.text}")
        return []

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


def apply_color_filter(img_array: np.ndarray, candidates: List[Dict], config: DetectionConfig) -> List[Dict]:
    """应用颜色过滤"""
    if not config.use_color_filter:
        return candidates

    verified = []
    for seg in candidates:
        bbox = seg["bbox"]
        r, g, b = get_region_color(img_array, bbox)

        # 颜色范围检查
        if not (config.color_r_min <= r <= config.color_r_max):
            continue
        if not (config.color_g_min <= g <= config.color_g_max):
            continue
        if not (config.color_b_min <= b <= config.color_b_max):
            continue

        # R > G > B 检查
        if config.require_r_gt_g_gt_b and not (r > g > b):
            continue

        verified.append(seg)

    return verified


def run_detection(image_path: str, config: DetectionConfig) -> Tuple[List[Dict], Dict]:
    """运行检测并评估"""
    # SAM3 检测
    candidates = detect_with_sam3(image_path, config.prompt, config.sam3_threshold)

    if not candidates:
        return [], {"total_detections": 0, "f1": 0, "precision": 0, "recall": 0,
                    "tp": 0, "fp": 0, "fn": len(GROUND_TRUTH), "perfect": False}

    # 颜色过滤
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    verified = apply_color_filter(img_array, candidates, config)

    # 评估
    evaluation = evaluate_detections(verified)

    return verified, evaluation


class EvolutionOptimizer:
    """进化优化器"""

    def __init__(self, image_path: str = GOLDEN_SAMPLE):
        self.image_path = image_path
        self.history = []  # 方案历史记录
        self.best_config = None
        self.best_f1 = 0

    def explore(self, config: DetectionConfig) -> Dict:
        """探索一个配置"""
        detections, evaluation = run_detection(self.image_path, config)

        record = {
            "id": len(self.history) + 1,
            "config": {
                "prompt": config.prompt,
                "sam3_threshold": config.sam3_threshold,
                "use_color_filter": config.use_color_filter,
                "color_params": f"R[{config.color_r_min}-{config.color_r_max}] "
                               f"G[{config.color_g_min}-{config.color_g_max}] "
                               f"B[{config.color_b_min}-{config.color_b_max}]"
            },
            "results": {
                "candidates": len(detections) if not config.use_color_filter else "N/A",
                "verified": len(detections),
                "precision": evaluation["precision"],
                "recall": evaluation["recall"],
                "f1": evaluation["f1"],
                "perfect": evaluation["perfect"]
            }
        }

        self.history.append(record)

        # 更新最佳配置
        if evaluation["f1"] > self.best_f1:
            self.best_f1 = evaluation["f1"]
            self.best_config = config
            print(f"  ★ 新最佳方案! F1={evaluation['f1']:.2%}")

        return record

    def print_history(self):
        """打印探索历史"""
        print("\n" + "=" * 80)
        print("方案探索历史")
        print("=" * 80)
        print(f"{'#':<3} {'Prompt':<35} {'Thr':<6} {'Color':<6} {'Det':<5} {'P':<8} {'R':<8} {'F1':<8}")
        print("-" * 80)
        for r in self.history:
            c = r["config"]
            res = r["results"]
            color = "Yes" if "color_params" in c else "No"
            prompt_short = c["prompt"][:32] + "..." if len(c["prompt"]) > 35 else c["prompt"]
            print(f"{r['id']:<3} {prompt_short:<35} {c['sam3_threshold']:<6.2f} {color:<6} "
                  f"{res['verified']:<5} {res['precision']:<8.2%} {res['recall']:<8.2%} {res['f1']:<8.2%}")
        print("=" * 80)

    def run_evolution(self):
        """运行进化探索"""
        print("开始进化探索...")
        print(f"测试图片: {self.image_path}")
        print(f"Ground Truth: {len(GROUND_TRUTH)} 个铜螺母")
        print("=" * 60)

        # ========== 第一轮：Prompt 探索 ==========
        print("\n[Round 1] Prompt 多样性探索")
        prompts = [
            "brass insert, metal circle",
            "copper nut, brass nut",
            "golden circle, metal insert",
            "brass insert",
            "copper insert, brass circle",
            "metal nut, brass fitting",
        ]

        for prompt in prompts:
            config = DetectionConfig(
                prompt=prompt,
                sam3_threshold=0.05,
                use_color_filter=True
            )
            print(f"\n  尝试: prompt='{prompt}'")
            self.explore(config)

        # ========== 第二轮：阈值探索 ==========
        print("\n[Round 2] SAM3 阈值探索 (使用当前最佳 prompt)")
        best_prompt = self.best_config.prompt if self.best_config else prompts[0]

        for threshold in [0.02, 0.03, 0.05, 0.08, 0.1]:
            config = DetectionConfig(
                prompt=best_prompt,
                sam3_threshold=threshold,
                use_color_filter=True
            )
            print(f"\n  尝试: threshold={threshold}")
            self.explore(config)

        # ========== 第三轮：颜色参数探索 ==========
        print("\n[Round 3] 颜色验证参数探索")
        best_threshold = self.best_config.sam3_threshold if self.best_config else 0.05

        # 放宽颜色范围
        color_configs = [
            {"r": (90, 210), "g": (30, 170), "b": (15, 120)},  # 放宽
            {"r": (100, 200), "g": (40, 160), "b": (20, 110)},  # 标准
            {"r": (110, 190), "g": (50, 150), "b": (25, 100)},  # 收紧
            {"r": (80, 220), "g": (20, 180), "b": (10, 130)},   # 大幅放宽
        ]

        for cc in color_configs:
            config = DetectionConfig(
                prompt=best_prompt,
                sam3_threshold=best_threshold,
                use_color_filter=True,
                color_r_min=cc["r"][0], color_r_max=cc["r"][1],
                color_g_min=cc["g"][0], color_g_max=cc["g"][1],
                color_b_min=cc["b"][0], color_b_max=cc["b"][1]
            )
            print(f"\n  尝试: 颜色范围 R{cc['r']} G{cc['g']} B{cc['b']}")
            self.explore(config)

        # ========== 第四轮：不使用颜色过滤 ==========
        print("\n[Round 4] 纯 SAM3 检测 (不使用颜色过滤)")
        for threshold in [0.3, 0.35, 0.4, 0.41, 0.45, 0.5]:
            config = DetectionConfig(
                prompt=best_prompt,
                sam3_threshold=threshold,
                use_color_filter=False
            )
            print(f"\n  尝试: 纯 SAM3, threshold={threshold}")
            self.explore(config)

        # ========== 输出结果 ==========
        self.print_history()

        print("\n" + "=" * 60)
        print("最佳方案")
        print("=" * 60)
        if self.best_config:
            print(f"Prompt: {self.best_config.prompt}")
            print(f"SAM3 Threshold: {self.best_config.sam3_threshold}")
            print(f"Color Filter: {self.best_config.use_color_filter}")
            if self.best_config.use_color_filter:
                print(f"Color Params: R[{self.best_config.color_r_min}-{self.best_config.color_r_max}] "
                      f"G[{self.best_config.color_g_min}-{self.best_config.color_g_max}] "
                      f"B[{self.best_config.color_b_min}-{self.best_config.color_b_max}]")
            print(f"Best F1: {self.best_f1:.2%}")

        return self.best_config, self.best_f1


if __name__ == "__main__":
    optimizer = EvolutionOptimizer()
    best_config, best_f1 = optimizer.run_evolution()
