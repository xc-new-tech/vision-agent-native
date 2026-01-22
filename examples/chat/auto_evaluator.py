"""
自动评估框架：铜螺母检测
基于 golden sample 的 ground truth 自动评估检测结果

评估指标：
- Precision: 检测结果中正确的比例
- Recall: ground truth 被检测到的比例
- F1 Score: Precision 和 Recall 的调和平均

匹配规则：检测框中心距离 ground truth < 阈值则认为匹配
"""

import json
import math
from typing import List, Dict, Tuple, Any

# Ground Truth: 从 golden sample 提取的 19 个铜螺母位置
GROUND_TRUTH = [
    [576, 995], [625, 986], [815, 973], [850, 974], [1166, 961],
    [1397, 986], [1523, 956], [1654, 929], [933, 1134], [1127, 1126],
    [1542, 1385], [1641, 1385], [602, 1475], [649, 1475], [744, 1468],
    [842, 1462], [874, 1475], [1014, 1469], [1397, 1448]
]

# 匹配距离阈值（像素）
MATCH_DISTANCE_THRESHOLD = 50


def distance(p1: List[int], p2: List[int]) -> float:
    """计算两点间距离"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def bbox_center(bbox: List[int]) -> List[int]:
    """获取 bbox 中心点"""
    return [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]


def evaluate_detections(detections: List[Dict],
                        ground_truth: List[List[int]] = GROUND_TRUTH,
                        match_threshold: float = MATCH_DISTANCE_THRESHOLD) -> Dict:
    """
    评估检测结果

    Args:
        detections: 检测结果列表，每个包含 bbox 字段
        ground_truth: ground truth 位置列表
        match_threshold: 匹配距离阈值

    Returns:
        评估结果字典
    """
    # 提取检测中心点
    det_centers = [bbox_center(d["bbox"]) for d in detections]

    # 匹配统计
    gt_matched = [False] * len(ground_truth)
    det_matched = [False] * len(detections)
    matches = []  # (gt_idx, det_idx, distance)

    # 贪婪匹配：为每个 ground truth 找最近的未匹配检测
    for gt_idx, gt_pos in enumerate(ground_truth):
        best_det_idx = -1
        best_dist = float('inf')

        for det_idx, det_center in enumerate(det_centers):
            if det_matched[det_idx]:
                continue
            dist = distance(gt_pos, det_center)
            if dist < match_threshold and dist < best_dist:
                best_dist = dist
                best_det_idx = det_idx

        if best_det_idx >= 0:
            gt_matched[gt_idx] = True
            det_matched[best_det_idx] = True
            matches.append((gt_idx, best_det_idx, best_dist))

    # 计算指标
    tp = sum(gt_matched)  # True Positives: 匹配的 ground truth
    fp = len(detections) - sum(det_matched)  # False Positives: 未匹配的检测
    fn = len(ground_truth) - tp  # False Negatives: 未检测到的 ground truth

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # 详细结果
    missed_gt = [i for i, matched in enumerate(gt_matched) if not matched]
    false_detections = [i for i, matched in enumerate(det_matched) if not matched]

    return {
        "total_gt": len(ground_truth),
        "total_detections": len(detections),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "missed_gt_indices": missed_gt,
        "false_detection_indices": false_detections,
        "matches": matches,
        "perfect": (tp == len(ground_truth) and fp == 0)
    }


def print_evaluation(result: Dict, verbose: bool = True):
    """打印评估结果"""
    print(f"\n{'='*60}")
    print(f"检测评估结果")
    print(f"{'='*60}")
    print(f"Ground Truth: {result['total_gt']} 个铜螺母")
    print(f"检测数量: {result['total_detections']} 个")
    print(f"{'='*60}")
    print(f"True Positives (正确检测): {result['tp']}")
    print(f"False Positives (误检): {result['fp']}")
    print(f"False Negatives (漏检): {result['fn']}")
    print(f"{'='*60}")
    print(f"Precision (精确率): {result['precision']:.2%}")
    print(f"Recall (召回率): {result['recall']:.2%}")
    print(f"F1 Score: {result['f1']:.2%}")
    print(f"{'='*60}")

    if result['perfect']:
        print("✓ 完美检测！所有铜螺母都被正确检测，无误检")
    else:
        if result['missed_gt_indices']:
            print(f"漏检的 Ground Truth: {result['missed_gt_indices']}")
            if verbose:
                for idx in result['missed_gt_indices']:
                    print(f"  - GT #{idx+1}: {GROUND_TRUTH[idx]}")
        if result['false_detection_indices']:
            print(f"误检数量: {len(result['false_detection_indices'])}")


def quick_evaluate(detections: List[Dict]) -> Tuple[float, float, float]:
    """
    快速评估，返回 (precision, recall, f1)
    用于进化迭代时快速获取评分
    """
    result = evaluate_detections(detections)
    return result['precision'], result['recall'], result['f1']


if __name__ == "__main__":
    # 测试：模拟一些检测结果
    test_detections = [
        {"bbox": [560, 980, 590, 1010]},  # 应该匹配 GT #1
        {"bbox": [610, 970, 640, 1000]},  # 应该匹配 GT #2
        {"bbox": [100, 100, 130, 130]},   # 误检
    ]

    result = evaluate_detections(test_detections)
    print_evaluation(result)
