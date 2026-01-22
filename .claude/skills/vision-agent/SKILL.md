---
name: vision-agent
description: 视觉AI助手 - 用于图像检测、分割、OCR等视觉任务。当用户需要分析图像、检测对象、识别文字、比较图像时使用此 Skill。
---

# Vision Agent Skill

视觉检测方案开发助手。核心目标：**根据任务找到准确、稳定的检测方案，并固化为可复用的 Python 代码**。

采用**进化式探索**方法：基于 golden sample 建立自动评估体系，然后自主进化迭代找到最优方案。

## 工作流程概览

```
┌─────────────────────────────────────────────────────────────┐
│  第一步：提取 Ground Truth（需用户确认）                      │
│  - 从 golden sample 提取目标数量/位置                        │
│  - 分析目标特征（颜色、形状等）                               │
│  - 用户确认 ground truth 是否正确                            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  第二步：构建自动评估器（需用户确认）                        │
│  - 基于 ground truth 设计评估指标                            │
│  - 实现自动 OK/NG 判断逻辑                                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  第三步：自主进化迭代（无需用户确认）                          │
│  - 探索 prompt、阈值、过滤参数                               │
│  - 自动评估每个方案                                          │
│  - 记录所有尝试，找到 F1=100% 的最优方案                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  第四步：输出最终检测脚本                                     │
│  - 包含最优参数                                              │
│  - 包含去重逻辑                                              │
│  - 包含自动 OK/NG 判断                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 第一步：提取 Ground Truth（⚠️ 需用户确认）

这是**唯一需要用户确认**的步骤。

### 1.1 从 Golden Sample 提取目标信息

如果用户提供了标注好的 golden sample（如绿色圆框标注），自动提取：

```python
# 提取标注位置（示例：检测绿色圆框）
def extract_ground_truth(golden_sample_path):
    # 1. 检测标注（如绿色圆框）
    # 2. 提取每个标注的中心位置
    # 3. 返回 ground truth 列表
    return positions  # [(x1, y1), (x2, y2), ...]
```

### 1.2 分析目标特征

在 ground truth 位置采样，学习目标特征：

```python
# 颜色特征分析
for pos in ground_truth_positions:
    rgb = sample_color_at(image, pos)
    # 统计 R, G, B 范围
    # 分析颜色比率（如 R > G > B 表示金黄色）
```

### 1.3 用户确认

向用户展示提取结果，**必须得到用户确认**：

```
提取的 Ground Truth:
- 数量: 19 个目标
- 位置: [列表]
- 颜色特征: R[70-220], G[10-150], B[20-100], R > G > B

请确认以上信息是否正确？
```

**只有用户确认后，才能进入下一步。**

---

## 第二步：构建自动评估器（自动执行）

基于确认的 ground truth，构建自动评估体系。

### 2.1 评估指标

```python
# 基于数量的评估（适用于位置可能变化的场景）
def evaluate_by_count(detections, expected_count):
    count = len(deduplicate(detections))
    status = "OK" if count == expected_count else "NG"
    return {"count": count, "expected": expected_count, "status": status}

# 基于位置的评估（适用于位置固定的场景）
def evaluate_by_position(detections, ground_truth, threshold=50):
    # 匹配检测与 ground truth
    tp = count_matches(detections, ground_truth, threshold)
    fp = len(detections) - tp
    fn = len(ground_truth) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}
```

### 2.2 去重逻辑

检测结果可能有重复，需要去重：

```python
def deduplicate(detections, distance_threshold=30):
    unique = []
    for det in detections:
        center = bbox_center(det["bbox"])
        is_dup = any(distance(center, bbox_center(u["bbox"])) < distance_threshold for u in unique)
        if not is_dup:
            unique.append(det)
    return unique
```

---

## 第三步：自主进化迭代（自动执行，无需用户确认）

### 3.1 定义搜索空间

```python
# Prompt 变体
prompts = [
    "brass insert, metal circle",
    "copper nut, brass nut",
    "golden circle, metal insert",
    # ...
]

# 阈值范围
thresholds = [0.02, 0.03, 0.05, 0.08, 0.1]

# 颜色过滤参数
color_ranges = [
    {"r": (70, 220), "g": (10, 150), "b": (20, 100)},
    {"r": (80, 210), "g": (20, 160), "b": (15, 110)},
    # ...
]
```

### 3.2 自动探索循环

```python
def run_evolution(golden_sample, ground_truth):
    best_f1 = 0
    best_config = None
    history = []

    # Round 1: Prompt 探索
    for prompt in prompts:
        config = {"prompt": prompt, "threshold": 0.05}
        detections = detect(golden_sample, config)
        result = evaluate(detections, ground_truth)
        history.append({"config": config, "result": result})

        if result["f1"] > best_f1:
            best_f1 = result["f1"]
            best_config = config

    # Round 2: 阈值探索（使用最佳 prompt）
    for threshold in thresholds:
        config = {"prompt": best_config["prompt"], "threshold": threshold}
        # ... 同上

    # Round 3: 颜色参数探索
    # ... 继续迭代

    return best_config, history
```

### 3.3 方案记录表

每次探索自动记录：

```
================================================================================
方案探索历史
================================================================================
#   Prompt                              Thr    Color  Det   P        R        F1
--------------------------------------------------------------------------------
1   brass insert, metal circle          0.05   Yes    17    100.00%  89.47%   94.44%
2   copper nut, brass nut               0.05   Yes    0     0.00%    0.00%    0.00%
3   brass insert, metal circle          0.05   Yes    18    100.00%  94.74%   97.30%
4   brass insert, metal circle          0.05   Yes    19    100.00%  100.00%  100.00% ★
================================================================================
```

### 3.4 迭代终止条件

- F1 = 100%（完美检测）
- 或探索了所有参数组合
- 或达到最大迭代次数

---

## 第四步：输出最终检测脚本

找到最优方案后，输出完整的、可独立运行的 Python 检测代码：

```python
"""
检测方案：SAM3 + 颜色验证
进化历史：经过 N 轮迭代，探索了 M 种方案
最终准确率：F1=100%
"""

import requests
import json
import numpy as np
from PIL import Image, ImageDraw

# 配置
VISION_API_URL = "http://localhost:8001"

# 检测参数（进化优化得出）
DETECTION_PROMPT = "brass insert, metal circle"
SAM3_THRESHOLD = 0.05

# 颜色验证参数（进化优化得出）
COLOR_R_RANGE = (70, 220)
COLOR_G_RANGE = (10, 150)
COLOR_B_RANGE = (20, 100)

# 期望数量（从 golden sample 提取）
EXPECTED_COUNT = 19

# 去重距离阈值
DEDUP_DISTANCE = 30


def detect(image_path: str) -> dict:
    """执行检测，返回结果"""
    # 1. SAM3 检测候选
    candidates = detect_with_sam3(image_path)

    # 2. 颜色验证
    verified = apply_color_filter(candidates)

    # 3. 去重
    unique = deduplicate(verified)

    # 4. 评估
    count = len(unique)
    status = "OK" if count == EXPECTED_COUNT else "NG"

    return {
        "count": count,
        "expected": EXPECTED_COUNT,
        "status": status,
        "detections": unique
    }


if __name__ == "__main__":
    result = detect("input.jpg")
    print(f"检测结果: {result['status']}")
    print(f"检测数量: {result['count']}/{result['expected']}")
```

---

## 可用的 MCP 工具

- `mcp__vision-agent__detect_objects` - YOLO-World 对象检测
- `mcp__vision-agent__segment_by_text` - SAM3 文本分割
- `mcp__vision-agent__segment_objects` - SAM2 基于bbox分割
- `mcp__vision-agent__ocr` - 文字识别
- `mcp__vision-agent__vqa` - 视觉问答
- `mcp__vision-agent__get_image_info` - 获取图像信息
- `mcp__vision-agent__compare_images` - 图像相似度比较
- `mcp__vision-agent__classify_image` - 图像分类
- `mcp__vision-agent__detect_faces` - 人脸检测
- `mcp__vision-agent__estimate_pose` - 姿态估计

---

## 关键原则

1. **Golden Sample 优先** - 必须先从标注样本提取 ground truth
2. **用户确认仅一次** - 只在提取 ground truth 时需要用户确认
3. **自主进化** - 参数探索过程完全自动，无需人工介入
4. **自动评估** - 基于 ground truth 自动计算准确率
5. **去重处理** - 检测结果必须去重后再评估
6. **代码输出** - 最终输出包含自动 OK/NG 判断的完整脚本

---

## 案例：铜螺母检测

### 输入
- Golden sample: 带绿色圆框标注的图片（19 个铜螺母）
- 测试图片: 无标注的待检图片

### 工作流程

1. **提取 Ground Truth**（用户确认）
   - 检测绿色圆框 → 提取 19 个位置
   - 分析颜色特征 → R[70-220], G[10-150], B[20-100], R > G > B
   - 用户确认 ✓

2. **构建评估器**（自动）
   - 基于数量评估：检测数 == 19 → OK，否则 NG

3. **自主进化**（自动）
   - 探索 6 种 prompt → 最佳: "brass insert, metal circle"
   - 探索 5 种阈值 → 最佳: 0.05
   - 探索 4 种颜色参数 → 最佳: R[70-220], G[10-150], B[20-100]
   - 最终 F1: 100%

4. **输出脚本**
   - `detect_copper_nuts_final.py`
   - 自动输出 OK/NG 状态

### 测试结果
| 样本 | 检测数 | 期望数 | 状态 |
|------|--------|--------|------|
| Golden sample | 19 | 19 | ✓ OK |
| NG sample | 17 | 19 | ✗ NG |
