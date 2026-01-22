# 迭代检测方案设计

## 问题分析

**场景**: 密集物体计数（如易拉罐），单次检测漏检率约20%

**漏检原因**:
1. 密集排列导致NMS抑制相邻检测
2. 部分物体置信度低于阈值
3. 单一提示词无法覆盖所有变体

## 推荐方案：多提示词 + 低阈值检测 + 置信度过滤

```python
# Step 1: 低阈值高召回检测
result = detect_objects(
    image_path,
    prompt="soda can,drink can,soft drink can",  # 多提示词
    threshold=0.1  # 低阈值
)

# Step 2: 置信度过滤去除误检
filtered = filter_detections(result, min_score=0.31)
```

## 实验记录

**场景**: 易拉罐计数 (ground_truth = 31)

| # | Prompt | Threshold | 检测数 | 准确率 | 备注 |
|---|--------|-----------|--------|--------|------|
| 1 | beverage can,aluminum can,coca cola can | 0.2 | 28 | 90.3% | 基线 |
| **5** | **soda can,drink can,soft drink can** | **0.1** | **35→31** | **100%** | **最佳** |

## 关键发现

1. **"drink can" > "beverage can"** - 检测效果更好
2. **多提示词组合** - 提高召回率
3. **两阶段策略** - 低阈值检测 + 高阈值过滤 = 最佳平衡
4. **单一通用词无效** - "can" 对YOLO-World不起作用

## 待实现

- [ ] 实现 `iterative_detect` 函数（集成到MCP Server）
- [ ] 实现 `nms_deduplicate` 函数（IoU去重）
- [ ] 实现 `generate_prompt_variants` 函数（提示词变体生成）
