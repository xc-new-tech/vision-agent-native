#!/usr/bin/env python3
"""
热熔螺母缺陷检测 - Agent-native 示例

演示如何用原子原语组合来实现缺陷检测。

Agent-native 原则:
- 使用原子原语 (vision_chat, read_file, write_json 等)
- 判断逻辑由 Agent 决定，不硬编码在工具里
- 结果保存到文件，可追溯
- 维护 context.md 状态
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# 确保可以导入本项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from primitives import (
    vision_chat,
    list_dir,
    read_file,
    write_file,
    write_json,
    append_file,
    file_exists,
)


# ============================================================
# 原语组合：缺陷检测流程
# ============================================================

def run_defect_detection(
    test_image: str,
    reference_image: str,
    output_dir: str = "./output",
    context_file: str = "./context.md",
):
    """
    用原语组合实现缺陷检测

    这不是一个"工具"，而是展示 Agent 如何组合原语来完成任务。
    在实际的 agent-native 系统中，这些步骤由 Agent 在循环中执行。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: 记录开始 (append to agent_log)
    log_entry = f"\n## {timestamp} - 缺陷检测\n"
    log_entry += f"- 参考图像: {reference_image}\n"
    log_entry += f"- 待检测图像: {test_image}\n"
    append_file(f"{output_dir}/agent_log.md", log_entry)
    print(f"[1/4] 记录任务开始")

    # Step 2: 使用 vision_chat 分析图像 (核心原语)
    print(f"[2/4] 调用 vision_chat 分析图像...")

    system_prompt = """你是工业视觉检测专家，专门进行热熔螺母缺陷检测。

## 重要概念
- 热熔螺母：嵌入在橙色塑料外壳中的小圆形金属件（黄铜色/金色），中间有螺纹孔
- 绿色圆圈：仅在标准样本上出现，是位置标注，不是实际的螺母
- 待检测样本上没有绿色圆圈，你需要直接观察金属螺母

## 检测方法
1. 先数标准样本上绿色圆圈的数量（约15-20个）
2. 在待检测样本的对应位置，检查是否能看到黄铜色金属螺母
3. 仔细检查外壳表面是否有：划痕、手写标记、污渍、裂纹
4. 如果所有螺母都在且表面完好无瑕，判断为 OK
5. 如果有任何问题（螺母缺失、表面划痕等），判断为 NG

## 输出格式 (JSON)
{"result": "OK"或"NG", "confidence": 0-100, "issues": [...], "details": "..."}"""

    user_prompt = """请仔细对比这两张图片：

【第一张图 - 标准样本】
- 绿色圆圈标注了热熔螺母应该存在的位置
- 数一下绿色圆圈有多少个

【第二张图 - 待检测样本】
- 没有绿色标注
- 检查每个对应位置是否有黄铜色金属螺母
- 检查外壳表面是否有划痕或标记

如果所有螺母都存在且外壳完好，输出 OK；否则输出 NG。
输出 JSON 结果。"""

    # 调用核心原语
    response = vision_chat(
        images=[reference_image, test_image],
        prompt=user_prompt,
        system=system_prompt,
    )
    print(f"[3/4] 收到分析结果")

    # Step 3: 解析结果
    result = {
        "timestamp": timestamp,
        "reference_image": reference_image,
        "test_image": test_image,
        "raw_response": response,
        "result": "UNKNOWN",
        "confidence": 0,
        "issues": [],
        "details": "",
    }

    try:
        # 提取 JSON
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(response[start:end])
            result.update(parsed)
    except Exception as e:
        result["parse_error"] = str(e)

    # Step 4: 保存结果到文件 (透明、可追溯)
    result_file = f"{output_dir}/detection_{timestamp}.json"
    write_json(result_file, result)
    print(f"[4/4] 结果已保存: {result_file}")

    # 更新 agent_log
    log_result = f"- 结果: {result['result']} (置信度: {result.get('confidence', 'N/A')}%)\n"
    if result.get('issues'):
        log_result += f"- 问题: {', '.join(result['issues'])}\n"
    log_result += f"- 详细结果: {result_file}\n"
    append_file(f"{output_dir}/agent_log.md", log_result)

    # 更新 context.md (如果存在)
    _update_context(context_file, result)

    return result


def _update_context(context_file: str, result: dict):
    """更新 context.md 状态"""
    if not file_exists(context_file):
        # 创建初始 context
        initial_context = """# Context

## Who I Am
缺陷检测 Agent - 使用视觉能力检测热熔螺母缺陷

## Recent Activity
"""
        write_file(context_file, initial_context)

    # 追加最近活动
    activity = f"- [{result['timestamp']}] 检测 {os.path.basename(result['test_image'])}: {result['result']}\n"
    append_file(context_file, activity)


# ============================================================
# CLI 入口
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="热熔螺母缺陷检测 (Agent-native 示例)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检测单张图片
  python detect_defect.py --test image.jpg --ref golden.jpg

  # 使用环境变量提供默认样本
  export DEFECT_TEST_IMAGE=/path/to/ng.jpg
  export DEFECT_REF_IMAGE=/path/to/golden.jpg
  python detect_defect.py

原语组合流程:
  1. vision_chat(images, prompt) - 核心视觉分析
  2. write_json(path, data) - 保存结果
  3. append_file(log, entry) - 记录日志

提示:
  如需使用 detect_objects/segment_objects 等本地模型，请先启动 local-vision-api。
""")

    parser.add_argument(
        "--test",
        default=os.getenv("DEFECT_TEST_IMAGE", ""),
        help="待检测图像路径 (可用 DEFECT_TEST_IMAGE 设置默认值)",
    )
    parser.add_argument(
        "--ref",
        default=os.getenv("DEFECT_REF_IMAGE", ""),
        help="参考标准样本路径 (可用 DEFECT_REF_IMAGE 设置默认值)",
    )
    parser.add_argument(
        "--output",
        default="./output",
        help="输出目录",
    )

    args = parser.parse_args()

    if not args.test or not os.path.exists(args.test):
        print("缺少待检测图像路径，请通过 --test 或 DEFECT_TEST_IMAGE 提供。")
        sys.exit(1)

    if not args.ref or not os.path.exists(args.ref):
        print("缺少参考图像路径，请通过 --ref 或 DEFECT_REF_IMAGE 提供。")
        sys.exit(1)

    print("=" * 60)
    print("热熔螺母缺陷检测 (Agent-native)")
    print("=" * 60)
    print(f"参考样本: {os.path.basename(args.ref)}")
    print(f"待检测: {os.path.basename(args.test)}")
    print("=" * 60)

    result = run_defect_detection(
        test_image=args.test,
        reference_image=args.ref,
        output_dir=args.output,
    )

    print("\n" + "=" * 60)
    print(f"检测结果: {result['result']}")
    if result.get('confidence'):
        print(f"置信度: {result['confidence']}%")
    if result.get('issues'):
        print(f"发现问题: {', '.join(result['issues'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
