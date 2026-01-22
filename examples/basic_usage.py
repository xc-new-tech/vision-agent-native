"""
基础使用示例

展示 Vision Agent Native 的核心用法
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vision_agent_native import VisionAgent, Workspace


def example_object_detection():
    """示例: 对象检测"""
    print("=" * 50)
    print("示例 1: 对象检测")
    print("=" * 50)

    image_path = os.getenv("EXAMPLE_IMAGE", "")
    if not image_path or not os.path.exists(image_path):
        print("未提供有效图像路径，请设置 EXAMPLE_IMAGE 或传入本地路径。")
        return

    agent = VisionAgent(verbose=True)

    result = agent.run(
        task="检测图像中的所有人物，返回他们的数量和位置",
        image_path=image_path,
        workspace=os.getenv("EXAMPLE_WORKSPACE", "./workspace_demo"),
    )

    print(f"\n结果: {result}")


def example_custom_workspace():
    """示例: 自定义工作区"""
    print("=" * 50)
    print("示例 2: 自定义工作区")
    print("=" * 50)

    workspace_path = os.getenv("EXAMPLE_WORKSPACE", "./my_workspace")
    workspace = Workspace(workspace_path)
    workspace.initialize()

    print(f"工作区摘要: {workspace.get_summary()}")

    agent = VisionAgent(verbose=True)
    result = agent.run(
        task="列出工作区中的所有文件",
        workspace=workspace,
    )

    print(f"\n结果: {result}")


def example_batch_processing():
    """示例: 批量处理"""
    print("=" * 50)
    print("示例 3: 批量处理")
    print("=" * 50)

    images_env = os.getenv("EXAMPLE_IMAGE_LIST", "")
    images = [img.strip() for img in images_env.split(",") if img.strip()]
    if not images:
        print("未提供图像列表，请设置 EXAMPLE_IMAGE_LIST (逗号分隔路径)。")
        return

    agent = VisionAgent(verbose=True)
    workspace = Workspace(os.getenv("EXAMPLE_BATCH_WORKSPACE", "./batch_workspace")).initialize()

    for img in images:
        if not os.path.exists(img):
            print(f"跳过不存在的图像: {img}")
            continue
        result = agent.run(
            task=f"分析图像 {img} 中的主要内容",
            workspace=workspace,
            image_path=img,
        )
        print(f"{img}: {result}")


def example_with_callback():
    """示例: 使用回调函数"""
    print("=" * 50)
    print("示例 4: 使用回调函数")
    print("=" * 50)

    image_path = os.getenv("EXAMPLE_IMAGE", "")
    if not image_path or not os.path.exists(image_path):
        print("未提供有效图像路径，请设置 EXAMPLE_IMAGE 或传入本地路径。")
        return

    def my_callback(event):
        if event["type"] == "log":
            print(f"[CALLBACK] {event['message']}")

    agent = VisionAgent(verbose=False, callback=my_callback)

    result = agent.run(
        task="描述这张图像的内容",
        workspace=os.getenv("EXAMPLE_CALLBACK_WORKSPACE", "./callback_demo"),
        image_path=image_path,
    )

    print(f"\n结果: {result}")


if __name__ == "__main__":
    mode = os.getenv("EXAMPLE_MODE", "workspace")

    if mode == "detect":
        example_object_detection()
    elif mode == "batch":
        example_batch_processing()
    elif mode == "callback":
        example_with_callback()
    else:
        example_custom_workspace()
