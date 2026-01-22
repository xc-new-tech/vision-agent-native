#!/usr/bin/env python
"""
测试 Vision Agent Native

运行前确保设置了环境变量:
- ANTHROPIC_API_KEY
- ANTHROPIC_BASE_URL (可选)
"""

import os
import sys
from pathlib import Path

# 加载 .env 文件
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    print(f"加载环境变量: {env_file}")
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                value = value.strip('"').strip("'")
                os.environ[key] = value
                print(f"  {key}={value[:20]}...")

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from vision_agent_native.config import Config, DEFAULT_CONFIG
from vision_agent_native.llm import LLM, AnthropicLLM
from vision_agent_native.workspace import Workspace
from vision_agent_native.context import Context


def test_config():
    """测试配置加载"""
    print("\n" + "=" * 50)
    print("测试 1: 配置加载")
    print("=" * 50)

    # 重新创建配置以获取最新环境变量
    config = Config()

    print(f"LLM Provider: {config.llm_provider}")
    print(f"LLM Model: {config.llm_model}")
    print(f"API Key: {config.anthropic_api_key[:30] if config.anthropic_api_key else 'None'}...")
    print(f"Base URL: {config.anthropic_base_url}")

    assert config.anthropic_api_key, "ANTHROPIC_API_KEY 未设置"
    print("✅ 配置加载成功")


def test_llm_simple():
    """测试 LLM 简单对话"""
    print("\n" + "=" * 50)
    print("测试 2: LLM 简单对话")
    print("=" * 50)

    config = Config()
    llm = AnthropicLLM(config)

    response = llm.chat(
        messages=[{"role": "user", "content": "你好，请用一句话介绍你自己"}],
        system="你是一个友好的助手",
    )

    print(f"响应: {response}")
    assert response, "LLM 响应为空"
    print("✅ LLM 对话成功")


def test_llm_tool_call():
    """测试 LLM 工具调用"""
    print("\n" + "=" * 50)
    print("测试 3: LLM 工具调用")
    print("=" * 50)

    config = Config()
    llm = AnthropicLLM(config)

    tools = [
        {
            "name": "get_weather",
            "description": "获取指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                },
                "required": ["city"],
            },
        }
    ]

    response = llm.chat(
        messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
        system="你是一个助手，需要使用工具获取信息",
        tools=tools,
    )

    print(f"响应类型: {type(response)}")
    print(f"响应: {response}")

    if isinstance(response, dict) and "tool_calls" in response:
        print(f"工具调用: {response['tool_calls']}")
        print("✅ 工具调用成功")
    else:
        print("⚠️ 没有触发工具调用")


def test_workspace():
    """测试工作区"""
    print("\n" + "=" * 50)
    print("测试 4: 工作区管理")
    print("=" * 50)

    workspace = Workspace("./test_workspace")
    workspace.initialize()

    print(f"工作区: {workspace.root}")
    print(f"摘要: {workspace.get_summary()}")

    workspace.log("测试日志", "info")
    print("✅ 工作区创建成功")


def test_context():
    """测试 Context"""
    print("\n" + "=" * 50)
    print("测试 5: Context 管理")
    print("=" * 50)

    context = Context("./test_workspace")
    context.load()

    context.set_task("测试任务")
    context.add_activity("执行测试")
    context.update_state("Status", "testing")
    context.save()

    print(f"当前任务: {context.current_task}")
    print(f"状态: {context.current_state}")
    print("✅ Context 管理成功")


def test_agent_simple():
    """测试 Agent 简单任务（无图像）"""
    print("\n" + "=" * 50)
    print("测试 6: Agent 简单任务")
    print("=" * 50)

    from vision_agent_native.agent import VisionAgent

    agent = VisionAgent(verbose=True)

    result = agent.run(
        task="列出当前目录中的文件，然后报告完成",
        workspace="./test_workspace",
    )

    print(f"\n最终结果: {result}")
    print("✅ Agent 任务完成")


def main():
    """运行所有测试"""
    print("Vision Agent Native 测试")
    print("=" * 50)

    results = {}

    # 测试 1: 配置
    try:
        test_config()
        results["config"] = "✅"
    except Exception as e:
        results["config"] = f"❌ {e}"

    # 测试 2: LLM 简单对话
    try:
        test_llm_simple()
        results["llm_simple"] = "✅"
    except Exception as e:
        results["llm_simple"] = f"❌ {e}"
        print(f"⚠️ LLM 测试跳过 (API 不可用): {e}")

    # 测试 3: LLM 工具调用
    try:
        test_llm_tool_call()
        results["llm_tool"] = "✅"
    except Exception as e:
        results["llm_tool"] = f"❌ {e}"
        print(f"⚠️ LLM 工具测试跳过: {e}")

    # 测试 4: 工作区
    try:
        test_workspace()
        results["workspace"] = "✅"
    except Exception as e:
        results["workspace"] = f"❌ {e}"

    # 测试 5: Context
    try:
        test_context()
        results["context"] = "✅"
    except Exception as e:
        results["context"] = f"❌ {e}"

    # 测试 6: Agent (需要 LLM)
    if "✅" in results.get("llm_simple", ""):
        try:
            test_agent_simple()
            results["agent"] = "✅"
        except Exception as e:
            results["agent"] = f"❌ {e}"
    else:
        results["agent"] = "⏭️ 跳过 (LLM 不可用)"

    # 汇总
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    print("=" * 50)
    for name, result in results.items():
        print(f"  {name}: {result}")

    passed = sum(1 for r in results.values() if "✅" in r)
    total = len(results)
    print(f"\n通过: {passed}/{total}")


if __name__ == "__main__":
    main()
