"""
LLM 接口

简化的 LLM 抽象层，支持 Anthropic 和 OpenAI
"""

import base64
import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

from vision_agent_native.config import Config, DEFAULT_CONFIG


class LLM(ABC):
    """LLM 抽象基类"""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        **kwargs,
    ) -> Union[str, Dict[str, Any]]:
        """发送聊天请求

        Args:
            messages: 消息列表 [{"role": str, "content": str, "images": list}]
            system: 系统提示词
            tools: 工具定义列表
            **kwargs: 其他参数

        Returns:
            如果没有工具调用，返回字符串
            如果有工具调用，返回 {"content": str, "tool_calls": list}
        """
        pass

    @staticmethod
    def create(config: Optional[Config] = None) -> "LLM":
        """工厂方法创建 LLM 实例"""
        config = config or DEFAULT_CONFIG

        if config.llm_provider == "anthropic":
            return AnthropicLLM(config)
        elif config.llm_provider == "openai":
            return OpenAILLM(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")


class AnthropicLLM(LLM):
    """Anthropic Claude 实现"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or DEFAULT_CONFIG

        import anthropic
        client_kwargs = {"api_key": self.config.anthropic_api_key}
        if self.config.anthropic_base_url:
            client_kwargs["base_url"] = self.config.anthropic_base_url
        self.client = anthropic.Anthropic(**client_kwargs)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        **kwargs,
    ) -> Union[str, Dict[str, Any]]:
        # 转换消息格式
        formatted_messages = self._format_messages(messages)

        # 构建请求参数
        request_params = {
            "model": self.config.llm_model,
            "max_tokens": self.config.llm_max_tokens,
            "messages": formatted_messages,
        }

        if system:
            request_params["system"] = system

        if tools:
            request_params["tools"] = self._format_tools(tools)

        # 发送请求
        response = self.client.messages.create(**request_params)

        # 解析响应
        return self._parse_response(response)

    def _format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict]:
        """转换为 Anthropic 消息格式"""
        formatted = []

        for msg in messages:
            role = msg["role"]
            # 将 observation 映射为 user
            if role in ["observation", "tool_result"]:
                role = "user"
            elif role not in ["user", "assistant"]:
                role = "user"

            content = []

            # 添加图像
            if "images" in msg and msg["images"]:
                # 支持两种格式:
                # 1. 字符串列表: ["base64data1", "base64data2"]
                # 2. 字典列表: [{"data": "base64data", "media_type": "image/png"}, ...]
                for img in msg["images"]:
                    if isinstance(img, dict):
                        media_type = img.get("media_type", "image/jpeg")
                        data = img.get("data", img)
                    else:
                        media_type = "image/jpeg"
                        data = img
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        }
                    })

            # 添加文本
            if msg.get("content"):
                content.append({
                    "type": "text",
                    "text": msg["content"],
                })

            formatted.append({"role": role, "content": content})

        return formatted

    def _format_tools(self, tools: List[Dict]) -> List[Dict]:
        """转换为 Anthropic 工具格式"""
        formatted = []
        for tool in tools:
            formatted.append({
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters", {"type": "object", "properties": {}}),
            })
        return formatted

    def _parse_response(self, response) -> Union[str, Dict[str, Any]]:
        """解析 Anthropic 响应"""
        text_content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })

        if tool_calls:
            return {
                "content": text_content,
                "tool_calls": tool_calls,
                "stop_reason": response.stop_reason,
            }

        return text_content


class OpenAILLM(LLM):
    """OpenAI 实现"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or DEFAULT_CONFIG

        from openai import OpenAI
        client_kwargs = {"api_key": self.config.openai_api_key}
        if self.config.openai_base_url:
            client_kwargs["base_url"] = self.config.openai_base_url
        self.client = OpenAI(**client_kwargs)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        **kwargs,
    ) -> Union[str, Dict[str, Any]]:
        # 转换消息格式
        formatted_messages = self._format_messages(messages, system)

        # 构建请求参数
        request_params = {
            "model": self.config.llm_model,
            "max_tokens": self.config.llm_max_tokens,
            "messages": formatted_messages,
        }

        if tools:
            request_params["tools"] = self._format_tools(tools)

        # 发送请求
        response = self.client.chat.completions.create(**request_params)

        # 解析响应
        return self._parse_response(response)

    def _format_messages(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
    ) -> List[Dict]:
        """转换为 OpenAI 消息格式"""
        formatted = []

        if system:
            formatted.append({"role": "system", "content": system})

        for msg in messages:
            role = msg["role"]
            if role in ["observation", "tool_result"]:
                role = "user"
            elif role not in ["user", "assistant", "system"]:
                role = "user"

            content = []

            # 添加图像
            if "images" in msg and msg["images"]:
                for img in msg["images"]:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img}",
                        }
                    })

            # 添加文本
            if msg.get("content"):
                content.append({
                    "type": "text",
                    "text": msg["content"],
                })

            # 如果只有文本，简化格式
            if len(content) == 1 and content[0]["type"] == "text":
                formatted.append({"role": role, "content": content[0]["text"]})
            else:
                formatted.append({"role": role, "content": content})

        return formatted

    def _format_tools(self, tools: List[Dict]) -> List[Dict]:
        """转换为 OpenAI 工具格式"""
        formatted = []
        for tool in tools:
            formatted.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                }
            })
        return formatted

    def _parse_response(self, response) -> Union[str, Dict[str, Any]]:
        """解析 OpenAI 响应"""
        message = response.choices[0].message

        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                })
            return {
                "content": message.content or "",
                "tool_calls": tool_calls,
                "stop_reason": response.choices[0].finish_reason,
            }

        return message.content or ""
