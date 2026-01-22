"""
Context 管理 - Agent 的工作记忆

Agent-native 原则:
- 使用 context.md 文件存储可移植的工作记忆
- Session 开始时读取，结束时更新
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from vision_agent_native.primitives.files import read_file, write_file


class Context:
    """Context 管理器

    管理 Agent 的工作记忆，存储在 context.md 文件中
    """

    def __init__(self, workspace_path: str = "./workspace"):
        self.workspace_path = Path(workspace_path)
        self.context_file = self.workspace_path / "context.md"

        # 内存中的上下文数据
        self.who_i_am: str = ""
        self.current_task: str = ""
        self.what_exists: List[str] = []
        self.recent_activity: List[str] = []
        self.guidelines: List[str] = []
        self.current_state: Dict[str, Any] = {}

        # 消息历史 (用于 LLM 对话)
        self.messages: List[Dict[str, Any]] = []

    def load(self) -> "Context":
        """从 context.md 加载上下文"""
        if not self.context_file.exists():
            return self

        content = read_file(str(self.context_file))
        self._parse_context(content)
        return self

    def save(self) -> "Context":
        """保存上下文到 context.md"""
        content = self._generate_context()
        write_file(str(self.context_file), content)
        return self

    def _parse_context(self, content: str) -> None:
        """解析 context.md 内容"""
        sections = self._split_sections(content)

        self.who_i_am = sections.get("Who I Am", "").strip()
        self.current_task = sections.get("Current Task", "").strip()
        self.what_exists = self._parse_list(sections.get("What Exists", ""))
        self.recent_activity = self._parse_list(sections.get("Recent Activity", ""))
        self.guidelines = self._parse_list(sections.get("Guidelines", ""))

        # 解析 Current State
        state_text = sections.get("Current State", "")
        self.current_state = {}
        for line in state_text.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                self.current_state[key.strip("- ")] = value.strip()

    def _split_sections(self, content: str) -> Dict[str, str]:
        """将内容分割为各个 section"""
        sections = {}
        current_section = None
        current_content = []

        for line in content.split("\n"):
            if line.startswith("## "):
                if current_section:
                    sections[current_section] = "\n".join(current_content)
                current_section = line[3:].strip()
                current_content = []
            elif current_section:
                current_content.append(line)

        if current_section:
            sections[current_section] = "\n".join(current_content)

        return sections

    def _parse_list(self, text: str) -> List[str]:
        """解析列表格式的文本"""
        items = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("- "):
                items.append(line[2:])
            elif line:
                items.append(line)
        return items

    def _generate_context(self) -> str:
        """生成 context.md 内容"""
        lines = ["# Context", ""]

        # Who I Am
        lines.extend(["## Who I Am", self.who_i_am or "(未设置)", ""])

        # Current Task
        lines.extend(["## Current Task", self.current_task or "(无)", ""])

        # What Exists
        lines.append("## What Exists")
        for item in self.what_exists:
            lines.append(f"- {item}")
        lines.append("")

        # Recent Activity
        lines.append("## Recent Activity")
        # 只保留最近 10 条
        for item in self.recent_activity[-10:]:
            lines.append(f"- {item}")
        lines.append("")

        # Guidelines
        lines.append("## Guidelines")
        for item in self.guidelines:
            lines.append(f"- {item}")
        lines.append("")

        # Current State
        lines.append("## Current State")
        for key, value in self.current_state.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

        return "\n".join(lines)

    # ========== 便捷方法 ==========

    def set_task(self, task: str) -> "Context":
        """设置当前任务"""
        self.current_task = task
        self.add_activity(f"任务设置: {task[:50]}...")
        return self

    def add_activity(self, activity: str) -> "Context":
        """添加活动记录"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.recent_activity.append(f"{timestamp} {activity}")
        return self

    def update_state(self, key: str, value: Any) -> "Context":
        """更新状态"""
        self.current_state[key] = str(value)
        self.current_state["Last update"] = datetime.now().isoformat()
        return self

    def add_resource(self, resource: str) -> "Context":
        """添加可用资源"""
        if resource not in self.what_exists:
            self.what_exists.append(resource)
        return self

    def add_message(
        self,
        role: str,
        content: str,
        images: Optional[List[str]] = None,
    ) -> "Context":
        """添加消息到历史

        Args:
            role: 角色 (user, assistant, observation)
            content: 消息内容
            images: base64 编码的图像列表
        """
        msg = {"role": role, "content": content}
        if images:
            msg["images"] = images
        self.messages.append(msg)
        return self

    def get_messages_for_llm(self) -> List[Dict[str, Any]]:
        """获取用于 LLM 的消息格式"""
        return self.messages

    def clear_messages(self) -> "Context":
        """清空消息历史"""
        self.messages = []
        return self

    def to_system_prompt(self) -> str:
        """生成系统提示词"""
        return f"""# Context

## Who I Am
{self.who_i_am}

## Current Task
{self.current_task}

## Available Resources
{chr(10).join('- ' + r for r in self.what_exists)}

## Guidelines
{chr(10).join('- ' + g for g in self.guidelines)}
"""
