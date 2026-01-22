"""
工作区管理

Agent-native 原则: 文件作为通用接口
Agent 和用户在同一个数据空间工作
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from vision_agent_native.primitives.files import (
    list_dir,
    read_file,
    write_file,
)


class Workspace:
    """工作区管理器

    目录结构:
    workspace/
    ├── context.md          # Agent 工作记忆
    ├── agent_log.md        # 操作日志
    ├── input/              # 输入文件
    ├── output/             # 输出结果
    └── temp/               # 临时文件
    """

    def __init__(self, root: str = "./workspace"):
        self.root = Path(root).absolute()
        self.input_dir = self.root / "input"
        self.output_dir = self.root / "output"
        self.temp_dir = self.root / "temp"
        self.context_file = self.root / "context.md"
        self.log_file = self.root / "agent_log.md"

    def initialize(self) -> "Workspace":
        """初始化工作区目录结构"""
        # 创建目录
        self.root.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)

        # 创建 context.md (如果不存在)
        if not self.context_file.exists():
            self._create_initial_context()

        # 创建 agent_log.md (如果不存在)
        if not self.log_file.exists():
            self._create_initial_log()

        return self

    def _create_initial_context(self) -> None:
        """创建初始 context.md"""
        content = f"""# Context

## Who I Am
Vision Agent - 一个视觉 AI 助手，帮助用户分析图像和视频。

## Current Task
(待设置)

## What Exists
- input/: 输入文件目录
- output/: 输出结果目录
- temp/: 临时文件目录

## Recent Activity
- {datetime.now().strftime("%Y-%m-%d %H:%M")} 工作区初始化

## Guidelines
- 所有输出保存到 output/ 目录
- 临时文件使用 temp/ 目录
- 操作完成后更新此文件

## Current State
- Status: idle
- Last update: {datetime.now().isoformat()}
"""
        write_file(str(self.context_file), content)

    def _create_initial_log(self) -> None:
        """创建初始 agent_log.md"""
        content = f"""# Agent Log

## {datetime.now().strftime("%Y-%m-%d")}

### {datetime.now().strftime("%H:%M:%S")} - 初始化
- 创建工作区: {self.root}
- 状态: ready

---
"""
        write_file(str(self.log_file), content)

    def list_inputs(self) -> List[Dict]:
        """列出输入目录中的文件"""
        return list_dir(str(self.input_dir))

    def list_outputs(self) -> List[Dict]:
        """列出输出目录中的文件"""
        return list_dir(str(self.output_dir))

    def add_input(self, source: str, name: Optional[str] = None) -> str:
        """添加输入文件

        Args:
            source: 源文件路径
            name: 目标文件名 (可选)

        Returns:
            复制后的文件路径
        """
        source_path = Path(source)
        dest_name = name or source_path.name
        dest_path = self.input_dir / dest_name

        shutil.copy2(source, dest_path)
        return str(dest_path)

    def get_output_path(self, name: str) -> str:
        """获取输出文件路径

        Args:
            name: 文件名

        Returns:
            完整的输出路径
        """
        return str(self.output_dir / name)

    def get_temp_path(self, name: str) -> str:
        """获取临时文件路径"""
        return str(self.temp_dir / name)

    def clean_temp(self) -> None:
        """清理临时目录"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir.mkdir()

    def log(self, message: str, level: str = "info") -> None:
        """记录日志到 agent_log.md

        Args:
            message: 日志内容
            level: 日志级别 (info, warn, error, success)
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        level_emoji = {
            "info": "ℹ️",
            "warn": "⚠️",
            "error": "❌",
            "success": "✅",
        }.get(level, "•")

        log_entry = f"### {timestamp} - {level_emoji} {message}\n\n"

        # 追加到日志文件
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)

    def get_summary(self) -> Dict:
        """获取工作区摘要"""
        return {
            "root": str(self.root),
            "inputs": len(self.list_inputs()),
            "outputs": len(self.list_outputs()),
            "context_exists": self.context_file.exists(),
            "log_exists": self.log_file.exists(),
        }
