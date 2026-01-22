"""
简化配置 - 只需要一个 LLM 配置

Agent-native 原则: 不需要为每个角色单独配置
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """单一配置类"""

    # LLM 配置
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "anthropic")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
    )
    llm_max_tokens: int = 8192
    llm_temperature: float = 0.0

    # API Keys (从环境变量读取)
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )
    anthropic_base_url: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_BASE_URL")
    )
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    openai_base_url: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL")
    )

    # Vision API (本地或远程)
    vision_api_url: str = field(
        default_factory=lambda: os.getenv(
            "VISION_API_URL", "http://localhost:8001"
        )
    )

    # Agent 配置
    max_turns: int = 20  # 最大循环次数
    verbose: bool = True

    # 工作区配置
    default_workspace: str = "./workspace"


# 全局默认配置
DEFAULT_CONFIG = Config()
