"""
Vision Agent Native - Agent-native Architecture

核心理念:
- Features are outcomes, not code
- Tools are atomic primitives
- Judgment belongs to the agent
- Files as universal interface
"""

from vision_agent_native.agent import VisionAgent
from vision_agent_native.context import Context
from vision_agent_native.workspace import Workspace

__version__ = "0.1.0"
__all__ = ["VisionAgent", "Context", "Workspace"]
