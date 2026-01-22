"""
Vision Agent - 核心 Agent 循环

Agent-native 原则:
- 单一循环，不分 planner/coder
- 判断属于 Agent，工具只执行
- 显式完成信号
"""

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from vision_agent_native.config import Config, DEFAULT_CONFIG
from vision_agent_native.context import Context
from vision_agent_native.llm import LLM
from vision_agent_native.workspace import Workspace
from vision_agent_native.primitives import files, images, vision


class Status(Enum):
    """任务状态"""
    CONTINUE = "continue"   # 继续执行
    COMPLETE = "complete"   # 任务完成
    ERROR = "error"         # 发生错误


@dataclass
class ToolResult:
    """工具执行结果"""
    status: Status
    data: Any
    message: str


# ========== 工具定义 ==========

TOOLS = [
    # 文件操作
    {
        "name": "read_file",
        "description": "读取文件内容",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "写入文件内容",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "content": {"type": "string", "description": "要写入的内容"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_dir",
        "description": "列出目录内容",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "目录路径"},
            },
            "required": ["path"],
        },
    },
    # 图像操作
    {
        "name": "load_image",
        "description": "加载图像文件",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "图像路径"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "save_image",
        "description": "保存图像到文件",
        "parameters": {
            "type": "object",
            "properties": {
                "image_var": {"type": "string", "description": "图像变量名"},
                "path": {"type": "string", "description": "保存路径"},
            },
            "required": ["image_var", "path"],
        },
    },
    {
        "name": "get_image_info",
        "description": "获取图像信息 (宽度、高度、格式)",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "图像路径"},
            },
            "required": ["path"],
        },
    },
    # 视觉操作
    {
        "name": "detect_objects",
        "description": "检测图像中的对象，返回边界框列表",
        "parameters": {
            "type": "object",
            "properties": {
                "image_var": {"type": "string", "description": "图像变量名"},
                "prompt": {"type": "string", "description": "要检测的对象描述"},
                "threshold": {"type": "number", "description": "置信度阈值 (0-1)", "default": 0.3},
            },
            "required": ["image_var", "prompt"],
        },
    },
    {
        "name": "segment_objects",
        "description": "基于检测结果分割对象，返回掩码",
        "parameters": {
            "type": "object",
            "properties": {
                "image_var": {"type": "string", "description": "图像变量名"},
                "detections_var": {"type": "string", "description": "检测结果变量名"},
            },
            "required": ["image_var", "detections_var"],
        },
    },
    {
        "name": "ocr",
        "description": "识别图像中的文字",
        "parameters": {
            "type": "object",
            "properties": {
                "image_var": {"type": "string", "description": "图像变量名"},
            },
            "required": ["image_var"],
        },
    },
    {
        "name": "vqa",
        "description": "视觉问答 - 回答关于图像的问题",
        "parameters": {
            "type": "object",
            "properties": {
                "image_var": {"type": "string", "description": "图像变量名"},
                "question": {"type": "string", "description": "问题"},
            },
            "required": ["image_var", "question"],
        },
    },
    # 辅助操作
    {
        "name": "filter_by_score",
        "description": "按置信度过滤检测结果",
        "parameters": {
            "type": "object",
            "properties": {
                "detections_var": {"type": "string", "description": "检测结果变量名"},
                "threshold": {"type": "number", "description": "置信度阈值"},
            },
            "required": ["detections_var", "threshold"],
        },
    },
    # 完成信号
    {
        "name": "complete",
        "description": "标记任务完成并返回最终结果",
        "parameters": {
            "type": "object",
            "properties": {
                "result": {"type": "string", "description": "最终结果"},
                "output_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "输出文件列表",
                },
            },
            "required": ["result"],
        },
    },
]

# ========== System Prompt ==========

SYSTEM_PROMPT = """你是 Vision Agent，一个视觉 AI 助手。

## 核心原则
1. **原子操作**: 每次只调用一个工具，观察结果后再决定下一步
2. **判断属于你**: 工具只执行操作，所有决策由你做出
3. **显式完成**: 任务完成后必须调用 `complete` 工具

## 可用工具
- `read_file`, `write_file`, `list_dir`: 文件操作
- `load_image`, `save_image`, `get_image_info`: 图像操作
- `detect_objects`: 检测图像中的对象
- `segment_objects`: 分割对象
- `ocr`: 文字识别
- `vqa`: 视觉问答
- `filter_by_score`: 过滤检测结果
- `complete`: 标记任务完成

## 工作流程
1. 理解任务
2. 加载必要的图像
3. 执行视觉操作 (检测/分割/OCR等)
4. 分析结果，决定下一步
5. 保存输出文件
6. 调用 `complete` 返回结果

## 变量管理
- 使用 `image_var` 引用之前加载的图像
- 使用 `detections_var` 引用检测结果
- 变量名格式: `image_0`, `detections_0`, etc.

## 注意事项
- 所有输出文件保存到 output/ 目录
- 边界框格式: [x1, y1, x2, y2]，归一化坐标 (0-1)
- 任务完成后必须调用 complete 工具
"""


class VisionAgent:
    """核心 Vision Agent

    单一循环架构，判断属于 Agent
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        verbose: bool = True,
        callback: Optional[Callable[[Dict], None]] = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.verbose = verbose
        self.callback = callback or (lambda x: None)

        # 创建 LLM
        self.llm = LLM.create(self.config)

        # 变量存储 (运行时)
        self.variables: Dict[str, Any] = {}
        self.var_counter: Dict[str, int] = {}

    def run(
        self,
        task: str,
        workspace: Optional[Union[str, Workspace]] = None,
        image_path: Optional[str] = None,
    ) -> str:
        """执行任务

        Args:
            task: 任务描述
            workspace: 工作区路径或 Workspace 对象
            image_path: 输入图像路径 (可选)

        Returns:
            任务结果
        """
        # 初始化工作区
        if workspace is None:
            workspace = Workspace(self.config.default_workspace)
        elif isinstance(workspace, str):
            workspace = Workspace(workspace)
        workspace.initialize()

        # 初始化 Context
        context = Context(str(workspace.root))
        context.load()
        context.set_task(task)
        context.update_state("Status", "running")

        # 如果有输入图像，复制到 input 目录
        if image_path:
            input_path = workspace.add_input(image_path)
            context.add_resource(f"input/{input_path.split('/')[-1]}")
            # 预加载图像
            img = images.load_image(input_path)
            var_name = self._store_variable("image", img)
            context.add_activity(f"加载图像: {image_path} -> {var_name}")

        # 记录开始
        workspace.log(f"开始任务: {task}", "info")
        self._log(f"开始任务: {task}")

        # 初始化消息
        user_message = self._build_user_message(task, workspace, image_path)
        context.add_message("user", user_message)

        # 主循环
        result = None
        for turn in range(self.config.max_turns):
            self._log(f"Turn {turn + 1}/{self.config.max_turns}")

            # 调用 LLM
            response = self.llm.chat(
                messages=context.get_messages_for_llm(),
                system=SYSTEM_PROMPT + "\n\n" + context.to_system_prompt(),
                tools=TOOLS,
            )

            # 处理响应
            if isinstance(response, str):
                # 纯文本响应，继续
                context.add_message("assistant", response)
                self._log(f"Assistant: {response[:200]}...")
                continue

            # 有工具调用
            content = response.get("content", "")
            tool_calls = response.get("tool_calls", [])

            if content:
                context.add_message("assistant", content)

            if not tool_calls:
                self._log("没有工具调用，继续...")
                continue

            # 执行工具
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["arguments"]

                self._log(f"调用工具: {tool_name}({json.dumps(tool_args, ensure_ascii=False)})")
                workspace.log(f"调用: {tool_name}", "info")

                # 执行
                tool_result = self._execute_tool(tool_name, tool_args, workspace)

                # 记录结果
                result_msg = f"工具 {tool_name} 结果: {tool_result.message}"
                context.add_message("observation", result_msg)
                context.add_activity(f"{tool_name}: {tool_result.status.value}")

                self._log(f"结果: {tool_result.message[:200]}...")

                # 检查完成状态
                if tool_result.status == Status.COMPLETE:
                    result = tool_result.data
                    context.update_state("Status", "complete")
                    context.save()
                    workspace.log(f"任务完成: {result}", "success")
                    self._log(f"任务完成: {result}")
                    return str(result)

                if tool_result.status == Status.ERROR:
                    workspace.log(f"错误: {tool_result.message}", "error")
                    # 继续让 Agent 处理错误

        # 超过最大轮次
        context.update_state("Status", "timeout")
        context.save()
        workspace.log(f"超过最大轮次 ({self.config.max_turns})", "warn")
        return f"任务未完成: 超过最大轮次 ({self.config.max_turns})"

    def _build_user_message(
        self,
        task: str,
        workspace: Workspace,
        image_path: Optional[str] = None,
    ) -> str:
        """构建用户消息"""
        parts = [f"任务: {task}"]

        if image_path:
            parts.append(f"输入图像: input/{image_path.split('/')[-1]} (已加载为 image_0)")

        # 列出可用文件
        inputs = workspace.list_inputs()
        if inputs:
            parts.append("可用输入文件:")
            for f in inputs[:10]:
                parts.append(f"  - {f['name']}")

        return "\n".join(parts)

    def _store_variable(self, prefix: str, value: Any) -> str:
        """存储变量并返回变量名"""
        count = self.var_counter.get(prefix, 0)
        var_name = f"{prefix}_{count}"
        self.variables[var_name] = value
        self.var_counter[prefix] = count + 1
        return var_name

    def _get_variable(self, var_name: str) -> Any:
        """获取变量值"""
        return self.variables.get(var_name)

    def _execute_tool(
        self,
        name: str,
        args: Dict[str, Any],
        workspace: Workspace,
    ) -> ToolResult:
        """执行工具"""
        try:
            if name == "complete":
                return ToolResult(
                    status=Status.COMPLETE,
                    data=args.get("result", ""),
                    message=f"任务完成: {args.get('result', '')}",
                )

            elif name == "read_file":
                content = files.read_file(args["path"])
                return ToolResult(
                    status=Status.CONTINUE,
                    data=content,
                    message=f"读取文件成功，长度: {len(content)} 字符",
                )

            elif name == "write_file":
                path = args["path"]
                # 确保写入 output 目录
                if not path.startswith(str(workspace.output_dir)):
                    path = workspace.get_output_path(path.split("/")[-1])
                files.write_file(path, args["content"])
                return ToolResult(
                    status=Status.CONTINUE,
                    data=path,
                    message=f"写入文件成功: {path}",
                )

            elif name == "list_dir":
                items = files.list_dir(args["path"])
                return ToolResult(
                    status=Status.CONTINUE,
                    data=items,
                    message=f"目录包含 {len(items)} 个项目: {[i['name'] for i in items[:5]]}...",
                )

            elif name == "load_image":
                img = images.load_image(args["path"])
                var_name = self._store_variable("image", img)
                return ToolResult(
                    status=Status.CONTINUE,
                    data=var_name,
                    message=f"图像已加载为 {var_name}，尺寸: {img.shape}",
                )

            elif name == "save_image":
                img = self._get_variable(args["image_var"])
                if img is None:
                    return ToolResult(
                        status=Status.ERROR,
                        data=None,
                        message=f"变量不存在: {args['image_var']}",
                    )
                path = args["path"]
                if not path.startswith(str(workspace.output_dir)):
                    path = workspace.get_output_path(path.split("/")[-1])
                images.save_image(img, path)
                return ToolResult(
                    status=Status.CONTINUE,
                    data=path,
                    message=f"图像已保存: {path}",
                )

            elif name == "get_image_info":
                info = images.get_image_info(args["path"])
                return ToolResult(
                    status=Status.CONTINUE,
                    data=info,
                    message=f"图像信息: {info}",
                )

            elif name == "detect_objects":
                img = self._get_variable(args["image_var"])
                if img is None:
                    return ToolResult(
                        status=Status.ERROR,
                        data=None,
                        message=f"变量不存在: {args['image_var']}",
                    )
                threshold = args.get("threshold", 0.3)
                detections = vision.detect_objects(img, args["prompt"], threshold)
                var_name = self._store_variable("detections", detections)
                return ToolResult(
                    status=Status.CONTINUE,
                    data=var_name,
                    message=f"检测到 {len(detections)} 个对象，存储为 {var_name}。结果: {json.dumps(detections, ensure_ascii=False)}",
                )

            elif name == "segment_objects":
                img = self._get_variable(args["image_var"])
                dets = self._get_variable(args["detections_var"])
                if img is None or dets is None:
                    return ToolResult(
                        status=Status.ERROR,
                        data=None,
                        message="变量不存在",
                    )
                segments = vision.segment_objects(img, dets)
                var_name = self._store_variable("segments", segments)
                return ToolResult(
                    status=Status.CONTINUE,
                    data=var_name,
                    message=f"分割完成，{len(segments)} 个掩码，存储为 {var_name}",
                )

            elif name == "ocr":
                img = self._get_variable(args["image_var"])
                if img is None:
                    return ToolResult(
                        status=Status.ERROR,
                        data=None,
                        message=f"变量不存在: {args['image_var']}",
                    )
                texts = vision.ocr(img)
                return ToolResult(
                    status=Status.CONTINUE,
                    data=texts,
                    message=f"识别到 {len(texts)} 段文字: {[t['text'][:20] for t in texts[:5]]}...",
                )

            elif name == "vqa":
                img = self._get_variable(args["image_var"])
                if img is None:
                    return ToolResult(
                        status=Status.ERROR,
                        data=None,
                        message=f"变量不存在: {args['image_var']}",
                    )
                answer = vision.vqa(img, args["question"])
                return ToolResult(
                    status=Status.CONTINUE,
                    data=answer,
                    message=f"回答: {answer}",
                )

            elif name == "filter_by_score":
                dets = self._get_variable(args["detections_var"])
                if dets is None:
                    return ToolResult(
                        status=Status.ERROR,
                        data=None,
                        message=f"变量不存在: {args['detections_var']}",
                    )
                filtered = vision.filter_by_score(dets, args["threshold"])
                var_name = self._store_variable("detections", filtered)
                return ToolResult(
                    status=Status.CONTINUE,
                    data=var_name,
                    message=f"过滤后剩余 {len(filtered)} 个对象，存储为 {var_name}",
                )

            else:
                return ToolResult(
                    status=Status.ERROR,
                    data=None,
                    message=f"未知工具: {name}",
                )

        except Exception as e:
            return ToolResult(
                status=Status.ERROR,
                data=None,
                message=f"工具执行错误: {str(e)}",
            )

    def _log(self, message: str) -> None:
        """输出日志"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")

        self.callback({"type": "log", "message": message})
