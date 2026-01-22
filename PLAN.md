# Vision Agent Native - Agent-native Architecture 重构计划

## 背景分析

### 当前 vision_agent 架构问题

1. **工具太重** - tools.py (~124KB) 包含大量业务逻辑，不是原子原语
2. **Agent 层级复杂** - V2、V3 多版本，planner/coder/tester 多角色分离
3. **配置分散** - 多个 config 文件，每个角色单独配置
4. **违反 Granularity 原则** - 工具捆绑了判断逻辑（如 `countgd_sam2_instance_segmentation` 既检测又分割）

### Agent-native Architecture 核心原则

1. **Parity** - Agent 能做用户能做的一切
2. **Granularity** - 工具是原子原语，判断属于 Agent
3. **Composability** - 通过组合原语实现新功能
4. **Files as Interface** - 文件作为通用接口
5. **Emergent Capability** - 涌现能力，能完成未被明确设计的任务

---

## 新架构设计

### 目录结构

```
vision_agent_native/
├── __init__.py
├── agent.py              # 单一 VisionAgent 循环
├── primitives/           # 原子原语工具
│   ├── __init__.py
│   ├── files.py          # 文件操作 (read, write, list)
│   ├── images.py         # 图像操作 (load, save, resize, crop)
│   ├── vision.py         # 视觉模型 (detect, segment, ocr, vqa)
│   └── execution.py      # 代码执行 (run_python)
├── context.py            # Context 文件管理
├── workspace.py          # 工作区管理
├── llm.py                # LLM 接口 (Anthropic/OpenAI)
├── config.py             # 简化配置
└── PLAN.md               # 本文件
```

### 核心设计

#### 1. 单一 Agent 循环（不再有 planner/coder 分离）

```python
class VisionAgent:
    def run(self, task: str, workspace: str) -> str:
        context = Context(workspace)
        context.load()  # 读取 context.md

        while True:
            # Agent 做判断，选择工具
            response = self.llm.chat(
                system=self.system_prompt,
                messages=context.messages,
                tools=self.primitives
            )

            if response.is_complete:
                context.save()
                return response.result

            # 执行原子工具
            result = self.execute_tool(response.tool_call)
            context.add_observation(result)

            if response.signals_error:
                # Agent 决定如何恢复，而非代码
                continue
```

#### 2. 原子化工具设计

```python
# 不好的设计 (当前架构)
def countgd_sam2_instance_segmentation(image, prompt):
    # 捆绑了: 检测 + 分割 + 后处理
    detections = countgd_detect(image, prompt)
    masks = sam2_segment(image, detections)
    return post_process(masks)

# 好的设计 (Agent-native)
@primitive
def detect(image: str, prompt: str) -> list:
    """检测图像中的对象，返回边界框"""
    ...

@primitive
def segment(image: str, boxes: list) -> list:
    """基于边界框分割对象，返回掩码"""
    ...

@primitive
def filter_by_score(detections: list, threshold: float) -> list:
    """按置信度过滤检测结果"""
    ...

# 后处理、过滤、组合由 Agent 判断决定
```

#### 3. 文件作为通用接口

```
workspace/
├── context.md            # Agent 工作记忆
│   ├── Who I Am
│   ├── Current Task
│   ├── What Exists (files, images)
│   ├── Recent Activity
│   └── Guidelines
├── agent_log.md          # 操作日志（透明）
├── input/                # 输入文件
│   └── image.jpg
├── output/               # 输出结果
│   ├── detections.json
│   └── result.png
└── temp/                 # 临时文件
```

#### 4. 完成信号机制

```python
class ToolResult:
    status: Literal["continue", "complete", "error"]
    data: Any
    message: str

# Agent 显式声明完成，而非猜测
```

---

## 待办事项

### Phase 1: 项目初始化
- [ ] 创建目录结构
- [ ] 编写 __init__.py
- [ ] 编写 config.py（简化配置）

### Phase 2: 核心原语工具
- [ ] 实现 primitives/files.py (read_file, write_file, list_dir)
- [ ] 实现 primitives/images.py (load_image, save_image, display)
- [ ] 实现 primitives/vision.py (detect, segment, ocr, vqa)
- [ ] 实现 primitives/execution.py (run_python)

### Phase 3: Context 管理
- [ ] 实现 context.py (load, save, update)
- [ ] 实现 workspace.py (初始化工作区)

### Phase 4: Agent 核心
- [ ] 实现 llm.py (Anthropic 接口)
- [ ] 实现 agent.py (单一循环 Agent)
- [ ] 实现工具注册和调用机制

### Phase 5: 测试与示例
- [ ] 编写基础测试
- [ ] 创建使用示例

---

## 与当前架构对比

| 特性 | 当前架构 | Agent-native |
|------|---------|--------------|
| Agent 层级 | 多层 (planner/coder/tester) | 单一循环 |
| 工具设计 | 重型复合工具 | 原子原语 |
| 状态管理 | 内存中 | 文件系统 |
| 决策位置 | 分散在代码中 | 集中在 Agent |
| 透明度 | 低（黑盒） | 高（可检查文件） |
| 扩展性 | 需要代码修改 | 修改 prompt |

---

## 审查

### 已完成的工作

#### 1. 目录结构 ✅
```
vision_agent_native/
├── __init__.py           # 包入口
├── config.py             # 简化配置 (单一 Config 类)
├── llm.py                # LLM 接口 (Anthropic/OpenAI)
├── context.py            # Context 文件管理
├── workspace.py          # 工作区管理
├── agent.py              # 核心 Agent 循环
├── primitives/
│   ├── __init__.py
│   ├── files.py          # 文件操作原语
│   ├── images.py         # 图像操作原语
│   └── vision.py         # 视觉模型原语
├── examples/
│   ├── __init__.py
│   └── basic_usage.py    # 使用示例
└── PLAN.md               # 本文件
```

#### 2. 核心组件实现 ✅

| 组件 | 文件 | 功能 |
|------|------|------|
| Config | `config.py` | 单一配置类，支持 Anthropic/OpenAI |
| LLM | `llm.py` | AnthropicLLM, OpenAILLM 实现 |
| Context | `context.py` | context.md 文件管理，消息历史 |
| Workspace | `workspace.py` | 工作区目录结构，日志 |
| Agent | `agent.py` | 核心循环，工具执行 |

#### 3. 原语工具 ✅

**文件操作** (`primitives/files.py`):
- `read_file`, `write_file`, `append_file`
- `list_dir`, `delete_file`, `file_exists`
- `read_json`, `write_json`

**图像操作** (`primitives/images.py`):
- `load_image`, `save_image`
- `get_image_info`, `resize_image`, `crop_image`
- `image_to_base64`, `base64_to_image`

**视觉操作** (`primitives/vision.py`):
- `detect_objects` - 对象检测
- `segment_objects` - 实例分割
- `ocr` - 文字识别
- `vqa` - 视觉问答
- `filter_by_score`, `filter_by_label`, `nms` - 辅助函数

#### 4. Agent-native 原则实现 ✅

| 原则 | 实现方式 |
|------|----------|
| Parity | Agent 可调用所有文件/图像/视觉工具 |
| Granularity | 工具原子化，detect 和 segment 分离 |
| Composability | Agent 通过工具组合完成任务 |
| Files as Interface | context.md, agent_log.md, workspace 目录 |
| 显式完成 | `complete` 工具标记任务完成 |

### 与原架构对比

| 维度 | 原 vision_agent | 新 vision_agent_native |
|------|-----------------|------------------------|
| 代码行数 | ~10000+ 行 | ~1200 行 |
| Agent 类型 | 4种 (V2, V3, Planner, Coder) | 1种 (VisionAgent) |
| 配置项 | 12+ 个 LLM 配置 | 1 个 Config |
| 工具数量 | 50+ 复合工具 | 15 原子工具 |
| 状态管理 | 内存 | 文件系统 |

### 使用方式

```python
from vision_agent_native import VisionAgent

agent = VisionAgent(verbose=True)
result = agent.run(
    task="检测图像中的所有人物",
    image_path="test.jpg",
    workspace="./workspace",
)
print(result)
```

### 后续工作

- [ ] 添加更多视觉工具 (深度估计、姿态估计等)
- [ ] 实现代码执行原语 (`execution.py`)
- [ ] 添加单元测试
- [ ] 集成本地视觉 API (YOLO + SAM2)
