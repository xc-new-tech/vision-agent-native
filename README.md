# Vision Agent Native

一个 Agent-native 架构的视觉 AI 助手框架。

## 核心理念

- **Features are outcomes, not code** - 功能是结果，而非代码
- **Tools are atomic primitives** - 工具是原子性的原语操作
- **Judgment belongs to the agent** - 判断属于 Agent，工具只执行
- **Files as universal interface** - 文件作为通用接口

## 特性

- 单一 VisionAgent 循环（无 planner/coder 分离）
- 原子化工具（detect、segment、ocr 独立操作）
- 基于文件的状态管理
- 支持 Anthropic 和 OpenAI LLM
- 本地视觉 API（YOLO-World + SAM3）

## 快速开始

### 安装

```bash
# 克隆仓库
git clone <repo-url>
cd vision-agent-native

# 安装依赖
pip install -e .
```

### 配置环境变量

在项目根目录创建 `.env` 文件：

```bash
ANTHROPIC_API_KEY=your_key
ANTHROPIC_BASE_URL=optional_custom_url   # 可选
VISION_API_URL=http://localhost:8001     # 本地视觉 API
```

### 启动本地视觉 API

```bash
cd local-vision-api
pip install -r requirements.txt
python app.py
```

### 基础使用

```python
from vision_agent_native import VisionAgent

# 创建 Agent
agent = VisionAgent(verbose=True)

# 执行任务
result = agent.run(
    task="检测图像中的所有人物，返回数量和位置",
    image_path="path/to/image.jpg",
    workspace="./workspace",
)

print(result)
```

## 项目结构

```
vision_agent_native/
├── agent.py          # 核心 Agent 循环 - VisionAgent 类
├── llm.py            # LLM 抽象层（Anthropic/OpenAI）
├── config.py         # 配置类
├── context.py        # Context 文件管理
├── workspace.py      # 工作区目录结构
├── primitives/       # 原子工具
│   ├── files.py      # read_file, write_file, list_dir
│   ├── images.py     # load_image, save_image, resize, crop
│   └── vision.py     # detect_objects, segment_objects, ocr, vqa
├── local-vision-api/ # 本地 YOLO-World + SAM3 API 服务
└── examples/         # 示例代码
    ├── basic_usage.py
    └── chat/         # 聊天演示（FastAPI + Next.js）
```

## 可用工具

| 工具 | 描述 |
|------|------|
| `read_file` | 读取文件内容 |
| `write_file` | 写入文件 |
| `list_dir` | 列出目录 |
| `load_image` | 加载图像 |
| `save_image` | 保存图像 |
| `get_image_info` | 获取图像信息 |
| `detect_objects` | 对象检测 |
| `segment_objects` | 对象分割 |
| `ocr` | 文字识别 |
| `vqa` | 视觉问答 |
| `filter_by_score` | 按置信度过滤 |
| `complete` | 标记任务完成 |

## 数据流

1. 用户提供任务 + 可选图像
2. 图像复制到 `workspace/input/`，预加载为 `image_0`
3. Agent 循环：LLM 决策 → 执行工具 → 观察结果 → 重复
4. 输出文件保存到 `workspace/output/`
5. `complete` 工具信号任务完成

## 变量系统

Agent 按前缀存储运行时变量：

- `image_0, image_1, ...` - 加载的图像
- `detections_0, detections_1, ...` - 检测结果
- `segments_0, segments_1, ...` - 分割掩码

## 本地视觉 API

使用 YOLO-World 和 SAM3 模型提供本地视觉检测和分割服务。

### 模型

| 模型 | 功能 |
|------|------|
| YOLO-World | 开放词汇对象检测 |
| SAM3 | 概念分割（支持文本 prompt） |

### API 端点

- `POST /v1/tools/text-to-object-detection` - 对象检测
- `POST /v1/tools/sam2` - 边界框分割
- `POST /v1/tools/sam3-concept` - 文本概念分割

详见 [local-vision-api/README.md](local-vision-api/README.md)

## Chat 演示

提供基于 FastAPI + Next.js 的聊天界面演示。

```bash
cd examples/chat
make setup  # 安装依赖
make run    # 启动服务
```

详见 [examples/chat/README.md](examples/chat/README.md)

## 开发说明

- 边界框格式：`[x1, y1, x2, y2]` 归一化坐标（0-1）
- 工具返回 `ToolResult(status, data, message)`
- Status：`CONTINUE`（继续）、`COMPLETE`（完成）、`ERROR`（错误）
- 视觉 API 需运行以支持 detect/segment 操作

## 运行测试

```bash
python test_agent.py
```

## License

MIT
