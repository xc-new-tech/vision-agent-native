# Vision Agent MCP Server

为 Claude Code 提供图像检测、分割、OCR 等视觉 AI 工具。

## 安装

```bash
cd mcp-server
pip install -e .
```

或直接安装依赖：

```bash
pip install "fastmcp>=2.0,<3" pillow numpy requests
```

## 配置到 Claude Code

编辑 `~/.claude/settings.json`，添加：

```json
{
  "mcpServers": {
    "vision-agent": {
      "command": "python",
      "args": ["/path/to/vision_agent_native/mcp-server/server.py"],
      "env": {
        "VISION_API_URL": "http://localhost:8001"
      }
    }
  }
}
```

## 前置要求

需要运行本地 Vision API：

```bash
cd ../local-vision-api
python app.py
```

## 可用工具

### 核心视觉工具

| 工具 | 功能 |
|------|------|
| `detect_objects` | 对象检测 (YOLO-World) |
| `segment_objects` | 实例分割 (SAM2) |
| `segment_by_text` | 文本概念分割 (SAM3) |
| `ocr` | 文字识别 |
| `vqa` | 视觉问答 |

### 扩展视觉工具

| 工具 | 功能 |
|------|------|
| `estimate_pose` | 人体姿态估计 |
| `detect_faces` | 人脸检测 |
| `classify_image` | 图像分类 |
| `compare_images` | 图像相似度 |
| `remove_background` | 背景移除 |

### 辅助工具

| 工具 | 功能 |
|------|------|
| `get_image_info` | 获取图像信息 |
| `filter_detections` | 过滤检测结果 |

## 使用示例

在 Claude Code 中：

```
请检测这张图片中的所有螺母: /path/to/image.jpg
```

Claude 会自动调用 `detect_objects` 工具进行检测。
