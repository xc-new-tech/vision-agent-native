# 本地视觉工具 API

不依赖 LandingAI，使用本地 **YOLO-World** 和 **SAM3** 模型提供视觉检测和分割服务。

## 模型说明

| 模型 | 功能 | 特点 |
|------|------|------|
| **YOLO-World** | 开放词汇对象检测 | 文本 prompt 检测任意物体 |
| **SAM3** | 概念分割 | Meta 最新模型 (2025.11)，支持文本概念分割 |

## 安装

### 1. 安装依赖

```bash
cd examples/local-vision-api
pip install -r requirements.txt
```

### 2. 模型下载

模型会在首次运行时自动下载，或手动指定路径：

```bash
# YOLO-World
export YOLO_MODEL_PATH="yolov8x-worldv2.pt"

# SAM3
export SAM3_MODEL_PATH="sam3_l.pt"
```

## 启动服务

```bash
# 启动服务 (默认端口 8001)
python app.py

# 或指定端口
PORT=8001 python app.py

# 或使用 uvicorn
uvicorn app:app --host 0.0.0.0 --port 8001
```

## API 端点

### 健康检查
```
GET /health
```

### 对象检测 (YOLO-World)
```
POST /v1/tools/text-to-object-detection

参数:
- image: 图像文件
- prompts: '["person", "car"]'  # JSON 类别列表
- confidence: 0.3               # 置信度阈值
- model: "yolo-world"
```

### 实例分割 (SAM3 - 边界框模式)
```
POST /v1/tools/sam2

参数:
- image: 图像文件
- bboxes: '[{"labels": ["person"], "bboxes": [[100,100,200,200]]}]'
- model: "sam2"

注: 端点名保持 sam2 以兼容现有代码
```

### 概念分割 (SAM3 新特性)
```
POST /v1/tools/sam3-concept

参数:
- image: 图像文件
- prompts: '["copper nut", "screw"]'  # 文本概念
- confidence: 0.3

SAM3 新特性: 无需边界框，直接用文本分割！
```

## 配置 VisionAgent 使用本地 API

在 `.env` 文件中添加:

```bash
# 使用本地视觉 API (替换 LandingAI)
LANDINGAI_URL="http://localhost:8001"
```

或设置环境变量:

```bash
export LANDINGAI_URL="http://localhost:8001"
```

这会将所有视觉工具请求路由到本地 API 服务。

## 返回格式

与 LandingAI API 完全兼容:

**对象检测:**
```json
{
  "data": [[
    {
      "label": "person",
      "score": 0.95,
      "bounding_box": [100, 100, 200, 300]
    }
  ]]
}
```

**实例分割:**
```json
{
  "data": [[[
    {
      "label": "person",
      "score": 0.98,
      "bounding_box": [100, 100, 200, 300],
      "mask": {"counts": [...], "size": [480, 640]}
    }
  ]]]
}
```

## 参考资料

- [SAM3 官方博客](https://ai.meta.com/blog/segment-anything-model-3/)
- [SAM3 GitHub](https://github.com/facebookresearch/sam3)
- [Ultralytics SAM3 文档](https://docs.ultralytics.com/models/sam-3/)
- [YOLO-World](https://docs.ultralytics.com/models/yolo-world/)
