# VisionAgent Native Chat Example

这是基于本项目 `VisionAgent` 的聊天示例，后端使用 FastAPI，前端使用 Next.js。

![screenshot](https://github.com/landing-ai/vision-agent/blob/main/assets/screenshot.png?raw=true)

## Prerequisites

- Python 3.9 or higher
- Node.js 18 or higher
- npm (comes with Node.js)

## Quick Start

### 1. Setup

#### On Windows (PowerShell)

```powershell
python setup.py
```

#### On Linux/macOS (with Make)

```bash
make setup
```

### 2. 配置环境变量

复制 `.env` 并按需修改：

```bash
PORT_BACKEND=8888
PORT_FRONTEND=3000
BACKEND_HOST=localhost
```

LLM 相关的 key 请使用仓库根目录的 `.env`。

### 3. Run the App

#### On Windows (PowerShell)

```powershell
python run.py
```

#### On Linux/macOS (with Make)

```bash
make run
```

This will:
- Launch the FastAPI backend
- Start the React frontend
- Open your browser to the application
- Handle proper cleanup when you press Ctrl+C

## Debug Mode

设置 `DEBUG_HIL=true` 会启用更详细的日志输出。

## Configuration

### Changing Ports

To modify the frontend or backend port:

1. Open `.env`
2. Change the `PORT_BACKEND` or `PORT_FRONTEND` variables:
   ```bash
   PORT_BACKEND=8000
   PORT_FRONTEND=3000
   ```

## Troubleshooting

- 端口冲突: 修改 `.env` 中的端口，或手动结束占用进程
- 服务未启动: 确认已完成 `setup.py`/`make setup`
- 浏览器未自动打开: 手动访问 `http://localhost:3000`
