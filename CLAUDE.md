# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vision Agent Native is an agent-native vision AI assistant framework. It follows a simplified architecture where:
- Single VisionAgent loop (no planner/coder separation)
- Atomic primitive tools (detect, segment, ocr are separate operations)
- File-based state management (context.md, agent_log.md)
- LLM makes all decisions, tools only execute

## Common Commands

```bash
# Run tests
python test_agent.py

# Start local vision API (required for detection/segmentation)
cd local-vision-api && python app.py

# Run example
python examples/basic_usage.py

# Run chat demo (frontend + backend)
cd examples/chat && make run
```

## Environment Setup

Required environment variables in `.env`:
```bash
ANTHROPIC_API_KEY=your_key
ANTHROPIC_BASE_URL=optional_custom_url   # optional
VISION_API_URL=http://localhost:8001     # local vision API
```

## Architecture

```
vision_agent_native/
├── agent.py          # Core agent loop - VisionAgent class
├── llm.py            # LLM abstraction (Anthropic/OpenAI)
├── config.py         # Single Config class
├── context.py        # Context file management (context.md)
├── workspace.py      # Workspace directory structure
├── primitives/       # Atomic tools
│   ├── files.py      # read_file, write_file, list_dir
│   ├── images.py     # load_image, save_image, resize, crop
│   └── vision.py     # detect_objects, segment_objects, ocr, vqa
└── local-vision-api/ # Local YOLO-World + SAM3 API server
```

### Key Components

**VisionAgent (agent.py)**: Main agent loop. Manages tool execution cycle, variable storage (`image_0`, `detections_0`), and completion signals. Tools defined in `TOOLS` list.

**LLM (llm.py)**: Abstraction layer supporting Anthropic and OpenAI. Factory method `LLM.create(config)` returns appropriate implementation.

**Vision Primitives (primitives/vision.py)**:
- `vision_chat()` - Core LLM vision capability
- `detect_objects()` - Calls local YOLO-World API
- `segment_objects()` - Calls local SAM3 API
- `ocr()` - Uses LLM for text recognition

**Local Vision API (local-vision-api/app.py)**: FastAPI server running YOLO-World and SAM3 models. Endpoints:
- `POST /v1/tools/text-to-object-detection` - Detection
- `POST /v1/tools/sam2` - Segmentation with bboxes
- `POST /v1/tools/sam3-concept` - Text-based concept segmentation

### Data Flow

1. User provides task + optional image
2. Image copied to `workspace/input/`, preloaded as `image_0`
3. Agent loop: LLM decides tool → execute → observe → repeat
4. Output files saved to `workspace/output/`
5. `complete` tool signals task completion

### Variable System

Agent stores runtime variables by prefix:
- `image_0, image_1, ...` - Loaded images
- `detections_0, detections_1, ...` - Detection results
- `segments_0, segments_1, ...` - Segmentation masks

## Development Notes

- Detection bbox format: `[x1, y1, x2, y2]` normalized (0-1)
- All tool results return `ToolResult(status, data, message)`
- Status: `CONTINUE` (keep going), `COMPLETE` (done), `ERROR` (failed)
- Vision API must be running for detect/segment operations
- LLM handles errors by deciding recovery strategy
