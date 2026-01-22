import asyncio
import base64
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ROOT_ENV, override=False)
load_dotenv(override=True)

from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from vision_agent_native.agent import VisionAgent

PORT_FRONTEND = os.getenv("PORT_FRONTEND", "3000")
DEBUG_HIL = os.getenv("DEBUG_HIL", "false")
BACKEND_HOST = os.getenv("BACKEND_HOST", "localhost")

app = FastAPI()

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single WebSocket client tracking
active_client: Optional[WebSocket] = None
active_client_lock = asyncio.Lock()

# Add a global flag to track if processing should be canceled
processing_canceled = False
processing_canceled_lock = asyncio.Lock()


async def _async_update_callback(message: Dict[str, Any]):
    global processing_canceled

    # Check if processing has been canceled
    async with processing_canceled_lock:
        if processing_canceled:
            # Skip sending updates if processing has been canceled
            return

    # Debug: Log message with media info
    role = message.get("role", "unknown")
    has_media = "media" in message and message["media"]
    media_count = len(message["media"]) if has_media else 0
    print(f"[WS] Sending message: role={role}, has_media={has_media}, media_count={media_count}")

    # Try to send message to active WebSocket client
    async with active_client_lock:
        if active_client:
            try:
                await active_client.send_json(message)
            except Exception:
                print("Client disconnected unexpectedly.")
        else:
            print("No active client to send to.")


def update_callback(message: Dict[str, Any]):
    # Needed for non-async context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_async_update_callback(message))
    loop.close()


agent = VisionAgent(verbose=True, callback=update_callback)


async def reset_cancellation_flag():
    global processing_canceled
    async with processing_canceled_lock:
        processing_canceled = False


def process_messages_background(agent: VisionAgent, messages: List[Dict[str, Any]]):
    global processing_canceled
    if processing_canceled:
        return

    for message in messages:
        if "media" in message and message["media"] is None:
            del message["media"]

    last_message = messages[-1] if messages else None
    if not last_message:
        return

    task = last_message.get("content", "")
    image_path = None
    media = last_message.get("media")
    if media and isinstance(media, list) and media:
        image_path = _save_media_to_temp(media[0])

    workspace_path = os.path.abspath(os.getenv("CHAT_WORKSPACE", "./chat_workspace"))
    current_dir = os.getcwd()

    try:
        os.chdir(workspace_path)
        result = agent.run(
            task=task,
            workspace=workspace_path,
            image_path=image_path,
        )
    finally:
        os.chdir(current_dir)

    update_callback({"role": "assistant", "content": str(result)})


class Message(BaseModel):
    role: str
    content: str
    media: Optional[List[str]] = None


def _save_media_to_temp(b64_media: str) -> str:
    if "," in b64_media:
        header, payload = b64_media.split(",", 1)
        media_type = header.split(";")[0].replace("data:", "")
    else:
        payload = b64_media
        media_type = "image/png"

    ext_map = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
    }
    ext = ext_map.get(media_type, ".png")

    media_bytes = base64.b64decode(payload)
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    temp_file.write(media_bytes)
    temp_file.flush()
    temp_file.close()
    return temp_file.name


@app.post("/chat/{version}")
async def chat(
    version: str, messages: List[Message], background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    await reset_cancellation_flag()

    if version not in ["v2", "v3"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported version: {version}. Supported versions are 'v2' and 'v3'.",
        )

    background_tasks.add_task(
        process_messages_background, agent, [m.model_dump() for m in messages]
    )
    return {"status": "Processing started"}


@app.post("/cancel")
async def cancel_processing():
    """Cancel any ongoing message processing."""
    global processing_canceled
    async with processing_canceled_lock:
        processing_canceled = True

    # Also clear the active websocket if possible
    async with active_client_lock:
        if active_client:
            try:
                # Send a cancellation message that the frontend can detect
                await active_client.send_json(
                    {"role": "system", "content": "Processing canceled by user."}
                )
            except Exception:
                pass

    return {"status": "Processing canceled"}


@app.post("/reset")
async def reset_session():
    """Reset the session state."""
    global processing_canceled

    async with processing_canceled_lock:
        processing_canceled = False

    return {"status": "Session reset successfully"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global active_client

    # Accept new connections, replacing any existing one
    async with active_client_lock:
        if active_client:
            # Close existing connection before accepting new one
            try:
                await active_client.close()
            except Exception:
                pass
            active_client = None

        # Accept the new connection
        await websocket.accept()
        active_client = websocket
        print(f"[WS] New client connected")

    try:
        while True:
            await websocket.receive_json()
    except WebSocketDisconnect:
        async with active_client_lock:
            if active_client == websocket:
                active_client = None
                print(f"[WS] Client disconnected")


@app.post("/send_message")
async def send_message(message: Message):
    await _async_update_callback(message.model_dump())
    return {"status": "sent"}
