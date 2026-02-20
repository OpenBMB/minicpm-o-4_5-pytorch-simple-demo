"""Mock Worker — 轻量模拟 Worker，无 GPU，用于集成测试

模拟 Worker 的 HTTP + WebSocket 端点：
- GET  /health          → 健康检查
- POST /chat            → 模拟 Chat 推理（可配延迟）
- POST /streaming/stop  → 停止 Streaming
- POST /clear_cache     → 清缓存
- GET  /cache_info      → 缓存信息
- WS   /ws/streaming    → 模拟 Streaming（prefill_done → chunk × N → done）
- WS   /ws/duplex       → 模拟 Duplex（prepared → result × N → stopped）

启动方式：
    PYTHONPATH=. .venv/base/bin/python tests/mock_worker.py --port 22400

可配延迟（通过 CLI 参数或运行时 POST /config）：
    --chat-delay     Chat 推理延迟秒数（默认 0.5）
    --stream-delay   Streaming 每个 chunk 延迟秒数（默认 0.1）
    --duplex-delay   Duplex 每个 result 延迟秒数（默认 0.1）
"""

import argparse
import asyncio
import json
import logging
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mock_worker")

# ============ 全局状态 ============

WORKER_STATUS = "idle"  # idle | busy_chat | busy_streaming | duplex_active
TOTAL_REQUESTS = 0
CURRENT_SESSION_ID: Optional[str] = None

# 可配延迟（秒）
CHAT_DELAY = 0.5
STREAM_CHUNK_DELAY = 0.1
STREAM_CHUNKS = 5
DUPLEX_CHUNK_DELAY = 0.1
DUPLEX_CHUNKS = 10
GPU_ID = 0


def create_app() -> FastAPI:
    """创建 Mock Worker FastAPI 应用"""
    app = FastAPI(title="Mock Worker")

    # ============ Health ============

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "worker_status": WORKER_STATUS,
            "gpu_id": GPU_ID,
            "model_loaded": True,
            "current_session_id": CURRENT_SESSION_ID,
            "total_requests": TOTAL_REQUESTS,
            "avg_inference_time_ms": 100.0,
            "kv_cache_length": 0,
        }

    # ============ Chat ============

    @app.post("/chat")
    async def chat(request: dict):
        global WORKER_STATUS, TOTAL_REQUESTS
        WORKER_STATUS = "busy_chat"
        TOTAL_REQUESTS += 1

        await asyncio.sleep(CHAT_DELAY)

        WORKER_STATUS = "idle"
        return {
            "text": f"[Mock] Chat response (delay={CHAT_DELAY}s)",
            "success": True,
            "audio_data": None,
            "token_stats": {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
            },
        }

    # ============ Streaming Stop ============

    @app.post("/streaming/stop")
    async def streaming_stop():
        return {"success": True, "message": "Stop signal sent"}

    # ============ Cache ============

    @app.post("/clear_cache")
    async def clear_cache():
        return {"success": True, "message": "Cache cleared"}

    @app.get("/cache_info")
    async def cache_info():
        return {"status": "no_cache", "note": "mock worker"}

    # ============ Streaming WebSocket ============

    @app.websocket("/ws/streaming")
    async def streaming_ws(ws: WebSocket):
        global WORKER_STATUS, TOTAL_REQUESTS, CURRENT_SESSION_ID
        await ws.accept()
        logger.info("Streaming WS connected")

        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                msg_type = msg.get("type", "")

                if msg_type == "prefill":
                    WORKER_STATUS = "busy_streaming"
                    CURRENT_SESSION_ID = msg.get("session_id")
                    TOTAL_REQUESTS += 1

                    messages = msg.get("messages", [])
                    await ws.send_json({
                        "type": "prefill_done",
                        "prompt_length": len(messages) * 10,
                        "message_count": len(messages),
                        "cached_tokens": 0,
                        "input_tokens": len(messages) * 10,
                    })

                elif msg_type == "generate":
                    full_text = ""
                    for i in range(STREAM_CHUNKS):
                        await asyncio.sleep(STREAM_CHUNK_DELAY)
                        chunk_text = f"chunk_{i} "
                        full_text += chunk_text
                        await ws.send_json({
                            "type": "chunk",
                            "text_delta": chunk_text,
                            "audio_data": None,
                            "is_first_chunk": i == 0,
                        })

                    await ws.send_json({
                        "type": "done",
                        "elapsed_ms": int(STREAM_CHUNKS * STREAM_CHUNK_DELAY * 1000),
                        "session_id": CURRENT_SESSION_ID,
                        "stopped": False,
                        "token_stats": {
                            "input_tokens": 10,
                            "output_tokens": STREAM_CHUNKS * 5,
                        },
                    })

                    WORKER_STATUS = "idle"
                    CURRENT_SESSION_ID = None

                elif msg_type == "stop":
                    WORKER_STATUS = "idle"
                    CURRENT_SESSION_ID = None

                elif msg_type == "close":
                    break

        except WebSocketDisconnect:
            logger.info("Streaming WS disconnected")
        except Exception as e:
            logger.error(f"Streaming WS error: {e}")
        finally:
            WORKER_STATUS = "idle"
            CURRENT_SESSION_ID = None

    # ============ Duplex WebSocket ============

    @app.websocket("/ws/duplex")
    async def duplex_ws(ws: WebSocket):
        global WORKER_STATUS, TOTAL_REQUESTS, CURRENT_SESSION_ID
        await ws.accept()

        session_id = ws.query_params.get("session_id", "unknown")
        CURRENT_SESSION_ID = session_id
        logger.info(f"Duplex WS connected: session={session_id}")

        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                msg_type = msg.get("type", "")

                if msg_type == "prepare":
                    WORKER_STATUS = "duplex_active"
                    TOTAL_REQUESTS += 1
                    await asyncio.sleep(0.1)
                    await ws.send_json({
                        "type": "prepared",
                        "prompt_length": 50,
                    })

                elif msg_type == "audio_chunk":
                    await asyncio.sleep(DUPLEX_CHUNK_DELAY)
                    await ws.send_json({
                        "type": "result",
                        "is_listen": True,
                        "text": "",
                        "audio_base64": None,
                        "end_of_turn": False,
                        "wall_clock_ms": int(DUPLEX_CHUNK_DELAY * 1000),
                        "kv_cache_length": 100,
                    })

                elif msg_type == "pause":
                    WORKER_STATUS = "duplex_paused"
                    await ws.send_json({"type": "paused", "timeout": 60})

                elif msg_type == "resume":
                    WORKER_STATUS = "duplex_active"
                    await ws.send_json({"type": "resumed"})

                elif msg_type == "stop":
                    await ws.send_json({"type": "stopped"})
                    break

        except WebSocketDisconnect:
            logger.info(f"Duplex WS disconnected: session={session_id}")
        except Exception as e:
            logger.error(f"Duplex WS error: {e}")
        finally:
            WORKER_STATUS = "idle"
            CURRENT_SESSION_ID = None

    return app


def main():
    global CHAT_DELAY, STREAM_CHUNK_DELAY, STREAM_CHUNKS, DUPLEX_CHUNK_DELAY, DUPLEX_CHUNKS, GPU_ID

    parser = argparse.ArgumentParser(description="Mock Worker for integration testing")
    parser.add_argument("--port", type=int, default=22400)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--chat-delay", type=float, default=0.5)
    parser.add_argument("--stream-delay", type=float, default=0.1)
    parser.add_argument("--stream-chunks", type=int, default=5)
    parser.add_argument("--duplex-delay", type=float, default=0.1)
    parser.add_argument("--duplex-chunks", type=int, default=10)
    args = parser.parse_args()

    GPU_ID = args.gpu_id
    CHAT_DELAY = args.chat_delay
    STREAM_CHUNK_DELAY = args.stream_delay
    STREAM_CHUNKS = args.stream_chunks
    DUPLEX_CHUNK_DELAY = args.duplex_delay
    DUPLEX_CHUNKS = args.duplex_chunks

    app = create_app()
    logger.info(
        f"Mock Worker starting on {args.host}:{args.port} "
        f"(chat={CHAT_DELAY}s, stream={STREAM_CHUNK_DELAY}s×{STREAM_CHUNKS}, "
        f"duplex={DUPLEX_CHUNK_DELAY}s×{DUPLEX_CHUNKS})"
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
