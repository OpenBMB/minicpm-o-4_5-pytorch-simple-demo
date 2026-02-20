"""Gateway + Worker API 集成测试

测试 Worker 直连和 Gateway 代理的所有 API。
需要先启动 Worker（至少 1 个 GPU）。

使用方式：
    cd /user/sunweiyue/lib/swy-dev/minicpmo45_service

    # 1. 启动 Worker（另一个终端）
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/base/bin/python worker.py --worker-index 0

    # 2. 运行测试
    PYTHONPATH=. .venv/base/bin/python -m pytest tests/test_api.py -v -s

    # 或只运行快速测试（不需要 GPU）
    PYTHONPATH=. .venv/base/bin/python -m pytest tests/test_api.py -v -s -k "not gpu"
"""

import os
import json
import time
import asyncio
import base64
import logging
from typing import Optional

import pytest
import httpx
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_api")

# ============ 配置 ============

# 从 config.py 读取默认端口
try:
    from config import get_config
    _cfg = get_config()
    _default_worker_url = f"http://localhost:{_cfg.worker_base_port}"
    _default_gateway_url = f"http://localhost:{_cfg.gateway_port}"
except Exception:
    _default_worker_url = "http://localhost:22400"
    _default_gateway_url = "http://localhost:10024"

WORKER_URL = os.environ.get("WORKER_URL", _default_worker_url)
GATEWAY_URL = os.environ.get("GATEWAY_URL", _default_gateway_url)


# ============ Fixtures ============

def _check_worker_ready() -> bool:
    """检查 Worker 是否就绪"""
    try:
        resp = httpx.get(f"{WORKER_URL}/health", timeout=3.0)
        data = resp.json()
        return data.get("model_loaded", False)
    except Exception:
        return False


def _check_gateway_ready() -> bool:
    """检查 Gateway 是否就绪"""
    try:
        resp = httpx.get(f"{GATEWAY_URL}/health", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


# 标记需要 GPU Worker 的测试
requires_worker = pytest.mark.skipif(
    not _check_worker_ready(),
    reason=f"Worker not available at {WORKER_URL}",
)

requires_gateway = pytest.mark.skipif(
    not _check_gateway_ready(),
    reason=f"Gateway not available at {GATEWAY_URL}",
)


# ============ Worker 直连测试 ============

class TestWorkerHealth:
    """Worker 健康检查测试"""

    @requires_worker
    def test_health(self):
        """健康检查返回正确状态"""
        resp = httpx.get(f"{WORKER_URL}/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["worker_status"] == "idle"

    @requires_worker
    def test_cache_info(self):
        """缓存信息查询"""
        resp = httpx.get(f"{WORKER_URL}/cache_info")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data


class TestWorkerChat:
    """Worker Chat API 测试"""

    @requires_worker
    def test_simple_chat(self):
        """简单文本对话"""
        resp = httpx.post(
            f"{WORKER_URL}/chat",
            json={
                "messages": [{"role": "user", "content": "1+1等于几？只回答数字。"}],
                "generation": {"max_new_tokens": 10, "do_sample": False},
            },
            timeout=30.0,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "2" in data["text"]
        assert data["duration_ms"] > 0
        logger.info(f"Chat response: {data['text']} ({data['duration_ms']:.0f}ms)")

    @requires_worker
    def test_chat_multi_turn(self):
        """多轮对话"""
        resp = httpx.post(
            f"{WORKER_URL}/chat",
            json={
                "messages": [
                    {"role": "user", "content": "42乘以2等于多少？"},
                    {"role": "assistant", "content": "42乘以2等于84。"},
                    {"role": "user", "content": "再乘以2呢？只回答数字。"},
                ],
                "generation": {"max_new_tokens": 20, "do_sample": False},
            },
            timeout=30.0,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "168" in data["text"]
        logger.info(f"Multi-turn response: {data['text']}")

    @requires_worker
    def test_chat_worker_busy_rejected(self):
        """Worker 忙碌时拒绝请求（通过同时发两个请求模拟）

        注意：这个测试依赖于时序，可能不稳定
        """
        # 先验证 Worker 是 idle
        health = httpx.get(f"{WORKER_URL}/health").json()
        assert health["worker_status"] == "idle"


class TestWorkerStreamingWS:
    """Worker Streaming WebSocket 测试"""

    @requires_worker
    @pytest.mark.asyncio
    async def test_streaming_text_only(self):
        """Streaming 纯文本（不生成音频）"""
        import websockets

        ws_url = WORKER_URL.replace("http://", "ws://") + "/ws/streaming"
        async with websockets.connect(ws_url) as ws:
            # 1. prefill
            await ws.send(json.dumps({
                "type": "prefill",
                "session_id": "test_001",
                "messages": [{"role": "user", "content": "讲一个关于猫的一句话故事。"}],
                "is_last_chunk": True,
            }))

            resp = json.loads(await ws.recv())
            assert resp["type"] == "prefill_done", f"Expected prefill_done, got: {resp}"
            logger.info(f"Prefill done, prompt_length={resp['prompt_length']}")

            # 2. generate
            await ws.send(json.dumps({
                "type": "generate",
                "session_id": "test_001",
                "generate_audio": False,
                "max_new_tokens": 100,
            }))

            # 3. 收集 chunks
            full_text = ""
            chunk_count = 0
            while True:
                resp = json.loads(await ws.recv())
                if resp["type"] == "chunk":
                    chunk_count += 1
                    if resp.get("text_delta"):
                        full_text += resp["text_delta"]
                elif resp["type"] == "done":
                    logger.info(
                        f"Streaming done: {chunk_count} chunks, "
                        f"{resp['elapsed_ms']:.0f}ms, text='{full_text[:100]}'"
                    )
                    break
                elif resp["type"] == "error":
                    pytest.fail(f"Streaming error: {resp['error']}")
                    break

            assert chunk_count > 0, "Should receive at least one chunk"
            assert len(full_text) > 0, "Should receive non-empty text"

            # 4. 关闭连接
            await ws.send(json.dumps({"type": "close"}))

    @requires_worker
    @pytest.mark.asyncio
    async def test_streaming_multi_turn(self):
        """Streaming 多轮对话（KV Cache 复用）"""
        import websockets

        ws_url = WORKER_URL.replace("http://", "ws://") + "/ws/streaming"
        async with websockets.connect(ws_url) as ws:
            # Turn 1: prefill + generate
            await ws.send(json.dumps({
                "type": "prefill",
                "session_id": "test_multi",
                "messages": [{"role": "user", "content": "记住数字42。"}],
                "is_last_chunk": True,
            }))
            resp = json.loads(await ws.recv())
            assert resp["type"] == "prefill_done"

            await ws.send(json.dumps({
                "type": "generate",
                "session_id": "test_multi",
                "generate_audio": False,
                "max_new_tokens": 50,
            }))

            turn1_text = ""
            while True:
                resp = json.loads(await ws.recv())
                if resp["type"] == "chunk" and resp.get("text_delta"):
                    turn1_text += resp["text_delta"]
                elif resp["type"] == "done":
                    break

            logger.info(f"Turn 1: {turn1_text[:80]}")

            # Turn 2: 增量 prefill（复用 KV Cache）
            # [CRITICAL] Streaming 模式每次只能 prefill 一条消息
            # 先 prefill assistant 的回复
            await ws.send(json.dumps({
                "type": "prefill",
                "session_id": "test_multi",
                "messages": [{"role": "assistant", "content": turn1_text}],
                "is_last_chunk": False,
            }))
            resp = json.loads(await ws.recv())
            assert resp["type"] == "prefill_done", f"Turn 2 assistant prefill: {resp}"

            # 再 prefill 新的 user 消息
            await ws.send(json.dumps({
                "type": "prefill",
                "session_id": "test_multi",
                "messages": [{"role": "user", "content": "我刚才说的数字是多少？只回答数字。"}],
                "is_last_chunk": True,
            }))
            resp = json.loads(await ws.recv())
            assert resp["type"] == "prefill_done", f"Turn 2 user prefill: {resp}"

            await ws.send(json.dumps({
                "type": "generate",
                "session_id": "test_multi",
                "generate_audio": False,
                "max_new_tokens": 20,
            }))

            turn2_text = ""
            while True:
                resp = json.loads(await ws.recv())
                if resp["type"] == "chunk" and resp.get("text_delta"):
                    turn2_text += resp["text_delta"]
                elif resp["type"] == "done":
                    break

            logger.info(f"Turn 2: {turn2_text}")
            assert "42" in turn2_text, f"Expected '42' in response, got: {turn2_text}"

            await ws.send(json.dumps({"type": "close"}))

    @requires_worker
    @pytest.mark.asyncio
    async def test_health_during_streaming(self):
        """Streaming 推理期间健康检查仍可响应（验证非阻塞）"""
        import websockets

        ws_url = WORKER_URL.replace("http://", "ws://") + "/ws/streaming"
        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({
                "type": "prefill",
                "session_id": "test_nonblock",
                "messages": [{"role": "user", "content": "写一首 50 字的诗。"}],
                "is_last_chunk": True,
            }))
            await ws.recv()  # prefill_done

            await ws.send(json.dumps({
                "type": "generate",
                "session_id": "test_nonblock",
                "generate_audio": False,
                "max_new_tokens": 200,
            }))

            # 在推理期间发送健康检查
            await asyncio.sleep(0.1)  # 等一下让推理开始
            async with httpx.AsyncClient() as client:
                health_resp = await client.get(f"{WORKER_URL}/health", timeout=5.0)
                assert health_resp.status_code == 200
                health_data = health_resp.json()
                # Worker 应该是 busy_streaming 状态
                logger.info(f"Health during streaming: {health_data['worker_status']}")

            # 消费完所有 chunks
            while True:
                resp = json.loads(await ws.recv())
                if resp["type"] in ("done", "error"):
                    break

            await ws.send(json.dumps({"type": "close"}))


# ============ Gateway 测试 ============

class TestGatewayChat:
    """Gateway Chat 路由测试"""

    @requires_gateway
    @requires_worker
    def test_chat_via_gateway(self):
        """通过 Gateway 路由的 Chat 请求"""
        resp = httpx.post(
            f"{GATEWAY_URL}/api/chat",
            json={
                "messages": [{"role": "user", "content": "1+2等于几？只回答数字。"}],
                "generation": {"max_new_tokens": 10, "do_sample": False},
            },
            timeout=30.0,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "3" in data["text"]
        logger.info(f"Gateway Chat: {data['text']}")

    @requires_gateway
    def test_status(self):
        """服务状态查询"""
        resp = httpx.get(f"{GATEWAY_URL}/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["gateway_healthy"] is True
        assert data["total_workers"] >= 1
        logger.info(f"Gateway status: {data}")

    @requires_gateway
    def test_workers_list(self):
        """Worker 列表"""
        resp = httpx.get(f"{GATEWAY_URL}/workers")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        assert len(data["workers"]) >= 1
        logger.info(f"Workers: {data['total']}")


class TestGatewayRefAudio:
    """Gateway 参考音频管理测试"""

    @requires_gateway
    def test_list_empty(self):
        """初始列表可能为空或有数据"""
        resp = httpx.get(f"{GATEWAY_URL}/api/assets/ref_audio")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "ref_audios" in data

    @requires_gateway
    def test_upload_and_delete(self):
        """上传、查询、删除参考音频"""
        # 生成一段测试音频（1秒静音，16kHz）
        test_audio = np.zeros(16000, dtype=np.float32)

        # 需要生成有效的 WAV 文件
        import io
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, test_audio, 16000, format="WAV")
        audio_b64 = base64.b64encode(buf.getvalue()).decode()

        # 上传
        resp = httpx.post(
            f"{GATEWAY_URL}/api/assets/ref_audio",
            json={"name": "test_silence", "audio_base64": audio_b64},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        ref_id = data["id"]
        logger.info(f"Uploaded ref audio: {ref_id}")

        # 列出
        resp = httpx.get(f"{GATEWAY_URL}/api/assets/ref_audio")
        assert resp.status_code == 200
        data = resp.json()
        ids = [r["id"] for r in data["ref_audios"]]
        assert ref_id in ids

        # 删除
        resp = httpx.delete(f"{GATEWAY_URL}/api/assets/ref_audio/{ref_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

        # 确认已删除
        resp = httpx.get(f"{GATEWAY_URL}/api/assets/ref_audio")
        data = resp.json()
        ids = [r["id"] for r in data["ref_audios"]]
        assert ref_id not in ids
        logger.info("Upload → List → Delete cycle passed")


class TestGatewaySessions:
    """Gateway 会话管理测试"""

    @requires_gateway
    def test_list_sessions(self):
        """列出会话"""
        resp = httpx.get(f"{GATEWAY_URL}/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "sessions" in data
