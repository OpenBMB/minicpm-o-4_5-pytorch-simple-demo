# Worker API Reference

The Worker is the system's inference execution unit. Each Worker exclusively occupies one GPU and holds a single `UnifiedProcessor` instance. For detailed processing flows and implementation details, see [ChatView Mode Details](architecture/streaming.html) and [Duplex Mode Details](architecture/duplex.html).

## WorkerStatus Enum

| Status | Description | Can Accept Tasks |
|--------|-------------|------------------|
| `LOADING` | Model loading | No |
| `IDLE` | Idle | Yes |
| `BUSY_STREAMING` | Executing Streaming | No (allowed within the same connection) |
| `DUPLEX_ACTIVE` | Duplex session active | No |
| `DUPLEX_PAUSED` | Duplex session paused | No |
| `ERROR` | Error | No |

## MiniCPMOWorker Methods

| Method | Description |
|--------|-------------|
| `load_model()` | Synchronously load model, initialize UnifiedProcessor |
| `chat_prefill(session_id, msgs, ...)` | Chat prefill (one-shot KV Cache fill) |
| `chat_non_streaming_generate(session_id, ...)` | Chat non-streaming generate (HF generate + TTS) |
| `chat_streaming_generate(session_id, ...)` | Chat streaming generate (yield StreamingChunk) |
| `duplex_prepare(...)` | Duplex preparation (system prompt + reference audio) |
| `duplex_prefill(...)` | Duplex prefill (audio + video frames) |
| `duplex_generate(force_listen)` | Duplex single-step generation |
| `duplex_finalize()` | Duplex deferred finalize (feed terminator + sliding window maintenance) |
| `duplex_stop()` | Stop Duplex |
| `duplex_cleanup()` | Clean up Duplex resources, release GPU memory |

## HTTP / WebSocket Endpoints

| Endpoint | Method | Function |
|----------|--------|----------|
| `/health` | GET | Health check (returns status, gpu_id, model_loaded, kv_cache_length) |
| `/streaming/stop` | POST | Stop Streaming |
| `/ws/chat` | WS | Chat unified interface (streaming/non-streaming) |
| `/ws/duplex` | WS | Duplex full-duplex session |
| `/cache_info` | GET | KV Cache information |
| `/clear_cache` | POST | Clear KV Cache |
