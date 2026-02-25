# Worker API Reference

The Worker is the system's inference execution unit. Each Worker exclusively occupies one GPU and holds a single `UnifiedProcessor` instance. For detailed processing flows and implementation details, see [Streaming Mode Details](architecture/streaming.html) and [Duplex Mode Details](architecture/duplex.html).

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
| `streaming_prefill(request)` | Streaming prefill |
| `streaming_init_tts(ref_audio_data)` | Initialize Streaming TTS |
| `streaming_generate(session_id, ...)` | Streaming generation (Generator) |
| `streaming_complete_turn(...)` | Complete one Streaming turn |
| `reset_streaming_session()` | Reset KV Cache (called when Gateway indicates clear_kv_cache) |
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
| `/ws/streaming` | WS | Streaming conversational session |
| `/ws/duplex` | WS | Duplex full-duplex session |
| `/cache_info` | GET | KV Cache information |
| `/clear_cache` | POST | Clear KV Cache |
