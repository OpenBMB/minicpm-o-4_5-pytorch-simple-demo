# API Reference

This document lists all HTTP REST APIs and WebSocket protocols provided by the system.

The Gateway listens on `https://localhost:8006` by default.

---

## HTTP REST API

### Health & Status

#### GET /health

Health check.

**Response**:
```json
{"status": "ok"}
```

#### GET /status

Global service status.

**Response**:
```json
{
  "total_workers": 4,
  "idle_workers": 2,
  "busy_workers": 2,
  "queue_length": 3,
  "workers": [...]
}
```

#### GET /workers

Worker list.

**Response**:
```json
{
  "workers": [
    {
      "url": "http://localhost:22400",
      "index": 0,
      "status": "idle",
      "current_task": null,
      "current_session_id": null,
      "cached_hash": "abc123",
      "busy_since": null
    }
  ]
}
```

---

### Chat API

#### POST /api/chat

Stateless Chat inference. Each request performs a full prefill without reusing the KV Cache.

**Request Body**:
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "generation": {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.8
  },
  "tts": {
    "enabled": true,
    "mode": "audio_assistant",
    "ref_audio_path": "assets/ref_audio/ref_minicpm_signature.wav"
  }
}
```

**Multimodal Messages**:
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image"},
        {"type": "image", "data": "<base64>"},
        {"type": "audio", "data": "<base64 PCM float32>", "sample_rate": 16000}
      ]
    }
  ]
}
```

**Response**:
```json
{
  "text": "Hello! How can I help you?",
  "audio_data": "<base64>",
  "audio_sample_rate": 24000,
  "tokens_generated": 15,
  "duration_ms": 234.5,
  "token_stats": {
    "cached_tokens": 0,
    "input_tokens": 42,
    "generated_tokens": 15,
    "total_tokens": 57
  },
  "recording_session_id": "chat_abc123",
  "success": true,
  "error": null
}
```

#### POST /api/streaming/stop

Stop an ongoing streaming generation.

**Request Body**:
```json
{"session_id": "stream_abc123"}
```

---

### Configuration & Presets

#### GET /api/frontend_defaults

Get frontend default configuration.

**Response**:
```json
{
  "playback_delay_ms": 200,
  "chat_vocoder": "token2wav"
}
```

#### GET /api/presets

Get the list of System Prompt presets.

**Response**:
```json
[
  {
    "id": "default_en",
    "name": "English Assistant",
    "system_prompt": "You are a helpful assistant."
  }
]
```

---

### Reference Audio Management

#### GET /api/default_ref_audio

Get default reference audio information.

#### GET /api/assets/ref_audio

List all reference audios.

**Response**:
```json
{
  "audios": [
    {
      "id": "ref_001",
      "name": "Default Voice",
      "source": "builtin",
      "file_path": "assets/ref_audio/ref_minicpm_signature.wav"
    }
  ]
}
```

#### POST /api/assets/ref_audio/external

Upload a reference audio. Uploaded audio is automatically normalized to 16kHz mono 16-bit WAV.

**Request Body**:
```json
{
  "name": "My Voice",
  "audio_data": "<base64 audio>"
}
```

#### DELETE /api/assets/ref_audio/{audio_id}

Delete a specific reference audio.

---

### Queue Management

#### GET /api/queue

Get a snapshot of the queue status.

**Response**:
```json
{
  "queue_length": 3,
  "entries": [
    {
      "ticket_id": "tk_001",
      "position": 1,
      "task_type": "streaming",
      "eta_seconds": 15.0
    }
  ],
  "running": [
    {
      "worker_url": "http://localhost:22400",
      "task_type": "streaming",
      "session_id": "stream_xyz",
      "started_at": "2026-02-24T10:30:00Z",
      "elapsed_s": 5.2
    }
  ]
}
```

#### GET /api/queue/{ticket_id}

Query the status of a specific queue ticket.

#### DELETE /api/queue/{ticket_id}

Cancel a queued request.

---

### ETA Configuration

#### GET /api/config/eta

Get ETA configuration.

**Response**:
```json
{
  "eta_chat_s": 15.0,
  "eta_streaming_s": 20.0,
  "eta_audio_duplex_s": 120.0,
  "eta_omni_duplex_s": 90.0,
  "eta_ema_alpha": 0.3,
  "eta_ema_min_samples": 3
}
```

#### PUT /api/config/eta

Update ETA configuration.

---

### KV Cache

#### GET /api/cache

Query the KV Cache status of all Workers.

---

### Session Management

#### GET /api/sessions/{session_id}

Get session metadata.

**Response**:
```json
{
  "session_id": "omni_abc123",
  "type": "omni_duplex",
  "created_at": "2026-02-24T10:00:00Z",
  "config": {}
}
```

#### GET /api/sessions/{session_id}/recording

Get session recording timeline data.

#### GET /api/sessions/{session_id}/assets/{relative_path}

Get session asset files (audio/video chunks, etc.).

#### GET /api/sessions/{session_id}/download

Download the entire session as a package.

#### POST /api/sessions/{session_id}/upload-recording

Upload frontend-recorded audio/video files. Size limit: 200MB.

---

### App Management

#### GET /api/apps

Get the list of enabled apps (for frontend use).

#### GET /api/admin/apps

Get the list of all apps (including enabled status, for Admin use).

#### PUT /api/admin/apps

Toggle app enabled status.

**Request Body**:
```json
{
  "app_id": "omni",
  "enabled": false
}
```

---

## WebSocket Protocol

### Streaming Protocol

**Connection**: `wss://localhost:8006/ws/streaming/{session_id}`

#### Client → Server

| Message Type | Fields | Description |
|---------|------|------|
| `prefill` | `messages`, `generation`, `streaming`, `use_tts_template`, `enable_thinking` | Prefill: send message history |
| `generate` | — | Start streaming generation |
| `stop` | — | Stop generation |

**prefill Example**:
```json
{
  "type": "prefill",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "generation": {"max_new_tokens": 512},
  "streaming": {
    "generate_audio": true,
    "ref_audio_path": "assets/ref_audio/ref_minicpm_signature.wav"
  }
}
```

#### Server → Client

| Message Type | Fields | Description |
|---------|------|------|
| `queued` | `ticket_id`, `position`, `eta_seconds` | Enqueued |
| `queue_update` | `position`, `eta_seconds` | Queue position update |
| `queue_done` | — | Left queue, processing started |
| `prefill_done` | `prompt` | Prefill complete |
| `chunk` | `chunk_index`, `text_delta`, `audio_data`, `is_final` | Streaming output chunk |
| `done` | `full_text`, `total_chunks`, `total_duration_ms` | Generation complete |
| `error` | `message` | Error |

---

### Duplex Protocol

**Connection**: `wss://localhost:8006/ws/duplex/{session_id}`

#### Client → Server

| Message Type | Fields | Description |
|---------|------|------|
| `prepare` | `system_prompt`, `config`, `ref_audio_path` | Initialize session |
| `audio_chunk` | `audio` (Base64 PCM float32) | Send audio chunk |
| `video_frame` | `frame` (Base64 JPEG) | Send video frame (Omni only) |
| `pause` | — | Pause session |
| `resume` | — | Resume session |
| `stop` | — | Stop session |
| `client_diagnostic` | `metrics` | Client diagnostic information |

**prepare Example**:
```json
{
  "type": "prepare",
  "prefix_system_prompt": "You are a fun assistant.",
  "config": {
    "generate_audio": true,
    "chunk_ms": 1000,
    "temperature": 0.7,
    "force_listen_count": 3
  },
  "ref_audio_path": "assets/ref_audio/ref_minicpm_signature.wav"
}
```

**audio_chunk Example**:
```json
{
  "type": "audio_chunk",
  "audio": "<base64 PCM float32, 16kHz, 1s>"
}
```

#### Server → Client

| Message Type | Fields | Description |
|---------|------|------|
| `queued` | `ticket_id`, `position`, `eta_seconds` | Enqueued |
| `queue_update` | `position`, `eta_seconds` | Queue position update |
| `queue_done` | — | Left queue |
| `prepared` | — | Preparation complete |
| `result` | `is_listen`, `text`, `audio_data`, `end_of_turn`, `cost_all_ms` | Single-step result |
| `paused` | — | Paused |
| `resumed` | — | Resumed |
| `stopped` | `session_id` | Stopped |
| `timeout` | — | Pause timeout |
| `error` | `message` | Error |

**result Example**:
```json
{
  "type": "result",
  "is_listen": false,
  "text": "Hello",
  "audio_data": "<base64, 24kHz>",
  "end_of_turn": false,
  "current_time": 5000,
  "cost_llm_ms": 45.2,
  "cost_tts_ms": 12.3,
  "cost_all_ms": 78.5,
  "n_tokens": 3,
  "server_send_ts": 1708771200.123
}
```

---

## Audio Format Specification

| Direction | Format | Sample Rate | Encoding |
|------|------|--------|------|
| Client → Server | PCM Float32 | 16000 Hz | Base64 |
| Server → Client | PCM Float32 | 24000 Hz | Base64 |

- Input audio must be **16kHz mono**
- Output audio is **24kHz mono**
- Base64 encodes the raw bytes of the Float32 array
