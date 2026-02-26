# API 参考

本文档列出系统提供的所有 HTTP REST API 和 WebSocket 协议。

Gateway 默认监听 `https://localhost:8006`。

---

## HTTP REST API

### 健康与状态

#### GET /health

健康检查。

**响应**：
```json
{"status": "ok"}
```

#### GET /status

服务全局状态。

**响应**：
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

Worker 列表。

**响应**：
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

无状态 Chat 推理。每次请求完整 prefill，不复用 KV Cache。

**请求体**：
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

**多模态消息**：
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "描述这张图片"},
        {"type": "image", "data": "<base64>"},
        {"type": "audio", "data": "<base64 PCM float32>", "sample_rate": 16000},
        {"type": "video", "data": "<base64 video file>", "stack_frames": 1}
      ]
    }
  ]
}
```

> **注意**：包含视频内容时需设置 `omni_mode: true`，且仅支持 Chat 模式（不支持 Streaming）。

**响应**：
```json
{
  "text": "你好！有什么可以帮你的？",
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

停止正在进行的 Streaming 生成。

**请求体**：
```json
{"session_id": "stream_abc123"}
```

---

### 配置与预设

#### GET /api/frontend_defaults

获取前端默认配置。

**响应**：
```json
{
  "playback_delay_ms": 200,
  "chat_vocoder": "token2wav"
}
```

#### GET /api/presets

获取 System Prompt 预设列表。

**响应**：
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

### 参考音频管理

#### GET /api/default_ref_audio

获取默认参考音频信息。

#### GET /api/assets/ref_audio

列出所有参考音频。

**响应**：
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

上传参考音频。上传时自动归一化为 16kHz 单声道 16-bit WAV。

**请求体**：
```json
{
  "name": "My Voice",
  "audio_data": "<base64 audio>"
}
```

#### DELETE /api/assets/ref_audio/{audio_id}

删除指定参考音频。

---

### 队列管理

#### GET /api/queue

获取队列状态快照。

**响应**：
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

查询指定排队凭证状态。

#### DELETE /api/queue/{ticket_id}

取消排队。

---

### ETA 配置

#### GET /api/config/eta

获取 ETA 配置。

**响应**：
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

更新 ETA 配置。

---

### KV Cache

#### GET /api/cache

查询所有 Worker 的 KV Cache 状态。

---

### 会话管理

#### GET /api/sessions/{session_id}

获取会话元数据。

**响应**：
```json
{
  "session_id": "omni_abc123",
  "type": "omni_duplex",
  "created_at": "2026-02-24T10:00:00Z",
  "config": {}
}
```

#### GET /api/sessions/{session_id}/recording

获取会话录制时间线数据。

#### GET /api/sessions/{session_id}/assets/{relative_path}

获取会话资源文件（音频/视频 chunk 等）。

#### GET /api/sessions/{session_id}/download

打包下载整个会话。

#### POST /api/sessions/{session_id}/upload-recording

上传前端录制的音频/视频文件。大小限制 200MB。

---

### 应用管理

#### GET /api/apps

获取已启用的应用列表（前端用）。

#### GET /api/admin/apps

获取所有应用列表（含启用状态，Admin 用）。

#### PUT /api/admin/apps

切换应用启用状态。

**请求体**：
```json
{
  "app_id": "omni",
  "enabled": false
}
```

---

## WebSocket 协议

### Streaming 协议

**连接**：`wss://localhost:8006/ws/streaming/{session_id}`

#### 客户端 → 服务端

| 消息类型 | 字段 | 说明 |
|---------|------|------|
| `prefill` | `messages`, `generation`, `streaming`, `use_tts_template`, `enable_thinking` | 预填充，发送消息历史 |
| `generate` | — | 开始流式生成 |
| `stop` | — | 停止生成 |

**prefill 示例**：
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

#### 服务端 → 客户端

| 消息类型 | 字段 | 说明 |
|---------|------|------|
| `queued` | `ticket_id`, `position`, `eta_seconds` | 排入队列 |
| `queue_update` | `position`, `eta_seconds` | 队列位置更新 |
| `queue_done` | — | 离开队列，开始处理 |
| `prefill_done` | `prompt` | 预填充完成 |
| `chunk` | `chunk_index`, `text_delta`, `audio_data`, `is_final` | 流式输出 chunk |
| `done` | `full_text`, `total_chunks`, `total_duration_ms` | 生成完成 |
| `error` | `message` | 错误 |

---

### Duplex 协议

**连接**：`wss://localhost:8006/ws/duplex/{session_id}`

#### 客户端 → 服务端

| 消息类型 | 字段 | 说明 |
|---------|------|------|
| `prepare` | `system_prompt`, `config`, `ref_audio_path` | 初始化会话 |
| `audio_chunk` | `audio` (Base64 PCM float32) | 发送音频块 |
| `video_frame` | `frame` (Base64 JPEG) | 发送视频帧（仅 Omni） |
| `pause` | — | 暂停会话 |
| `resume` | — | 恢复会话 |
| `stop` | — | 停止会话 |
| `client_diagnostic` | `metrics` | 客户端诊断信息 |

**prepare 示例**：
```json
{
  "type": "prepare",
  "prefix_system_prompt": "你是一个有趣的助手。",
  "config": {
    "generate_audio": true,
    "chunk_ms": 1000,
    "temperature": 0.7,
    "force_listen_count": 3
  },
  "ref_audio_path": "assets/ref_audio/ref_minicpm_signature.wav"
}
```

**audio_chunk 示例**：
```json
{
  "type": "audio_chunk",
  "audio": "<base64 PCM float32, 16kHz, 1s>"
}
```

#### 服务端 → 客户端

| 消息类型 | 字段 | 说明 |
|---------|------|------|
| `queued` | `ticket_id`, `position`, `eta_seconds` | 排入队列 |
| `queue_update` | `position`, `eta_seconds` | 队列位置更新 |
| `queue_done` | — | 离开队列 |
| `prepared` | — | 准备完成 |
| `result` | `is_listen`, `text`, `audio_data`, `end_of_turn`, `cost_all_ms` | 单步结果 |
| `paused` | — | 已暂停 |
| `resumed` | — | 已恢复 |
| `stopped` | `session_id` | 已停止 |
| `timeout` | — | 暂停超时 |
| `error` | `message` | 错误 |

**result 示例**：
```json
{
  "type": "result",
  "is_listen": false,
  "text": "你好",
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

## 音频格式规范

| 方向 | 格式 | 采样率 | 编码 |
|------|------|--------|------|
| 客户端 → 服务端 | PCM Float32 | 16000 Hz | Base64 |
| 服务端 → 客户端 | PCM Float32 | 24000 Hz | Base64 |

- 输入音频必须为 **16kHz 单声道**
- 输出音频为 **24kHz 单声道**
- Base64 编码的是 Float32 数组的原始字节
