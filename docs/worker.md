# Worker 模块详解

Worker 是系统的推理执行单元，每个 Worker 独占一张 GPU，持有一个 `UnifiedProcessor` 实例，提供 Chat / Streaming / Duplex 三种推理模式。

## 模块结构

```
worker.py                  # Worker 主服务
session_recorder.py        # 会话录制器
session_cleanup.py         # 会话清理器
```

## worker.py — Worker 主服务

Worker 基于 **FastAPI** 构建，通过独立端口提供 HTTP 和 WebSocket 服务。

### 启动与模型加载

Worker 启动时在 `lifespan()` 异步上下文中调用 `load_model()`（同步操作，约 15s），通过 `asyncio.to_thread()` 避免阻塞事件循环。加载完成后：

1. 创建 `UnifiedProcessor` 实例（加载模型权重 + TTS）
2. `gc.collect()` + `torch.cuda.empty_cache()` 清理加载残留
3. 打印 Device Map（确认所有组件在 GPU 上）
4. 状态从 `LOADING` → `IDLE`

### 核心类

#### WorkerStatus 枚举

| 状态 | 说明 | 可接受任务 |
|------|------|----------|
| `LOADING` | 模型加载中 | 否 |
| `IDLE` | 空闲 | 是 |
| `BUSY_CHAT` | 正在执行 Chat | 否 |
| `BUSY_STREAMING` | 正在执行 Streaming | 否（同连接内可） |
| `DUPLEX_ACTIVE` | Duplex 会话活跃 | 否 |
| `DUPLEX_PAUSED` | Duplex 会话暂停 | 否 |
| `ERROR` | 异常 | 否 |

#### MiniCPMOWorker

Worker 主类，封装模型加载和推理逻辑。

| 方法 | 说明 |
|------|------|
| `load_model()` | 同步加载模型，初始化 UnifiedProcessor |
| `chat(request)` | Chat 推理（无状态，完整 prefill） |
| `streaming_prefill(request)` | Streaming 预填充 |
| `streaming_init_tts(ref_audio_data)` | 初始化 Streaming TTS |
| `streaming_generate(session_id, ...)` | Streaming 流式生成（Generator） |
| `streaming_complete_turn(...)` | 完成一轮 Streaming |
| `reset_streaming_session()` | 重置 KV Cache（Gateway 指示 clear_kv_cache 时调用） |
| `duplex_prepare(...)` | Duplex 准备（系统提示词 + 参考音频） |
| `duplex_prefill(...)` | Duplex 预填充（音频 + 视频帧） |
| `duplex_generate(force_listen)` | Duplex 生成一步 |
| `duplex_finalize()` | Duplex 延迟 finalize（feed 终止符 + 滑窗维护） |
| `duplex_stop()` | 停止 Duplex |
| `duplex_cleanup()` | 清理 Duplex 资源，释放 GPU 显存 |

### HTTP / WebSocket 端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `/health` | GET | 健康检查（返回 status, gpu_id, model_loaded, kv_cache_length） |
| `/chat` | POST | Chat 推理 |
| `/streaming/stop` | POST | 停止 Streaming |
| `/ws/streaming` | WS | Streaming 流式对话 |
| `/ws/duplex` | WS | Duplex 全双工会话 |
| `/cache_info` | GET | KV Cache 信息 |
| `/clear_cache` | POST | 清除 KV Cache |

---

## 三种推理模式详解

### Chat 推理流程

Chat 是最简单的无状态模式。每次请求完整 prefill，不复用 KV Cache。

1. 检查 Worker 状态（必须 IDLE，否则等待最多 5s）
2. 设置状态 → `BUSY_CHAT`
3. `processor.set_chat_mode()` 获取 `ChatView`
4. `ChatView.chat(request)` 执行推理：
   - `_convert_messages_to_model_format()` 转换多模态消息
   - `model.chat()` 执行前向传播 + 可选 TTS
5. 记录 token 统计（`input_tokens`, `generated_tokens`）
6. 状态恢复 → `IDLE`
7. 可选：`TurnBasedSessionRecorder` 录制本轮对话

### Streaming WebSocket 处理流程

支持 **KV Cache 复用** 的多轮对话模式。Worker 使用固定 `session_id="streaming"` 管理 KV Cache 状态。

#### prefill 阶段

1. 检查状态（IDLE 或 BUSY_STREAMING）
2. 设置状态 → `BUSY_STREAMING`
3. 检查 Gateway 发来的 `clear_kv_cache` 标志：
   - `true`（缓存未命中）→ `reset_streaming_session()` 清除 KV Cache
   - `false`（缓存命中）→ 保留已有 KV Cache
4. 解码前端发送的 `ref_audio_base64` → 缓存到 `_streaming_ref_audio_cache`
5. 构建消息列表（支持 text / audio / image 多模态内容）
6. 记录 prefill 前后 KV Cache 长度差 → `cached_tokens` / `input_tokens`
7. `streaming_prefill(request)` 执行预填充
8. 发送 `prefill_done`（含 `cached_tokens`, `input_tokens`）

**KV Cache 复用关键**：`cached_tokens` 表示复用的缓存 token 数。缓存命中时 `cached_tokens > 0`，只需增量处理新消息，首 token 延迟显著降低。

#### generate 阶段

1. 从 `_streaming_ref_audio_cache` 取出 ref audio（prefill 时缓存的）
2. `streaming_init_tts(ref_audio)` 初始化 TTS
3. 在 `run_in_executor` 中运行 `streaming_generate()`：
   - Generator 逐 chunk yield `StreamingChunk`
   - 每个 chunk 通过 `asyncio.Queue` 传到主协程
   - 主协程逐个发送 chunk 给客户端
   - 每 yield 一个 chunk 后检查 `stop_event`
4. 发送 `done`（含完整 `token_stats`）
5. 状态恢复 → `IDLE`

#### 停止控制

- 每个 WS 连接创建独立的 `threading.Event`
- `threading.Event` 跨线程安全（asyncio 线程 ↔ generate 工作线程）
- 客户端发 `stop` 或断开连接时 `set()` 触发中断
- HTTP `POST /streaming/stop` 广播到所有活跃 session

### Duplex WebSocket 处理流程

最复杂的独占模式，支持全双工实时交互。Worker 在整个 Duplex 会话期间被独占。

#### prepare 阶段

1. 设置状态 → `DUPLEX_ACTIVE`（独占 Worker）
2. 解码 LLM ref_audio 和 TTS ref_audio（两者可以不同）：
   - LLM ref_audio → 嵌入 system prompt
   - TTS ref_audio → 初始化 vocoder
3. `duplex_prepare(system_prompt, ref_audio, tts_audio)` 初始化双工会话
4. 初始化 `DuplexSessionRecorder`（可选）
5. 发送 `prepared`

#### 全双工循环

每轮循环处理一个音频 chunk（约 1 秒）：

1. 解码 `audio_base64` → float32 音频波形（16kHz）
2. 解码 `frame_base64_list` → PIL Image 列表（仅 Omni 模式）
3. 等待上一轮 finalize 完成（`asyncio.Event` 栅栏）
4. 在线程中执行：
   - `duplex_prefill(audio, frames)` — 预填充音频 + 视频
   - `duplex_generate(force_listen)` — 模型决定 listen 或 speak
5. 发送 `result`（含 `is_listen`, `text`, `audio_data`, 性能指标, `kv_cache_length`）
6. **Deferred Finalize**（默认开启）：
   - 先发送结果给客户端（overlap 网络传输）
   - 异步执行 `duplex_finalize()`（约 37ms，feed 终止符 + 滑窗维护）
   - 通过 `asyncio.Event` 栅栏保证在下轮 prefill 前完成
   - 实测：LISTEN wall_clock 降低约 30ms，SPEAK 降低约 50ms

#### 暂停与恢复

- `pause` → `DUPLEX_PAUSED` + 启动超时看门狗
- `resume` → `DUPLEX_ACTIVE` + 取消看门狗
- 超时（默认 60s）→ 自动释放 Worker，通知客户端

#### 停止与清理

- `stop` → `duplex_stop()`
- `finally` 块（无论正常/异常结束）：
  - `duplex_stop()` 停止生成
  - `duplex_cleanup()` 释放 GPU 资源：
    - 释放 KV Cache、TTS caches 等
    - `gc.collect()` + `torch.cuda.empty_cache()`
    - 释放约 1.5GB 显存（诊断数据：stop 后泄漏 ~1,591 MB → cleanup 后残留 ~48 MB）
  - 状态恢复 → `IDLE`

---

## session_recorder.py — 会话录制器

自动录制所有推理会话的输入输出数据，支持后续回放和分析。

### 会话目录结构

```
data/sessions/{session_id}/
├── meta.json                # 会话元数据（类型、创建时间、配置）
├── recording.json           # Timeline 录制数据
├── user_audio/              # 用户音频 chunks (WAV)
├── user_frames/             # 用户视频帧 (JPEG，仅 Omni)
├── ai_audio/                # AI 生成音频 (WAV)
├── user_images/             # 用户上传图片 (PNG)
├── merged_replay.wav        # 合并回放音频（Duplex）
└── merged_replay.mp4        # 合并回放视频（Omni）
```

### 核心类

#### DuplexSessionRecorder

专为 Duplex 会话设计，记录每个 chunk 的 timeline 数据。

| 方法 | 说明 |
|------|------|
| `record_chunk(...)` | 记录一个 chunk（音频 + 文本 + 性能指标） |
| `save_user_audio(index, waveform)` | 保存用户音频 |
| `save_user_frame(index, jpeg_bytes)` | 保存用户视频帧 |
| `save_ai_audio(turn, chunk, waveform)` | 保存 AI 音频 |
| `finalize()` | 结束录制，生成合并回放 |

#### TurnBasedSessionRecorder

专为 Turn-based 会话设计。

| 方法 | 说明 |
|------|------|
| `start_turn(index, ts, input)` | 开始一轮对话 |
| `end_turn(timing)` | 结束一轮 |
| `add_streaming_chunk(text_delta, audio)` | 累积 streaming chunk |
| `record_chat_turn(...)` | 记录 chat turn |

使用 `ThreadPoolExecutor`（4 线程）异步写入文件，不阻塞推理。

---

## session_cleanup.py — 会话清理器

定期清理过期会话数据。

### 清理策略

1. **按时间** — 删除超过 `retention_days` 的会话
2. **按容量** — 超过 `max_storage_gb` 时按 LRU 删除

### 运行方式

- **自动**：Gateway 后台任务，每 24 小时执行
- **手动**：`python session_cleanup.py --data-dir data --retention-days 30 --max-storage-gb 50`
