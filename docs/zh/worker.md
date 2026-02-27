# Worker API 参考

Worker 是系统的推理执行单元，每个 Worker 独占一张 GPU，持有一个 `UnifiedProcessor` 实例。详细的处理流程和实现细节见 [ChatView 模式详解](architecture/chat.html) 和 [Duplex 模式详解](architecture/duplex.html)。

## WorkerStatus 枚举

| 状态 | 说明 | 可接受任务 |
|------|------|----------|
| `LOADING` | 模型加载中 | 否 |
| `IDLE` | 空闲 | 是 |
| `BUSY_HALF_DUPLEX` | 正在执行 Half-Duplex 会话 | 否（独占） |
| `DUPLEX_ACTIVE` | Duplex 会话活跃 | 否 |
| `DUPLEX_PAUSED` | Duplex 会话暂停 | 否 |
| `ERROR` | 异常 | 否 |

## MiniCPMOWorker 方法

| 方法 | 说明 |
|------|------|
| `load_model()` | 同步加载模型，初始化 UnifiedProcessor |
| `chat_prefill(session_id, msgs, ...)` | Chat prefill（一次性 KV Cache 填充） |
| `chat_non_streaming_generate(session_id, ...)` | Chat 非流式生成（HF generate + TTS） |
| `chat_streaming_generate(session_id, ...)` | Chat 流式生成（yield StreamingChunk） |
| `duplex_prepare(...)` | Duplex 准备（系统提示词 + 参考音频） |
| `duplex_prefill(...)` | Duplex 预填充（音频 + 视频帧） |
| `duplex_generate(force_listen)` | Duplex 生成一步 |
| `duplex_finalize()` | Duplex 延迟 finalize（feed 终止符 + 滑窗维护） |
| `duplex_stop()` | 停止 Duplex |
| `duplex_cleanup()` | 清理 Duplex 资源，释放 GPU 显存 |

## HTTP / WebSocket 端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `/health` | GET | 健康检查（返回 status, gpu_id, model_loaded, kv_cache_length） |
| `/streaming/stop` | POST | 停止 Streaming |
| `/ws/chat` | WS | Chat 统一接口（流式/非流式） |
| `/ws/duplex` | WS | Duplex 全双工会话 |
| `/cache_info` | GET | KV Cache 信息 |
| `/clear_cache` | POST | 清除 KV Cache |
