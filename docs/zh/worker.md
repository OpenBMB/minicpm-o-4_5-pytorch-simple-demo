# Worker API 参考

Worker 是系统的推理执行单元，每个 Worker 独占一张 GPU，持有一个 `UnifiedProcessor` 实例。详细的处理流程和实现细节见 [Streaming 模式详解](architecture/streaming.html) 和 [Duplex 模式详解](architecture/duplex.html)。

## WorkerStatus 枚举

| 状态 | 说明 | 可接受任务 |
|------|------|----------|
| `LOADING` | 模型加载中 | 否 |
| `IDLE` | 空闲 | 是 |
| `BUSY_STREAMING` | 正在执行 Streaming | 否（同连接内可） |
| `DUPLEX_ACTIVE` | Duplex 会话活跃 | 否 |
| `DUPLEX_PAUSED` | Duplex 会话暂停 | 否 |
| `ERROR` | 异常 | 否 |

## MiniCPMOWorker 方法

| 方法 | 说明 |
|------|------|
| `load_model()` | 同步加载模型，初始化 UnifiedProcessor |
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

## HTTP / WebSocket 端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `/health` | GET | 健康检查（返回 status, gpu_id, model_loaded, kv_cache_length） |
| `/streaming/stop` | POST | 停止 Streaming |
| `/ws/streaming` | WS | Streaming 流式对话 |
| `/ws/duplex` | WS | Duplex 全双工会话 |
| `/cache_info` | GET | KV Cache 信息 |
| `/clear_cache` | POST | 清除 KV Cache |
