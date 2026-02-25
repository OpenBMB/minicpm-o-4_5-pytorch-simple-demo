# Gateway 模块详解

Gateway 是系统的请求入口和路由中枢，不加载模型，负责请求分发、WebSocket 代理、队列管理和资源管理。

## 模块结构

```
gateway.py                         # Gateway 主服务
gateway_modules/
├── __init__.py                    # 包标识
├── worker_pool.py                 # WorkerPool 调度器
├── app_registry.py                # 应用注册表
├── models.py                      # 数据模型定义
└── ref_audio_registry.py          # 参考音频注册表
```

## gateway.py — 主服务

Gateway 基于 **FastAPI** 构建，使用 **uvicorn** 运行，提供 HTTP REST API 和 WebSocket 端点。

### 生命周期管理

`lifespan()` 异步上下文管理器负责：
1. 从 `config.py` 加载配置
2. 初始化 `WorkerPool` 并启动健康检查
3. 初始化 `RefAudioRegistry` 和 `AppRegistry`
4. 启动后台 session 清理任务（每天执行一次）
5. 配置 HTTPS（自签名证书）或 HTTP 模式

### 核心路由

#### HTTP 端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/status` | GET | 服务全局状态 |
| `/workers` | GET | Worker 列表 |
| `/api/chat` | POST | Chat 推理（无状态） |
| `/api/streaming/stop` | POST | 停止 Streaming 生成 |
| `/api/frontend_defaults` | GET | 前端默认配置 |
| `/api/presets` | GET | System Prompt 预设列表 |
| `/api/default_ref_audio` | GET | 默认参考音频 |
| `/api/assets/ref_audio` | GET | 参考音频列表 |
| `/api/assets/ref_audio/external` | POST | 上传参考音频 |
| `/api/assets/ref_audio/{id}` | DELETE | 删除参考音频 |
| `/api/queue` | GET | 队列状态 |
| `/api/queue/{ticket_id}` | GET/DELETE | 查询/取消排队项 |
| `/api/config/eta` | GET/PUT | ETA 配置管理 |
| `/api/cache` | GET | KV Cache 状态 |
| `/api/sessions/{sid}` | GET | 会话元数据 |
| `/api/sessions/{sid}/recording` | GET | 会话录制数据 |
| `/api/sessions/{sid}/assets/{rel}` | GET | 会话资源文件 |
| `/api/sessions/{sid}/download` | GET | 下载会话 |
| `/api/sessions/{sid}/upload-recording` | POST | 上传前端录制 |
| `/api/apps` | GET | 已启用应用列表 |
| `/api/admin/apps` | GET/PUT | 应用管理（Admin） |

#### WebSocket 端点

| 端点 | 功能 |
|------|------|
| `/ws/streaming/{session_id}` | Streaming 流式对话代理 |
| `/ws/duplex/{session_id}` | Duplex 全双工会话代理 |

### WebSocket 代理机制

**Streaming 代理**：
1. 接收客户端 WebSocket 连接
2. 将请求入队 `WorkerPool.enqueue("streaming", history_hash)`
3. 排队期间向客户端推送 `queued` / `queue_update` 消息
4. 获取到 Worker 后，建立到 Worker 的 WebSocket 连接
5. 双向转发消息（client ↔ worker）
6. 完成后释放 Worker

**Duplex 代理**：
1. 接收客户端 WebSocket 连接
2. 将请求入队 `WorkerPool.enqueue("omni_duplex" / "audio_duplex")`
3. 获取到 Worker 后独占该 Worker
4. 启动两个并行任务：`client_to_worker` 和 `worker_to_client`
5. 客户端断开或发送 `stop` 后清理释放

### 安全措施

- Session ID 通过正则 `^[a-zA-Z0-9_-]{1,64}$` 校验，防止路径遍历
- 上传文件大小限制 200MB
- HTTPS 默认启用（自签名证书）

---

## worker_pool.py — WorkerPool 调度器

WorkerPool 是 Gateway 的核心调度组件，管理所有 Worker 连接和请求分发。

### 核心类

#### WorkerConnection

Worker 连接数据类，维护单个 Worker 的状态信息。

| 属性 | 类型 | 说明 |
|------|------|------|
| `url` | str | Worker 地址 |
| `index` | int | Worker 序号 |
| `status` | GatewayWorkerStatus | 当前状态 |
| `current_task` | str | 当前任务类型 |
| `current_session_id` | str | 当前会话 ID |
| `cached_hash` | str | 缓存的历史 hash |
| `busy_since` | datetime | 忙碌开始时间 |

关键方法：
- `mark_busy(task, session_id)` — 标记为忙碌
- `mark_idle()` — 标记为空闲，更新 `cached_hash`
- `update_duplex_status(status)` — 更新 Duplex 状态
- `to_info()` — 转换为 API 响应模型

#### QueueEntry

队列条目，包含 `ticket`（排队凭证）、`future`（等待结果）和 `history_hash`（消息历史摘要）。

#### EtaTracker

ETA 估算追踪器，结合基准值和 EMA（指数移动平均）动态估算等待时间。

- `get_eta(task_type)` — 获取单任务预估耗时
- `record_duration(task_type, duration)` — 记录实际耗时，更新 EMA
- `update_config(config)` — 更新基准值

#### WorkerPool

| 方法 | 说明 |
|------|------|
| `start()` | 启动健康检查循环 |
| `stop()` | 停止并清理 |
| `enqueue(task_type, history_hash)` | 入队请求，返回 (ticket, future) |
| `release_worker(worker_url, cached_hash)` | 释放 Worker |
| `cancel(ticket_id)` | 取消排队 |
| `get_ticket(ticket_id)` | 查询排队状态 |
| `get_queue_status()` | 队列快照 |
| `get_all_workers()` | Worker 列表 |

### FIFO 队列机制

队列使用 `OrderedDict` 实现，所有请求类型（Chat / Streaming / Duplex）共享一个统一队列，保证公平的先到先服务。

**入队流程**（`enqueue()`）：
1. 先尝试立即分配 — 如有空闲 Worker，`Future` 立刻 resolve，请求不进入队列
2. 无空闲 Worker 时检查容量（默认 1000），满则拒绝
3. 创建 `QueueEntry`（含 `QueueTicket` + `asyncio.Future`），追加到 OrderedDict 尾部

**调度流程**（`_dispatch_next()`）— 唯一的 Worker 分配入口：
1. 取队头 Entry
2. 根据请求类型匹配空闲 Worker（Streaming 走 LRU 路由，其余走通用路由）
3. 找到 → 立即 `mark_busy()` + `future.set_result(worker)` + 移除队头
4. 循环直到无空闲 Worker 或队列为空

**触发时机**：Worker 释放、排队取消、健康检查恢复 IDLE 后均触发 `_dispatch_next()`。

**Gateway ↔ Worker 通信**：队列只负责 Worker 分配（决定哪个 Worker 处理哪个请求），不参与数据传输。Gateway 获得 Worker 引用后，直接通过 HTTP（Chat: `POST /chat`）或 WebSocket（Streaming: `/ws/streaming`、Duplex: `/ws/duplex`）连接 Worker 的内部端口（22400+）进行通信。

### LRU 缓存路由（Streaming 专用）

LRU 路由的目的是让同一个会话的多轮对话尽量命中同一个 Worker，复用其 GPU 上的 KV Cache，从而跳过历史消息的重复计算。

**实现位置**：`WorkerPool._route_streaming_worker()`（`gateway_modules/worker_pool.py`）

**每个 Worker 维护的缓存状态**（在 Gateway 侧的 `WorkerConnection` 上）：
- `cached_hash: Optional[str]` — 当前 Worker 上 KV Cache 对应的消息历史 hash
- `last_cache_used_at: Optional[datetime]` — 上次缓存被使用的时间（LRU 淘汰依据）

**路由优先级**（4 级 fallback）：
1. **缓存命中** — 遍历空闲 Worker，找 `cached_hash == history_hash` 的 → 增量 prefill
2. **无缓存 Worker** — 找 `cached_hash == None` 的空闲 Worker → 不淘汰任何缓存
3. **LRU 淘汰** — 所有空闲 Worker 都有缓存时，选 `last_cache_used_at` 最旧的 → 覆盖其缓存
4. **无空闲** — 返回 None，请求入 FIFO 队列等待

**hash 计算**：`compute_history_hash()` 将消息列表的 `[{role, content}]` 序列化后做 SHA-256。

**缓存更新**：Gateway 在 `release_worker()` 时将本次请求的 `history_hash` 写入 `cached_hash` 和 `last_cache_used_at`。

**缓存命中时 Gateway 如何通知 Worker**：Gateway 在转发 `prefill` 消息时附带 `clear_kv_cache` 字段 —— 命中时为 `false`（Worker 保留已有 KV Cache，仅增量预填充新消息），未命中时为 `true`（Worker 调用 `reset_session()` 清除旧缓存后全量重新预填充）。

**Non-Streaming 也考虑缓存**：Chat/Duplex 分配时 `_get_idle_worker()` 优先选无缓存的 Worker，避免不必要地淘汰 Streaming 的缓存。

### ETA 估算

`EtaTracker` 结合 Admin 配置的基准值和运行时 EMA（指数移动平均）动态估算等待时间：

- 每种任务类型维护独立的基准值和 EMA 动态值
- 有足够样本（默认 >= 3）时使用 EMA 值，否则用基准值
- `_recalc_positions_and_eta()` 使用最小堆模拟 dispatch 链，逐个弹出最早空闲的 Worker 分配给队头，精确计算每个排队者的等待时间
- ETA 结果通过 WebSocket 实时推送给排队中的客户端

### 健康检查

每 10 秒轮询所有 Worker 的 `/health` 端点（`_health_check_loop()`），更新状态。健康检查后触发 `_dispatch_next()`（可能有 Worker 从 OFFLINE 恢复为 IDLE）。

**Gateway 调度权威**：当 Gateway 已 dispatch 任务给 Worker（`_gateway_dispatched=True`）时，即使 Worker `/health` 暂时报告 `idle`（cleanup 瞬态），Gateway 也不会将其降级为 IDLE，防止误分配。`release_worker()` 清除此标记后健康检查恢复权威。

---

## app_registry.py — 应用注册表

管理前端应用的启用/禁用状态，支持运行时动态切换。

### 默认应用

| ID | 名称 | 默认状态 |
|----|------|---------|
| `turnbased` | Turn-based Chat | 启用 |
| `omni` | Omnimodal Full-Duplex | 启用 |
| `audio_duplex` | Audio Full-Duplex | 启用 |

### AppRegistry 类

| 方法 | 说明 |
|------|------|
| `is_enabled(app_id)` | 检查应用是否启用 |
| `set_enabled(app_id, enabled)` | 设置启用状态 |
| `get_enabled_apps()` | 获取已启用应用列表 |
| `get_all_apps()` | 获取所有应用（Admin 用） |

通过 `threading.Lock` 保证线程安全。

---

## models.py — Gateway 数据模型

为 Gateway 层定义所有 API 数据模型，基于 Pydantic。

### 枚举

- **GatewayWorkerStatus** — Worker 状态枚举：`idle` / `busy_streaming` / `duplex_active` / `duplex_paused` / `offline`

### 数据模型

| 模型 | 说明 |
|------|------|
| `WorkerInfo` | Worker 信息（url, index, status, task, session_id, cached_hash, busy_since） |
| `QueueTicket` | 排队凭证（ticket_id, position, eta_seconds, task_type） |
| `QueueTicketSummary` | 简要凭证（ticket_id, position） |
| `RunningTaskInfo` | 运行中任务（worker_url, task_type, session_id, started_at, elapsed_s） |
| `QueueStatus` | 队列快照（queue_length, entries, running） |
| `ServiceStatus` | 全局状态（total_workers, idle, busy, queue_length） |
| `EtaConfig` / `EtaStatus` | ETA 配置和状态 |
| `WorkersResponse` | Worker 列表响应 |

---

## ref_audio_registry.py — 参考音频注册表

管理 TTS 参考音频的存储和元数据。

### RefAudioRegistry 类

| 方法 | 说明 |
|------|------|
| `upload(name, audio_data, source)` | 上传参考音频（自动归一化为 16kHz mono 16-bit WAV） |
| `get(audio_id)` | 查询音频信息 |
| `get_file_path(audio_id)` | 获取文件路径 |
| `get_base64(audio_id)` | 获取 Base64 编码 |
| `list_all()` | 列出所有参考音频 |
| `delete(audio_id)` | 删除参考音频 |
| `exists(audio_id)` | 检查是否存在 |

音频上传时自动使用 `librosa` + `soundfile` 进行格式归一化。元数据持久化到 JSON 文件。
