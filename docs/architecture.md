# 系统架构与拓扑

## 整体架构

系统采用 **Frontend - Gateway - Worker Pool** 三层架构：

```mermaid
graph TB
    subgraph clientLayer [客户端层]
        Browser["浏览器 (HTML/JS)"]
    end

    subgraph gatewayLayer [网关层]
        Gateway["Gateway (:8006, HTTPS)"]
        WPool["WorkerPool 调度器"]
        Queue["FIFO 请求队列"]
        AppReg["AppRegistry"]
        RefAudio["RefAudioRegistry"]
        Gateway --> WPool
        Gateway --> Queue
        Gateway --> AppReg
        Gateway --> RefAudio
    end

    subgraph workerLayer [Worker 层]
        W0["Worker 0 (GPU 0)\n:22400"]
        W1["Worker 1 (GPU 1)\n:22401"]
        W2["Worker N (GPU N)\n:22400+N"]
    end

    subgraph modelLayer [模型层]
        UP["UnifiedProcessor"]
        ChatV["ChatView"]
        StreamV["StreamingView"]
        DuplexV["DuplexView"]
        UP --> ChatV
        UP --> StreamV
        UP --> DuplexV
    end

    Browser -->|"HTTPS / WSS"| Gateway
    WPool -->|"HTTP / WS (内部)"| W0
    WPool -->|"HTTP / WS (内部)"| W1
    WPool -->|"HTTP / WS (内部)"| W2
    W0 --> UP
```

### 各层职责

| 层 | 组件 | 职责 |
|----|------|------|
| **客户端层** | 浏览器前端 | 模式选择、音视频采集、WebSocket 通信、会话录制 |
| **网关层** | Gateway | 请求路由分发、WebSocket 代理、FIFO 排队、会话亲和、ETA 估算 |
| **Worker 层** | Worker x N | 每 Worker 独占一张 GPU，执行模型推理，管理 KV Cache |
| **模型层** | UnifiedProcessor | 统一模型加载，三种模式毫秒级热切换 |

## 请求处理流程

### Chat 模式（无状态 HTTP）

```mermaid
sequenceDiagram
    participant C as 客户端
    participant G as Gateway
    participant Q as 队列
    participant W as Worker

    C->>G: POST /api/chat (ChatRequest)
    G->>Q: enqueue("chat")
    Q->>W: 分配空闲 Worker
    W->>W: UnifiedProcessor.chat.chat()
    W-->>G: ChatResponse
    G-->>C: JSON 响应
    G->>Q: release_worker()
```

### Streaming 模式（WebSocket + LRU 缓存 + KV Cache 复用）

Streaming 模式的核心优化是 **KV Cache 复用**。当同一会话的多轮对话命中同一个 Worker 时，模型只需对新增消息做增量 prefill，而非重新处理整个历史，大幅降低延迟。

```mermaid
sequenceDiagram
    participant C as 客户端
    participant GW as Gateway
    participant Pool as WorkerPool
    participant W as Worker

    C->>GW: WS 连接 /ws/streaming/{session_id}
    GW->>Pool: enqueue("streaming", history_hash)

    alt 有空闲 Worker（立即分配）
        Pool->>Pool: LRU 缓存路由（见下方详解）
        Pool-->>GW: Future.resolve(Worker)
        GW-->>C: queue_done
    else 无空闲 Worker（入队等待）
        Pool-->>GW: 入 FIFO 队列
        GW-->>C: queued (position, eta)
        loop 等待释放
            Pool-->>GW: 位置更新
            GW-->>C: queue_update (position, eta)
        end
        Note over Pool: Worker 释放后 _dispatch_next()
        Pool-->>GW: Future.resolve(Worker)
        GW-->>C: queue_done
    end

    GW->>W: 建立到 Worker 的 WS 连接

    alt 缓存未命中（clear_kv_cache=true）
        GW->>W: prefill (clear_kv_cache=true, 全部消息)
        W->>W: reset_session() 清除旧 KV Cache
        W->>W: streaming_prefill() 全量预填充
    else 缓存命中（clear_kv_cache=false）
        GW->>W: prefill (clear_kv_cache=false, 仅新消息)
        W->>W: streaming_prefill() 增量预填充<br/>复用已有 KV Cache
    end
    W-->>GW: prefill_done (cached_tokens, input_tokens)
    GW-->>C: 转发 prefill_done

    C->>GW: generate
    GW->>W: 转发 generate
    W->>W: streaming_init_tts(ref_audio)
    loop 流式生成（逐 chunk）
        W->>W: streaming_generate() yield chunk
        W-->>GW: StreamingChunk (text_delta + audio_data)
        GW-->>C: 转发 chunk
    end
    W-->>GW: done (token_stats)
    GW-->>C: 转发 done

    GW->>Pool: release_worker(cached_hash=当前 hash)
    Note over Pool: 更新 Worker.cached_hash<br/>下次同 hash 请求可命中
```

#### LRU 缓存路由详解

LRU 缓存路由在 **Gateway 侧**的 `WorkerPool._route_streaming_worker()` 中实现（位于 `gateway_modules/worker_pool.py`），而非 Worker 侧。Gateway 通过 `WorkerConnection.cached_hash` 和 `last_cache_used_at` 字段追踪每个 Worker 当前缓存的会话历史 hash。

**路由 4 级优先级**：

```mermaid
flowchart TD
    Start["收到 Streaming 请求\n计算 history_hash"] --> Check1{"遍历空闲 Worker\nhash == cached_hash ?"}
    Check1 -->|"命中"| Hit["缓存命中\n分配该 Worker\n增量 prefill"]
    Check1 -->|"未命中"| Check2{"有 cached_hash==None\n的空闲 Worker ?"}
    Check2 -->|"有"| NoCache["分配无缓存 Worker\n避免不必要的淘汰"]
    Check2 -->|"无"| Check3{"有其他空闲 Worker ?"}
    Check3 -->|"有"| LRU["LRU 淘汰\n选 last_cache_used_at 最旧的\n覆盖其缓存"]
    Check3 -->|"无"| Enqueue["无空闲 Worker\n入 FIFO 队列等待"]
```

- **hash 计算**：`compute_history_hash()` 将消息列表的 `role + content` 序列化后做 SHA-256，确保相同对话历史产生相同 hash。
- **缓存更新时机**：Gateway 在 `release_worker()` 时将当前请求的 `history_hash` 写入 `Worker.cached_hash` 和 `last_cache_used_at`，供后续请求匹配。
- **Non-Streaming 请求也考虑缓存**：Chat/Duplex 分配时也优先选无缓存的 Worker（`_get_idle_worker()`），避免不必要地淘汰 Streaming 的缓存。

### Duplex 模式（WebSocket + 独占 Worker）

```mermaid
sequenceDiagram
    participant C as 客户端
    participant G as Gateway
    participant Q as 队列
    participant W as Worker

    C->>G: WS /ws/duplex/{session_id}
    G->>Q: enqueue("omni_duplex" / "audio_duplex")
    Q->>W: 分配 Worker（独占）
    C->>G: prepare (系统提示词 + 配置)
    G->>W: duplex_prepare()
    loop 全双工循环
        C->>G: audio_chunk (+ video_frame)
        G->>W: duplex_prefill(audio, frames)
        W->>W: duplex_generate()
        alt 模型决定说话
            W-->>G: result (text + audio_data)
            G-->>C: 转发 result
        else 模型决定聆听
            W-->>G: result (is_listen=true)
        end
    end
    C->>G: stop
    G->>W: duplex_cleanup()
    G->>Q: release_worker()
```

## FIFO 队列与 Worker 通信机制

队列在 Gateway 侧的 `WorkerPool` 中实现，使用 `OrderedDict` 保证 FIFO 顺序。核心通信机制如下：

```mermaid
flowchart TB
    subgraph enqueueFlow [入队流程]
        Req["请求到达"] --> TryImmediate{"有空闲 Worker ?"}
        TryImmediate -->|"有"| Assign["立即分配\nFuture.set_result(worker)\nWorker.mark_busy()"]
        TryImmediate -->|"无"| CapCheck{"队列未满 ?"}
        CapCheck -->|"满"| Reject["拒绝: QueueFullError"]
        CapCheck -->|"未满"| AddQueue["创建 QueueEntry\n含 asyncio.Future\n加入 OrderedDict"]
    end

    subgraph dispatchFlow [调度流程 _dispatch_next]
        Release["Worker 释放\nrelease_worker()"] --> Dispatch["_dispatch_next()"]
        HealthOK["健康检查恢复 IDLE"] --> Dispatch
        Cancel["取消排队项"] --> Dispatch
        Dispatch --> PeekHead{"取队头 Entry"}
        PeekHead --> FindWorker{"匹配空闲 Worker"}
        FindWorker -->|"找到"| DoAssign["Worker.mark_busy()\nFuture.set_result(worker)\n移除队头"]
        FindWorker -->|"无空闲"| Wait["等待下次触发"]
        DoAssign --> PeekHead
    end
```

**关键设计**：

1. **asyncio.Future 桥接**：每个排队请求持有一个 `asyncio.Future`，Gateway 的 WebSocket handler 通过 `await future` 阻塞等待分配结果。Worker 空闲时 `_dispatch_next()` 调用 `future.set_result(worker)` 唤醒等待者。
2. **单一调度点**：所有 Worker 分配都通过 `_dispatch_next()` 进行，在 Worker 释放、排队取消、健康检查恢复后触发，消除并发竞争。
3. **立即标记忙碌**：分配 Worker 时立即调用 `mark_busy()` 将状态改为忙碌，防止同一 Worker 被重复分配给多个请求。
4. **Gateway → Worker 通信**：Gateway 通过 HTTP（Chat）或 WebSocket（Streaming/Duplex）直连 Worker 的内部端口（22400+），不经过队列。队列只负责 Worker 分配，不参与数据传输。

## 模块依赖拓扑

```mermaid
graph LR
    subgraph entryPoints [入口]
        GW["gateway.py"]
        WK["worker.py"]
        SA["start_all.sh"]
    end

    subgraph gatewayMods [gateway_modules/]
        WP["worker_pool.py"]
        AR["app_registry.py"]
        MD["models.py"]
        RA["ref_audio_registry.py"]
    end

    subgraph coreMod [core/]
        SC["schemas/"]
        PR["processors/"]
        CP["capabilities.py"]
        FA["factory.py"]
    end

    subgraph modelMod [MiniCPMO45/]
        CFG["configuration_minicpmo.py"]
        MOD["modeling_minicpmo.py"]
        UNI["modeling_minicpmo_unified.py"]
        VIS["modeling_navit_siglip.py"]
        PRC["processing_minicpmo.py"]
        TOK["tokenization_minicpmo_fast.py"]
        UTL["utils.py"]
    end

    subgraph support [辅助模块]
        CONF["config.py"]
        SR["session_recorder.py"]
        SCLEAN["session_cleanup.py"]
    end

    SA --> GW
    SA --> WK
    GW --> WP
    GW --> AR
    GW --> MD
    GW --> RA
    GW --> CONF
    GW --> SCLEAN

    WK --> PR
    WK --> SC
    WK --> CONF
    WK --> SR

    PR --> MOD
    PR --> UNI
    FA --> PR

    UNI --> MOD
    UNI --> VIS
    UNI --> UTL
    MOD --> CFG
    MOD --> VIS
    MOD --> PRC
    MOD --> TOK
    MOD --> UTL

    WP --> MD
```

## 模型推理管线

```mermaid
graph LR
    subgraph inputMod [多模态输入]
        TXT["文本"]
        IMG["图像"]
        AUD["音频"]
    end

    subgraph encoders [编码器]
        TOK2["Tokenizer\n(Qwen2Fast)"]
        VE["SigLIP\nVision Encoder"]
        RS["Resampler"]
        AE["Whisper\nAudio Encoder"]
        AP["Audio\nProjection"]
    end

    subgraph llmBlock [语言模型]
        EMB["Embedding\n融合层"]
        LLM["Qwen3\nLLM Backbone"]
    end

    subgraph outputMod [输出]
        TXTOUT["文本输出"]
        TTS["TTS\n(Token2Wav / CosyVoice2)"]
        AUDOUT["音频输出\n(24kHz)"]
    end

    TXT --> TOK2 --> EMB
    IMG --> VE --> RS --> EMB
    AUD --> AE --> AP --> EMB
    EMB --> LLM
    LLM --> TXTOUT
    LLM --> TTS --> AUDOUT
```

## Worker 状态机

```mermaid
stateDiagram-v2
    [*] --> LOADING: 启动
    LOADING --> IDLE: 模型加载完成
    LOADING --> ERROR: 加载失败

    IDLE --> BUSY_CHAT: 分配 Chat 任务
    IDLE --> BUSY_STREAMING: 分配 Streaming 任务
    IDLE --> DUPLEX_ACTIVE: 分配 Duplex 任务

    BUSY_CHAT --> IDLE: 推理完成
    BUSY_STREAMING --> IDLE: 推理完成

    DUPLEX_ACTIVE --> DUPLEX_PAUSED: pause（客户端暂停）
    DUPLEX_PAUSED --> DUPLEX_ACTIVE: resume（客户端恢复）
    DUPLEX_PAUSED --> IDLE: 超时释放
    DUPLEX_ACTIVE --> IDLE: stop / cleanup

    ERROR --> [*]
```

## 前端组件拓扑

```mermaid
graph TB
    subgraph pages [页面]
        IDX["index.html\n首页"]
        TB["turnbased.html\n轮次对话"]
        OM["omni.html\nOmni 全双工"]
        AD["audio_duplex.html\n音频全双工"]
        ADM["admin.html\n管理面板"]
        SV["session-viewer.html\n会话回放"]
    end

    subgraph sharedComp [shared/]
        NAV["app-nav.js\n导航组件"]
        PS["preset-selector.js\n预设选择器"]
        SS["save-share.js\n保存分享"]
    end

    subgraph duplexLib [duplex/lib/]
        DS["duplex-session.js\n会话管理"]
        APL["audio-player.js\n音频播放"]
        CP2["capture-processor.js\n音频采集"]
        LU["lufs.js\n响度测量"]
        MC["mixer-controller.js\n混音器"]
        QC["queue-chimes.js\n排队音效"]
        SRec["session-recorder.js\n录制器"]
        SVR["session-video-recorder.js\n视频录制"]
    end

    subgraph duplexUI [duplex/ui/]
        DUI["duplex-ui.js\n指标面板"]
        RAI["ref-audio-init.js\n参考音频初始化"]
        TRC["tts-ref-controller.js\nTTS 控制"]
    end

    IDX --> NAV
    TB --> NAV
    TB --> PS
    OM --> NAV
    OM --> DS
    OM --> APL
    OM --> CP2
    AD --> NAV
    AD --> DS
    AD --> APL
    AD --> CP2
    ADM --> NAV
    SV --> NAV

    DS --> APL
    DS --> QC
    OM --> DUI
    AD --> DUI
    OM --> MC
    AD --> MC
    OM --> SRec
    AD --> SRec
    OM --> SVR
```
