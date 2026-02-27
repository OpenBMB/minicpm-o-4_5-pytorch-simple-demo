# 内部机制

## Worker 启动与模型加载

Worker 基于 **FastAPI** 构建，每个 Worker 独占一张 GPU，通过独立端口提供 HTTP 和 WebSocket 服务。

启动时在 `lifespan()` 异步上下文中调用 `load_model()`（同步操作，约 15s），通过 `asyncio.to_thread()` 避免阻塞事件循环。加载完成后：

1. 创建 `UnifiedProcessor` 实例（加载模型权重 + TTS）
2. `gc.collect()` + `torch.cuda.empty_cache()` 清理加载残留
3. 打印 Device Map（确认所有组件在 GPU 上）
4. 状态从 `LOADING` → `IDLE`

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
4. **Gateway → Worker 通信**：Gateway 通过 WebSocket（Streaming/Duplex）直连 Worker 的内部端口（22400+），不经过队列。队列只负责 Worker 分配，不参与数据传输。

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

    IDLE --> BUSY_HALF_DUPLEX: 分配 Half-Duplex 任务
    IDLE --> DUPLEX_ACTIVE: 分配 Duplex 任务

    BUSY_HALF_DUPLEX --> IDLE: 会话结束/超时

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

## 会话录制

`session_recorder.py` 自动录制所有推理会话的输入输出数据，支持后续回放和分析。

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

录制器使用 `ThreadPoolExecutor`（4 线程）异步写入文件，不阻塞推理。

| 录制器 | 用途 |
|--------|------|
| `DuplexSessionRecorder` | Duplex 会话，记录每个 chunk 的 timeline 数据 |
| `TurnBasedSessionRecorder` | Turn-based 会话，累积 streaming chunk |

## 会话清理

`session_cleanup.py` 定期清理过期会话数据。

### 清理策略

1. **按时间** — 删除超过 `retention_days` 的会话
2. **按容量** — 超过 `max_storage_gb` 时按 LRU 删除

### 运行方式

- **自动**：Gateway 后台任务，每 24 小时执行
- **手动**：`python session_cleanup.py --data-dir data --retention-days 30 --max-storage-gb 50`
