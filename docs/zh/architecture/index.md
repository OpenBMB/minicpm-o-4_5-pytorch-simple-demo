# 系统架构概述

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
        StreamV["StreamingView"]
        DuplexV["DuplexView"]
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
| **模型层** | UnifiedProcessor | 统一模型加载，Streaming / Duplex 毫秒级热切换 |

## 三种交互模式

系统提供三种交互模式，底层共用 **两个 WebSocket 接口**：

| 模式 | 特点 | 输入模态 | 输出模态 | 交互范式 | 对应接口 |
|------|------|---------|---------|---------|---------|
| **Turn-based Chat** | 低延迟流式交互，按钮或 VAD 触发回复，基础能力强 | 音频 + 文本 + 图像 + 视频 | 音频 + 文本 | 轮次对话 | Streaming |
| **Omnimodal Full-Duplex** | 全模态全双工，视觉 + 语音输入与语音输出同时进行 | 视觉 + 语音 | 文本 + 语音 | 全双工 | Duplex |
| **Audio Full-Duplex** | 语音全双工，语音输入和输出同时进行 | 语音 | 文本 + 语音 | 全双工 | Duplex |

```mermaid
graph LR
    subgraph modes [三种交互模式]
        TB["Turn-based Chat\n轮次对话"]
        OD["Omnimodal Full-Duplex\n全模态全双工"]
        AD["Audio Full-Duplex\n语音全双工"]
    end

    subgraph apis [两个 WebSocket 接口]
        StreamAPI["/ws/streaming/{session_id}\nStreamingView"]
        DuplexAPI["/ws/duplex/{session_id}\nDuplexView"]
    end

    TB --> StreamAPI
    OD --> DuplexAPI
    AD --> DuplexAPI
```

### Streaming 接口 — Turn-based Chat

Turn-based Chat 通过 **Streaming 接口** (`/ws/streaming/{session_id}`) 实现轮次式多模态对话。

用户发送完整消息后，模型流式返回文本 + 音频。支持 KV Cache 跨轮复用，多轮对话无需重新 prefill 历史。前端可通过按钮或 VAD（语音活动检测）触发回复。

详见 [Streaming 模式详解](streaming.html)。

### Duplex 接口 — Full-Duplex

Omnimodal Full-Duplex 和 Audio Full-Duplex 共用 **Duplex 接口** (`/ws/duplex/{session_id}`)，区别仅在于是否传入视频帧：

- **Omnimodal Full-Duplex**：每秒发送 `audio_chunk` + `video_frame`，模型同时处理视觉和语音
- **Audio Full-Duplex**：每秒仅发送 `audio_chunk`，无视觉输入

两者共享相同的 prefill-generate unit 循环，Worker 在整个会话期间被独占。

详见 [Duplex 模式详解](duplex.html)。
