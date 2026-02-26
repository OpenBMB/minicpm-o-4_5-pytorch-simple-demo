# System Architecture Overview

## Overall Architecture

The system adopts a **Frontend - Gateway - Worker Pool** three-tier architecture:

```mermaid
graph TB
    subgraph clientLayer [Client Layer]
        Browser["Browser (HTML/JS)"]
    end

    subgraph gatewayLayer [Gateway Layer]
        Gateway["Gateway (:8006, HTTPS)"]
        WPool["WorkerPool Scheduler"]
        Queue["FIFO Request Queue"]
        AppReg["AppRegistry"]
        RefAudio["RefAudioRegistry"]
        Gateway --> WPool
        Gateway --> Queue
        Gateway --> AppReg
        Gateway --> RefAudio
    end

    subgraph workerLayer [Worker Layer]
        W0["Worker 0 (GPU 0)\n:22400"]
        W1["Worker 1 (GPU 1)\n:22401"]
        W2["Worker N (GPU N)\n:22400+N"]
    end

    subgraph modelLayer [Model Layer]
        UP["UnifiedProcessor"]
        ChatV["ChatView"]
        DuplexV["DuplexView"]
        UP --> ChatV
        UP --> DuplexV
    end

    Browser -->|"HTTPS / WSS"| Gateway
    WPool -->|"HTTP / WS (Internal)"| W0
    WPool -->|"HTTP / WS (Internal)"| W1
    WPool -->|"HTTP / WS (Internal)"| W2
    W0 --> UP
```

### Responsibilities of Each Layer

| Layer | Component | Responsibilities |
|-------|-----------|-----------------|
| **Client Layer** | Browser Frontend | Mode selection, audio/video capture, WebSocket communication, session recording |
| **Gateway Layer** | Gateway | Request routing & dispatch, WebSocket proxy, FIFO queuing, session affinity, ETA estimation |
| **Worker Layer** | Worker x N | Each Worker owns a dedicated GPU, performs model inference, manages KV Cache |
| **Model Layer** | UnifiedProcessor | Unified model loading, millisecond-level hot-switching between Streaming / Duplex |

## Three Interaction Modes

The system provides three interaction modes, sharing **two WebSocket endpoints** under the hood:

| Mode | Features | Input Modalities | Output Modalities | Interaction Paradigm | Endpoint |
|------|----------|------------------|-------------------|---------------------|----------|
| **Turn-based Chat** | Low-latency streaming interaction, reply triggered by button or VAD, strong base capabilities | Audio + Text + Image + Video | Audio + Text | Turn-based dialogue | ChatView |
| **Omnimodal Full-Duplex** | Full-modality full-duplex, vision + voice input and voice output occur simultaneously | Vision + Voice | Text + Voice | Full-duplex | Duplex |
| **Audio Full-Duplex** | Voice full-duplex, voice input and output occur simultaneously | Voice | Text + Voice | Full-duplex | Duplex |

```mermaid
graph LR
    subgraph modes [Three Interaction Modes]
        TB["Turn-based Chat"]
        OD["Omnimodal Full-Duplex"]
        AD["Audio Full-Duplex"]
    end

    subgraph apis [Two WebSocket Endpoints]
        ChatAPI["/ws/chat\nChatView"]
        DuplexAPI["/ws/duplex/{session_id}\nDuplexView"]
    end

    TB --> ChatAPI
    OD --> DuplexAPI
    AD --> DuplexAPI
```

### Chat Endpoint — Turn-based Chat

Turn-based Chat uses **ChatView** (`/ws/chat` WebSocket) to implement turn-based multimodal dialogue.

ChatView splits inference into prefill and generate stages: prefill fills all messages into the KV Cache in one shot, and generate supports both streaming and non-streaming modes. The frontend can toggle the Streaming switch to choose between real-time token-by-token output or one-shot response.

See [ChatView Mode Details](streaming.html) for more information.

### Duplex Endpoint — Full-Duplex

Omnimodal Full-Duplex and Audio Full-Duplex share the **Duplex endpoint** (`/ws/duplex/{session_id}`), differing only in whether video frames are sent:

- **Omnimodal Full-Duplex**: Sends `audio_chunk` + `video_frame` every second; the model processes both vision and voice simultaneously
- **Audio Full-Duplex**: Sends only `audio_chunk` every second; no visual input

Both share the same prefill-generate unit loop, and the Worker is exclusively occupied throughout the entire session.

See [Duplex Mode Details](duplex.html) for more information.
