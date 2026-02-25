# 前端页面与路由

## 动态导航系统 (app-nav.js)

`AppNav` 组件从 `/api/apps` 获取已启用的应用列表，动态渲染导航链接。访问未启用的应用时自动重定向到首页。

---

## index.html — 首页

- 展示三种交互模式的卡片，显示模式名称和特性
- 从 `/api/apps` 获取启用状态，灰置未启用模式
- 展示最近会话列表（来自 localStorage，通过 `SaveShareUI` 管理）

---

## turnbased.html — 轮次对话

最复杂的非双工页面，支持 Chat 和 Streaming 两种模式。

### 全局状态对象

```javascript
const state = {
    messages: [],                // 消息列表 {role, content, displayText}
    systemContentList: [],       // 系统内容列表 (text + audio + image)
    isGenerating: false,         // 是否正在生成
    generationPhase: 'idle',     // 'idle' | 'queuing' | 'generating'
    currentTicketId: null,       // 排队 ticket ID（仅 streaming）
    abortController: null,       // Fetch abort（仅 chat）
    streamingWs: null,           // WebSocket 连接（仅 streaming）
    requestId: 'req_' + Date.now(),
    currentView: 'initial',      // 'initial' | 'conversation'
    editingIndex: -1,            // 正在编辑的消息索引
    ttsRefAudioMode: 'extract',  // 'extract' | 'independent'
    ttsRefAudioData: null,       // TTS 参考音频 base64
};
```

### Streaming 模式

通过 WebSocket (`WS /ws/streaming/{request_id}`) 进行流式对话：

1. 建立 WebSocket 连接
2. 发送 `prefill` 消息（含消息历史 + ref_audio_base64）
3. 收到 `prefill_done` 后发送 `generate`
4. 逐 chunk 接收 `StreamingChunk`（text_delta + audio_data）
5. 收到 `done` 后渲染完整结果
6. 排队时显示 `CountdownTimer`

### 消息构建流程

1. 从 `UserContentEditor` 获取用户输入 items
2. 音频 Blob → 重采样到 16kHz mono → Base64 PCM float32
3. 图片 File → Base64
4. 构建 `content list` 格式：`[{type:"text", text:...}, {type:"audio", data:...}, ...]`
5. `buildRequestMessages()` 组装完整消息列表（含 system prompt）

### TTS 参考音频

两种模式：
- **extract**：从系统内容中提取（需恰好 1 个音频且 <20s）
- **independent**：独立上传参考音频

在 Streaming 的 `prefill` 消息中通过 `tts_ref_audio_base64` 字段发送给 Worker。

### 响应渲染

- `addMessageUI()` — 添加消息气泡到对话区
- `updateLastAssistantMessage()` — 流式更新最后一条助手消息
- 音频播放：响应中的 `audio_data` 通过 `createMiniPlayer()` 创建内联播放器

---

## omni.html — Omni 全双工

视频 + 音频全双工交互页面。

### 媒体提供者

**LiveMediaProvider**（摄像头模式）：
- `getUserMedia({video, audio})` 获取摄像头和麦克风
- 支持前后摄像头切换（`flipCamera()`）
- 支持镜像模式（`_globalMirror`）
- 视频帧捕获：Canvas `drawImage()` → JPEG Base64（质量 0.7）

**FileMediaProvider**（文件模式）：
- 处理视频文件输入
- 预提取帧：`_extractFrames()` 按时间点提取
- 音频解码并重采样到 16kHz
- 三种音频源：`video`（文件音频）/ `mic`（麦克风）/ `mixed`（混合）

### 数据发送

每秒发送一个 chunk：

```javascript
media.onChunk = (chunk) => {
    const msg = {
        type: 'audio_chunk',
        audio_base64: arrayBufferToBase64(chunk.audio.buffer)
    };
    if (chunk.frameBase64) {
        msg.frame_base64_list = [chunk.frameBase64];
    }
    session.sendChunk(msg);
};
```

### UI 功能

- 视频全屏模式
- 实时字幕叠加
- `MetricsPanel` 实时指标
- `MixerController` 音频混音
- `SessionVideoRecorder` 视频录制

---

## audio_duplex.html — 音频全双工

纯音频全双工页面，与 Omni 共享大部分 duplex 库。

### 与 Omni 的区别

| 特性 | Omni | Audio Duplex |
|------|------|-------------|
| 视频帧 | 支持（摄像头/文件） | 无 |
| 波形可视化 | 无 | 有（AnalyserNode 实时绘制） |
| 文件模式 | 视频文件 | 音频文件（FileAudioProvider） |
| 录制 | SessionVideoRecorder | SessionRecorder（立体声 WAV） |

### 波形可视化

使用 `AnalyserNode` 获取时域数据，通过 `requestAnimationFrame` 循环绘制实时波形。

### FileAudioProvider

处理音频文件输入：
- 解码音频并重采样到 16kHz
- LUFS 归一化
- 支持 `mixed` 模式（文件音频 + 麦克风混合）

---

## admin.html — 管理面板

- Worker 状态监控（在线/离线/忙碌/Duplex 状态）
- 队列状态管理（查看/取消排队项）
- 应用启用/禁用开关
- ETA 配置编辑（基准值 + EMA 参数）
- 定时自动刷新

---

## session-viewer.html — 会话回放

- 从 `/api/sessions/{sid}` 加载元数据和录制数据
- 回放音频/视频（`merged_replay.wav` / `.mp4`）
- 显示对话文本时间线
- 支持通过 URL 分享（`/s/{session_id}`）
