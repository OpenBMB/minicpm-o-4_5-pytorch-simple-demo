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

Turn-based Chat 页面，统一通过 `/ws/chat` WebSocket 与后端通信。

### 状态管理

```javascript
const state = {
    messages: [],                // 消息列表 {role, content, displayText}
    systemContentList: [],       // 系统内容列表 (text + audio + image + video)
    isGenerating: false,         // 是否正在生成
    generationPhase: 'idle',     // 'idle' | 'generating'
};
```

### 消息构建流程

1. 用户输入文本 / 录音 / 上传图片、视频
2. 音频 Blob → 重采样到 16kHz mono → Base64 PCM float32
3. 图片 File → Base64
4. 视频 File → Base64（后端自动提取帧和音频，需 `omni_mode: true`）
5. 构建 `content list` 格式：`[{type:"text", text:...}, {type:"audio", data:...}, {type:"video", data:...}, ...]`
6. `buildRequestMessages()` 组装完整消息列表（含 system prompt）

### 通信模式

所有请求通过 `/ws/chat` WebSocket 发送，协议：

1. 连接 WebSocket
2. 发送 JSON（含 `messages`、`streaming`、`tts` 等参数）
3. 收到 `prefill_done` 后等待生成结果
4. `streaming=true`：逐 chunk 收到 `{type:"chunk", text_delta, audio_data}`，实时渲染
5. `streaming=false`：收到 `{type:"done", text, audio_data}`，一次性渲染
6. 音频通过 `StreamingAudioPlayer`（流式）或 `createMiniPlayer`（非流式）播放

### 前端开关

| 开关 | 说明 |
|------|------|
| Streaming | 实时逐字输出 vs 一次性返回 |
| Voice Response | 生成语音回复；开启时抑制特殊字符输出 |

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
