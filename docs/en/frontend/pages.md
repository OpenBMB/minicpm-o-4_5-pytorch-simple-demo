# Frontend Pages & Routing

## Dynamic Navigation System (app-nav.js)

The `AppNav` component fetches the list of enabled applications from `/api/apps` and dynamically renders navigation links. It automatically redirects to the home page when accessing a disabled application.

---

## index.html — Home Page

- Displays cards for three interaction modes, showing mode names and features
- Fetches enabled status from `/api/apps`, graying out disabled modes
- Displays a recent session list (from localStorage, managed by `SaveShareUI`)

---

## turnbased.html — Turn-based Chat

The most complex non-duplex page, supporting both Chat and Streaming modes.

### Global State Object

```javascript
const state = {
    messages: [],                // Message list {role, content, displayText}
    systemContentList: [],       // System content list (text + audio + image + video)
    isGenerating: false,         // Whether generation is in progress
    generationPhase: 'idle',     // 'idle' | 'queuing' | 'generating'
    currentTicketId: null,       // Queue ticket ID (streaming only)
    abortController: null,       // Fetch abort (chat only)
    streamingWs: null,           // WebSocket connection (streaming only)
    requestId: 'req_' + Date.now(),
    currentView: 'initial',      // 'initial' | 'conversation'
    editingIndex: -1,            // Index of the message being edited
    ttsRefAudioMode: 'extract',  // 'extract' | 'independent'
    ttsRefAudioData: null,       // TTS reference audio base64
};
```

### Streaming Mode

Streaming chat via WebSocket (`WS /ws/streaming/{request_id}`):

1. Establish WebSocket connection
2. Send `prefill` message (with message history + ref_audio_base64)
3. Send `generate` after receiving `prefill_done`
4. Receive `StreamingChunk` incrementally (text_delta + audio_data)
5. Render the complete result after receiving `done`
6. Display `CountdownTimer` while queuing

### Message Construction Flow

1. Get user input items from `UserContentEditor`
2. Audio Blob → resample to 16kHz mono → Base64 PCM float32
3. Image File → Base64
4. Video File → Base64 (backend auto-extracts frames and audio, requires `omni_mode: true`)
5. Build `content list` format: `[{type:"text", text:...}, {type:"audio", data:...}, {type:"video", data:...}, ...]`
6. `buildRequestMessages()` assembles the complete message list (including system prompt)

### TTS Reference Audio

Two modes:
- **extract**: Extract from system content (requires exactly 1 audio clip <20s)
- **independent**: Upload reference audio independently

Sent to the Worker via the `tts_ref_audio_base64` field in the Streaming `prefill` message.

### Response Rendering

- `addMessageUI()` — Adds a message bubble to the chat area
- `updateLastAssistantMessage()` — Streams updates to the last assistant message
- Audio playback: `audio_data` in the response is played via an inline player created by `createMiniPlayer()`

---

## omni.html — Omni Full-Duplex

Video + audio full-duplex interaction page.

### Media Providers

**LiveMediaProvider** (camera mode):
- `getUserMedia({video, audio})` to access camera and microphone
- Supports front/rear camera switching (`flipCamera()`)
- Supports mirror mode (`_globalMirror`)
- Video frame capture: Canvas `drawImage()` → JPEG Base64 (quality 0.7)

**FileMediaProvider** (file mode):
- Handles video file input
- Pre-extracts frames: `_extractFrames()` extracts at time points
- Decodes and resamples audio to 16kHz
- Three audio sources: `video` (file audio) / `mic` (microphone) / `mixed` (blended)

### Data Transmission

Sends one chunk per second:

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

### UI Features

- Video fullscreen mode
- Real-time subtitle overlay
- `MetricsPanel` real-time metrics
- `MixerController` audio mixing
- `SessionVideoRecorder` video recording

---

## audio_duplex.html — Audio Full-Duplex

Audio-only full-duplex page, sharing most of the duplex library with Omni.

### Differences from Omni

| Feature | Omni | Audio Duplex |
|---------|------|-------------|
| Video frames | Supported (camera/file) | None |
| Waveform visualization | None | Yes (AnalyserNode real-time rendering) |
| File mode | Video files | Audio files (FileAudioProvider) |
| Recording | SessionVideoRecorder | SessionRecorder (stereo WAV) |

### Waveform Visualization

Uses `AnalyserNode` to get time-domain data and renders a real-time waveform via `requestAnimationFrame` loop.

### FileAudioProvider

Handles audio file input:
- Decodes audio and resamples to 16kHz
- LUFS normalization
- Supports `mixed` mode (file audio + microphone blended)

---

## admin.html — Admin Panel

- Worker status monitoring (online/offline/busy/duplex status)
- Queue status management (view/cancel queued items)
- Application enable/disable toggles
- ETA configuration editing (baseline values + EMA parameters)
- Timed auto-refresh

---

## session-viewer.html — Session Replay

- Loads metadata and recording data from `/api/sessions/{sid}`
- Plays back audio/video (`merged_replay.wav` / `.mp4`)
- Displays conversation text timeline
- Supports sharing via URL (`/s/{session_id}`)
