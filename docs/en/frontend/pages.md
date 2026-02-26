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

The Turn-based Chat page communicates with the backend through the `/ws/chat` WebSocket.

### State Management

```javascript
const state = {
    messages: [],                // Message list {role, content, displayText}
    systemContentList: [],       // System content list (text + audio + image + video)
    isGenerating: false,         // Whether generation is in progress
    generationPhase: 'idle',     // 'idle' | 'generating'
};
```

### Message Building Flow

1. User inputs text / records audio / uploads images, videos
2. Audio Blob → resample to 16kHz mono → Base64 PCM float32
3. Image File → Base64
4. Video File → Base64 (backend auto-extracts frames and audio, requires `omni_mode: true`)
5. Build `content list` format: `[{type:"text", text:...}, {type:"audio", data:...}, {type:"video", data:...}, ...]`
6. `buildRequestMessages()` assembles the complete message list (including system prompt)

### Communication Mode

All requests are sent through the `/ws/chat` WebSocket with the following protocol:

1. Connect to WebSocket
2. Send JSON (with `messages`, `streaming`, `tts` and other parameters)
3. Wait for generation results after receiving `prefill_done`
4. `streaming=true`: Receive `{type:"chunk", text_delta, audio_data}` chunks, render in real time
5. `streaming=false`: Receive `{type:"done", text, audio_data}`, render all at once
6. Audio playback via `StreamingAudioPlayer` (streaming) or `createMiniPlayer` (non-streaming)

### Frontend Toggles

| Toggle | Description |
|--------|-------------|
| Streaming | Real-time token-by-token output vs one-shot response |
| Voice Response | Generate voice reply; suppresses special character output when enabled |

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
