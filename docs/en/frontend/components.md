# Frontend UI Components

## Shared Components (shared/)

### app-nav.js — Dynamic Navigation Bar

Fetches the list of enabled applications from `/api/apps` and dynamically renders navigation links.

- Automatically highlights the current page
- Redirects to home page when accessing a disabled application
- Responsive design (collapses on mobile)

### preset-selector.js — Preset Selector

The `PresetSelector` component manages system prompt presets.

- Loads preset list from `/api/presets`
- Auto-fills system prompt on dropdown selection
- Remembers last selection using localStorage

### save-share.js — Save & Share

`SaveShareUI` manages session saving and sharing:

- Uploads recordings to `/api/sessions/{sid}/upload-recording`
- Generates share links `/s/{session_id}`
- Recent sessions stored in localStorage (up to 20 entries)

---

## Content Editors

### ref-audio-player.js — Reference Audio Player

A reusable audio player component for reference audio preview:

- Upload: file selection or drag-and-drop
- Decode: auto-decodes and resamples to 16kHz mono
- Playback: play/pause, draggable progress bar, duration display
- Internally uses `AudioContext` for decoding, outputs Float32 PCM

### system-content-editor.js — System Content Editor

A list-based editor for managing multimodal system prompt content:

- **Text items**: Editable text
- **Audio items**: Reference audio, integrated with `RefAudioPlayer`
- Supports add/delete/drag-and-drop reordering

### user-content-editor.js — User Message Editor

A multimodal user input editor:

| Content Type | Input Method |
|-------------|-------------|
| Text | Text input field |
| Audio | Recording / file upload / drag-and-drop |
| Image | File upload / drag-and-drop / paste |
| Video | File upload / drag-and-drop (Chat mode only, auto-enables omni_mode) |

**Keyboard shortcuts**: `Space` to record (click to toggle / long press for push-to-talk), `ESC` to cancel.

---

## Duplex UI Components (duplex/ui/)

### duplex-ui.js — Metrics Panel + Settings Persistence

**MetricsPanel** real-time metrics panel:

| Metric | Color Thresholds |
|--------|-----------------|
| Latency | Green <200ms / Yellow <500ms / Red |
| TTFS (Time to First Sound) | Green <300ms / Yellow <600ms / Red |
| Drift | Green <50ms / Yellow <200ms / Red |
| Gaps | Green =0 / Yellow <3 / Red |
| KV Cache | Green <4096 / Yellow <6144 / Red |

**SettingsPersistence** settings persistence: declarative field definitions, automatic localStorage store/restore.

### ref-audio-init.js — Reference Audio Initialization

Initializes reference audio when a duplex page loads: fetches default and custom reference audio lists from the backend and builds a dropdown selector.

### tts-ref-controller.js — TTS Reference Audio Controller

Manages TTS reference audio selection, upload, and deletion. Supports using different audio for LLM ref and TTS ref.

---

## Common Utilities (lib/)

### countdown-timer.js — Countdown State Machine

A UI-agnostic countdown state machine that reports status via callbacks. Used for displaying queue wait times.

### chat-eta-estimator.js — ETA Estimator

EMA-based estimation using historical response times, with independent estimation for Chat/Streaming modes.
