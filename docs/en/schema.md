# Schemas & Processors

This module defines the system's data models (Pydantic Schemas), unified processor abstractions, and the capability declaration system. It serves as the bridge between the Worker service layer and the underlying model.

## Module Structure

```
core/
├── __init__.py              # Module entry point, unified exports
├── capabilities.py          # Capability declaration system
├── factory.py               # Factory pattern
├── internal/
│   └── __init__.py          # Internal implementation (reserved)
├── modes/
│   └── __init__.py          # Inference modes (reserved)
├── schemas/
│   ├── __init__.py          # Schema exports
│   ├── common.py            # Common type definitions
│   ├── streaming.py         # Streaming request/response
│   └── duplex.py            # Duplex request/response
└── processors/
    ├── __init__.py           # Processor exports
    ├── base.py               # Base class & Mixin
    └── unified.py            # Unified processor implementation
```

---

## schemas/ — Pydantic Data Models

### common.py — Common Types

#### Enumerations

| Enum | Values | Description |
|------|--------|-------------|
| `Role` | `system`, `user`, `assistant` | Message role |
| `TTSMode` | `default`, `audio_assistant`, `omni`, `audio_roleplay`, `voice_cloning` | TTS mode |
| `ContentType` | `text`, `image`, `audio`, `video` | Content type |

#### Content Models

| Model | Key Fields | Description |
|-------|-----------|-------------|
| `TextContent` | `type="text"`, `text: str` | Text content |
| `ImageContent` | `type="image"`, `data: str` | Image content (Base64) |
| `AudioContent` | `type="audio"`, `data: str`, `sample_rate: int = 16000` | Audio content (Base64 PCM float32) |
| `VideoContent` | `type="video"`, `data: str`, `stack_frames: int = 1` | Video content (Base64 video file, auto-extracts frames and audio) |

Type alias: `ContentItem = Union[TextContent, ImageContent, AudioContent, VideoContent]`

#### Message Model

```python
class Message(BaseModel):
    role: Role
    content: Union[str, List[ContentItem]]
```

`content` supports either a plain text string or a multimodal content list.

#### Configuration Models

| Model | Key Fields | Description |
|-------|-----------|-------------|
| `TTSSamplingParams` | `top_p=0.85`, `temperature=0.8`, `repetition_penalty=1.05` | TTS sampling parameters |
| `TTSConfig` | `enabled`, `mode`, `ref_audio_path`, `ref_audio_data`, `language` | TTS configuration |
| `ImageConfig` | `max_slice_nums`, `use_image_id` | Image processing configuration |
| `GenerationConfig` | `max_new_tokens=512`, `temperature=0.7`, `top_p=0.8` | Generation configuration |

### streaming.py — Streaming Schema

Data structures for streaming generation (`StreamingChunk`, etc.), reused by ChatView's `streaming_generate()`.

### duplex.py — Duplex Mode Schema

#### DuplexConfig

| Field | Default | Description |
|-------|---------|-------------|
| `generate_audio` | `True` | Generate audio |
| `ls_mode` | `"explicit"` | Listen/Speak mode |
| `force_listen_count` | `3` | Startup protection period (force listen for first N steps) |
| `max_new_speak_tokens_per_chunk` | `20` | Maximum speak tokens per chunk |
| `temperature` | `0.7` | Generation temperature |
| `top_k` | `20` | Top-K sampling |
| `top_p` | `0.8` | Top-P sampling |
| `listen_prob_scale` | `1.0` | Listen probability scale |
| `chunk_ms` | `1000` | Audio chunk duration (milliseconds) |
| `sample_rate` | `16000` | Sample rate |

#### DuplexPrepareRequest

Initialize duplex session: `prefix_system_prompt`, `suffix_system_prompt`, `ref_audio_path`, `prompt_wav_path`.

#### DuplexPrefillRequest

Per-step prefill: `audio_waveform` (Base64), `audio_path`, `frame_list` (video frame list), `max_slice_nums`.

#### DuplexGenerateResult

Single-step generation result:

| Field | Description |
|-------|-------------|
| `is_listen` | Whether in listening state |
| `text` | Generated text |
| `audio_data` | Base64 audio (24kHz) |
| `end_of_turn` | Whether the turn has ended |
| `current_time` | Current timestamp |
| `cost_llm_ms` | LLM inference latency |
| `cost_tts_ms` | TTS latency |
| `cost_all_ms` | Total latency |
| `n_tokens` | Number of generated LLM tokens |
| `n_tts_tokens` | Number of generated TTS tokens |

---

## processors/ — Inference Processors

### base.py — Base Class

#### BaseProcessor (Abstract Base Class)

```python
class BaseProcessor(ABC):
    def __init__(self, model_path: str, device: str = "cuda")
    
    @property
    @abstractmethod
    def mode(self) -> ProcessorMode: ...
    
    @property
    def capabilities(self) -> ProcessorCapabilities: ...
    
    @abstractmethod
    def _load_model(self) -> None: ...
    
    @abstractmethod
    def _release_resources(self) -> None: ...
    
    def is_ready(self) -> bool: ...
```

#### MiniCPMOProcessorMixin

A Mixin providing shared functionality for processors:

| Method | Description |
|--------|-------------|
| `_load_ref_audio(path, cache)` | Load reference audio (with caching support) |
| `_convert_content_to_model_format(content)` | Convert Schema content to model format |
| `_convert_messages_to_model_format(messages, tts_config)` | Convert message list to model format |
| `_resolve_ref_audio(tts_config)` | Resolve reference audio (path or Base64) |
| `_init_tts_mode(streaming)` | Initialize TTS mode |

### unified.py — Unified Processor

#### UnifiedProcessor

A unified processor that loads the model once and supports millisecond-level hot-switching between Streaming and Duplex modes.

```python
class UnifiedProcessor(BaseProcessor):
    def __init__(
        self, model_path, pt_path=None, device="cuda",
        ref_audio_path=None, duplex_config=None,
        preload_both_tts=True, compile=False,
        attn_implementation="auto"
    )
```

| Method/Property | Description |
|----------------|-------------|
| `_load_model()` | Load model and TTS |
| `_release_resources()` | Release all resources |
| `set_half_duplex_mode()` | Switch to Half-Duplex mode, returns `HalfDuplexView` |
| `set_duplex_mode()` | Switch to Duplex mode, returns `DuplexView` |
| `half_duplex` | HalfDuplexView property |
| `duplex` | DuplexView property |
| `kv_cache_length` | Current KV Cache length |

#### ChatView

ChatView provides the dedicated API for Turn-based Chat, supporting both streaming and non-streaming generation modes.

| Method/Property | Description |
|----------------|-------------|
| `prefill(session_id, msgs, ...)` | Prefill all messages into KV Cache in one shot |
| `generate(session_id, ...)` | Non-streaming generation (HF generate + TTS) |
| `streaming_generate(session_id, ...)` | Streaming generation, returns `Generator[StreamingChunk]` |
| `kv_cache_length` | Current KV Cache length |
| `chat(request)` | Legacy interface (combined prefill + generate) |

#### DuplexView

| Method | Description |
|--------|-------------|
| `prepare(system_prompt_text, ref_audio_path, prompt_wav_path)` | Prepare duplex session |
| `prefill(audio_waveform, audio_path, frame_list, max_slice_nums)` | Prefill audio/video |
| `generate(force_listen)` | Generate one step, returns `DuplexGenerateResult` |
| `finalize()` | End current turn |
| `set_break()` / `clear_break()` | Set/clear interrupt flag |
| `stop()` | Stop session |
| `is_break_set()` / `is_stopped()` | Query status |
| `cleanup()` | Clean up resources, release GPU memory |
| `offline_inference(task_input)` | Offline inference |

---

## capabilities.py — Capability Declaration System

### ProcessorMode Enum

```python
class ProcessorMode(Enum):
    HALF_DUPLEX = "half_duplex"
    DUPLEX = "duplex"
```

### ProcessorCapabilities Data Class

```python
@dataclass(frozen=True)
class ProcessorCapabilities:
    mode: ProcessorMode
    # Input capabilities
    supports_text: bool
    supports_image: bool
    supports_audio: bool
    supports_video: bool
    # Output capabilities
    supports_text_output: bool
    supports_audio_output: bool
    supports_streaming_output: bool
    # Interaction capabilities
    supports_multi_turn: bool
    supports_interrupt: bool
    supports_rollback: bool
    # Resource requirements
    requires_exclusive_worker: bool
    supports_kv_cache_reuse: bool
```

### Predefined Capability Constants

| Capability | Streaming | Duplex |
|------------|-----------|--------|
| Streaming output | Supported | Supported |
| Interrupt support | -- | Supported |
| Rollback support | Supported | -- |
| KV Cache reuse | Supported | Supported |
| Exclusive Worker | -- | Required |
| Multi-turn dialogue | Supported | Supported |

### Helper Functions

- `get_capabilities(mode)` — Get capability declaration for a given mode
- `supports_feature(mode, feature)` — Check if a mode supports a specific feature

---

## factory.py — Factory Pattern

### ProcessorFactory

| Method | Description |
|--------|-------------|
| `get_processor(...)` | Get or create a UnifiedProcessor (with caching) |
| `create(mode, ...)` | Create a View for the specified mode |
| `from_config(config)` | Create a View from a configuration dictionary |

Convenience function: `create_processor(mode, ...)` — Quickly create a View for the specified mode.
