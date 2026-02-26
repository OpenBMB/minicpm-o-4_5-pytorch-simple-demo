# Schema 与处理器

本模块定义了系统的数据模型（Pydantic Schema）、统一处理器抽象和能力声明系统，是连接 Worker 服务层与底层模型的桥梁。

## 模块结构

```
core/
├── __init__.py              # 模块入口，统一导出
├── capabilities.py          # 能力声明系统
├── factory.py               # 工厂模式
├── internal/
│   └── __init__.py          # 内部实现（预留）
├── modes/
│   └── __init__.py          # 推理模式（预留）
├── schemas/
│   ├── __init__.py          # Schema 导出
│   ├── common.py            # 公共类型定义
│   ├── streaming.py         # Streaming 请求/响应
│   └── duplex.py            # Duplex 请求/响应
└── processors/
    ├── __init__.py           # 处理器导出
    ├── base.py               # 基类与 Mixin
    └── unified.py            # 统一处理器实现
```

---

## schemas/ — Pydantic 数据模型

### common.py — 公共类型

#### 枚举

| 枚举 | 值 | 说明 |
|------|-----|------|
| `Role` | `system`, `user`, `assistant` | 消息角色 |
| `TTSMode` | `default`, `audio_assistant`, `omni`, `audio_roleplay`, `voice_cloning` | TTS 模式 |
| `ContentType` | `text`, `image`, `audio`, `video` | 内容类型 |

#### 内容模型

| 模型 | 关键字段 | 说明 |
|------|---------|------|
| `TextContent` | `type="text"`, `text: str` | 文本内容 |
| `ImageContent` | `type="image"`, `data: str` | 图像内容（Base64） |
| `AudioContent` | `type="audio"`, `data: str`, `sample_rate: int = 16000` | 音频内容（Base64 PCM float32） |
| `VideoContent` | `type="video"`, `data: str`, `stack_frames: int = 1` | 视频内容（Base64 视频文件，自动提取帧和音频） |

类型别名：`ContentItem = Union[TextContent, ImageContent, AudioContent, VideoContent]`

#### Message 模型

```python
class Message(BaseModel):
    role: Role
    content: Union[str, List[ContentItem]]
```

`content` 支持纯文本字符串或多模态内容列表。

#### 配置模型

| 模型 | 关键字段 | 说明 |
|------|---------|------|
| `TTSSamplingParams` | `top_p=0.85`, `temperature=0.8`, `repetition_penalty=1.05` | TTS 采样参数 |
| `TTSConfig` | `enabled`, `mode`, `ref_audio_path`, `ref_audio_data`, `language` | TTS 配置 |
| `ImageConfig` | `max_slice_nums`, `use_image_id` | 图像处理配置 |
| `GenerationConfig` | `max_new_tokens=512`, `temperature=0.7`, `top_p=0.8` | 生成配置 |

### streaming.py — Streaming Schema

流式生成使用的数据结构（`StreamingChunk` 等），被 ChatView 的 `streaming_generate()` 复用。

### duplex.py — Duplex 模式 Schema

#### DuplexConfig

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `generate_audio` | `True` | 生成音频 |
| `ls_mode` | `"explicit"` | Listen/Speak 模式 |
| `force_listen_count` | `3` | 启动保护期（前 N 步强制聆听） |
| `max_new_speak_tokens_per_chunk` | `20` | 每 chunk 最大说话 token 数 |
| `temperature` | `0.7` | 生成温度 |
| `top_k` | `20` | Top-K 采样 |
| `top_p` | `0.8` | Top-P 采样 |
| `listen_prob_scale` | `1.0` | 聆听概率缩放 |
| `chunk_ms` | `1000` | 音频 chunk 时长（毫秒） |
| `sample_rate` | `16000` | 采样率 |

#### DuplexPrepareRequest

初始化双工会话：`prefix_system_prompt`, `suffix_system_prompt`, `ref_audio_path`, `prompt_wav_path`。

#### DuplexPrefillRequest

每步预填充：`audio_waveform`（Base64）, `audio_path`, `frame_list`（视频帧列表）, `max_slice_nums`。

#### DuplexGenerateResult

单步生成结果：

| 字段 | 说明 |
|------|------|
| `is_listen` | 是否处于聆听状态 |
| `text` | 生成的文本 |
| `audio_data` | Base64 音频（24kHz） |
| `end_of_turn` | 是否 turn 结束 |
| `current_time` | 当前时间戳 |
| `cost_llm_ms` | LLM 推理耗时 |
| `cost_tts_ms` | TTS 耗时 |
| `cost_all_ms` | 总耗时 |
| `n_tokens` | 生成的 LLM token 数 |
| `n_tts_tokens` | 生成的 TTS token 数 |

---

## processors/ — 推理处理器

### base.py — 基类

#### BaseProcessor（抽象基类）

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

为处理器提供共用功能的 Mixin：

| 方法 | 说明 |
|------|------|
| `_load_ref_audio(path, cache)` | 加载参考音频（支持缓存） |
| `_convert_content_to_model_format(content)` | 将 Schema 内容转换为模型格式 |
| `_convert_messages_to_model_format(messages, tts_config)` | 将消息列表转换为模型格式 |
| `_resolve_ref_audio(tts_config)` | 解析参考音频（路径或 Base64） |
| `_init_tts_mode(streaming)` | 初始化 TTS 模式 |

### unified.py — 统一处理器

#### UnifiedProcessor

统一处理器，一次加载模型，支持 Streaming / Duplex 两种模式毫秒级热切换。

```python
class UnifiedProcessor(BaseProcessor):
    def __init__(
        self, model_path, pt_path=None, device="cuda",
        ref_audio_path=None, duplex_config=None,
        preload_both_tts=True, compile=False,
        attn_implementation="auto"
    )
```

| 方法/属性 | 说明 |
|----------|------|
| `_load_model()` | 加载模型和 TTS |
| `_release_resources()` | 释放所有资源 |
| `set_streaming_mode()` | 切换到 Streaming 模式，返回 `StreamingView` |
| `set_duplex_mode()` | 切换到 Duplex 模式，返回 `DuplexView` |
| `streaming` | StreamingView 属性 |
| `duplex` | DuplexView 属性 |
| `kv_cache_length` | 当前 KV Cache 长度 |

#### ChatView

ChatView 提供 Turn-based Chat 的专用 API，支持流式和非流式两种生成模式。

| 方法/属性 | 说明 |
|----------|------|
| `prefill(session_id, msgs, ...)` | 一次性 prefill 所有消息到 KV Cache |
| `generate(session_id, ...)` | 非流式生成（HF generate + TTS） |
| `streaming_generate(session_id, ...)` | 流式生成，返回 `Generator[StreamingChunk]` |
| `kv_cache_length` | 当前 KV Cache 长度 |
| `chat(request)` | 兼容旧接口（一体式 prefill + generate） |

#### DuplexView

| 方法 | 说明 |
|------|------|
| `prepare(system_prompt_text, ref_audio_path, prompt_wav_path)` | 准备双工会话 |
| `prefill(audio_waveform, audio_path, frame_list, max_slice_nums)` | 预填充音频/视频 |
| `generate(force_listen)` | 生成一步，返回 `DuplexGenerateResult` |
| `finalize()` | 结束当前 turn |
| `set_break()` / `clear_break()` | 设置/清除中断标志 |
| `stop()` | 停止会话 |
| `is_break_set()` / `is_stopped()` | 查询状态 |
| `cleanup()` | 清理资源，释放 GPU 显存 |
| `offline_inference(task_input)` | 离线推理 |

---

## capabilities.py — 能力声明系统

### ProcessorMode 枚举

```python
class ProcessorMode(Enum):
    STREAMING = "streaming"
    DUPLEX = "duplex"
```

### ProcessorCapabilities 数据类

```python
@dataclass(frozen=True)
class ProcessorCapabilities:
    mode: ProcessorMode
    # 输入能力
    supports_text: bool
    supports_image: bool
    supports_audio: bool
    supports_video: bool
    # 输出能力
    supports_text_output: bool
    supports_audio_output: bool
    supports_streaming_output: bool
    # 交互能力
    supports_multi_turn: bool
    supports_interrupt: bool
    supports_rollback: bool
    # 资源需求
    requires_exclusive_worker: bool
    supports_kv_cache_reuse: bool
```

### 预定义能力常量

| 能力 | Streaming | Duplex |
|------|-----------|--------|
| 流式输出 | 支持 | 支持 |
| 打断支持 | -- | 支持 |
| 回溯支持 | 支持 | -- |
| KV Cache 复用 | 支持 | 支持 |
| 独占 Worker | -- | 需要 |
| 多轮对话 | 支持 | 支持 |

### 辅助函数

- `get_capabilities(mode)` — 根据模式获取能力声明
- `supports_feature(mode, feature)` — 检查某模式是否支持特定特性

---

## factory.py — 工厂模式

### ProcessorFactory

| 方法 | 说明 |
|------|------|
| `get_processor(...)` | 获取或创建 UnifiedProcessor（带缓存） |
| `create(mode, ...)` | 创建指定模式的 View |
| `from_config(config)` | 从配置字典创建 View |

便捷函数：`create_processor(mode, ...)` — 快速创建指定模式的 View。
