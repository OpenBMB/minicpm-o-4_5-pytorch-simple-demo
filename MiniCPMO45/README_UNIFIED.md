# MiniCPMO Unified Model

本文档说明 `modeling_minicpmo_unified.py` 中的统一模型设计。

## 背景

原始实现中，单工（Chat/Streaming）和双工（Duplex）分别由不同的类实现：
- `MiniCPMOForCausalLM`：单工对话
- `MiniCPMODuplex`：双工对话

这导致两个问题：
1. **无法热切换**：切换模式需要重新加载模型参数
2. **代码重复**：两个类共享大量底层逻辑

## 解决方案

`modeling_minicpmo_unified.py` 提供统一的 `MiniCPMO` 类，采用**组合模式**：

```
MiniCPMO（统一入口）
├── 继承自 MiniCPMOForCausalLM（单工能力）
└── 组合 DuplexCapability（双工能力）
```

## 核心类

### MiniCPMO

统一模型类，支持单工和双工模式热切换。

```python
from MiniCPMO45.modeling_minicpmo_unified import MiniCPMO

# 加载模型
model = MiniCPMO.from_pretrained(model_path)
model.init_unified(device="cuda")

# ========== 单工模式 ==========
# Chat（离线）
response = model.chat(msgs, tokenizer=tokenizer)

# Streaming（在线）
model.streaming_prefill(msgs, tokenizer=tokenizer)
for chunk in model.streaming_generate():
    print(chunk["text"])

# ========== 双工模式 ==========
# 离线推理
result = model.duplex_chat(
    user_audio=audio_16k,
    system_prompt="你是一个友好的助手。",
    ref_audio_path="/path/to/ref.wav",
)
print(result["full_text"])

# 在线流式
model.duplex_prepare(prefix_system_prompt="...", prompt_wav_path="...")
model.duplex_prefill(audio_waveform=chunk)
result = model.duplex_generate()
```

### DuplexCapability

双工能力组件，封装双工对话的全部逻辑。

采用组合模式，接受外部传入的 `MiniCPMO` 实例，不自己加载模型。

## API 详细说明

### 1. chat() - 单工离线对话

支持文本、图像、音频输入，可选 TTS 输出。

```python
# 基础文本对话
msgs = [{"role": "user", "content": "你好，介绍一下你自己"}]
response = model.chat(msgs, tokenizer=tokenizer, max_new_tokens=512)
print(response)  # 文本输出

# 带图像的对话
from PIL import Image
img = Image.open("image.jpg")
msgs = [{"role": "user", "content": [img, "这张图片里有什么？"]}]
response = model.chat(msgs, tokenizer=tokenizer)

# 带音频输入的对话
import librosa
audio, sr = librosa.load("audio.wav", sr=16000)
msgs = [{"role": "user", "content": [audio, "请转录这段音频"]}]
response = model.chat(msgs, tokenizer=tokenizer)

# 带 TTS 输出的对话（生成语音）
response, audio_wav = model.chat(
    msgs, 
    tokenizer=tokenizer,
    generate_audio=True,
    use_tts_template=True,
    tts_sampling_params={"temperature": 0.3, "top_p": 0.7},
)
# audio_wav 是 numpy array，可直接保存或播放
```

**关键参数**：
- `msgs`: 对话消息列表，支持文本/图像/音频混合
- `tokenizer`: 分词器
- `max_new_tokens`: 最大生成 token 数
- `generate_audio`: 是否生成语音（默认 False）
- `use_tts_template`: 使用 TTS 模板（生成语音时设为 True）
- `output_audio_path`: 保存音频的路径（可选）

---

### 2. streaming_prefill() / streaming_generate() - 单工在线流式

用于实时流式生成文本（和可选的语音）。

```python
# 准备消息
msgs = [{"role": "user", "content": "讲一个关于AI的故事"}]

# Step 1: 预填充
model.streaming_prefill(
    msgs=msgs,
    tokenizer=tokenizer,
    max_new_tokens=1024,
)

# Step 2: 流式生成
for chunk in model.streaming_generate(
    decode_text=True,
    tokenizer=tokenizer,
):
    print(chunk["text"], end="", flush=True)
    # chunk 包含: {"text": str, "token_id": int, ...}

# 带 TTS 的流式生成
model.streaming_prefill(
    msgs=msgs,
    tokenizer=tokenizer,
    use_tts_template=True,
)
for chunk in model.streaming_generate(
    decode_text=True,
    tokenizer=tokenizer,
    generate_audio=True,
):
    print(chunk["text"], end="", flush=True)
    if chunk.get("audio"):
        play_audio(chunk["audio"])  # 实时播放音频
```

**streaming_prefill 关键参数**：
- `msgs`: 对话消息
- `tokenizer`: 分词器
- `max_new_tokens`: 最大生成 token 数
- `use_tts_template`: 是否使用 TTS 模板

**streaming_generate 关键参数**：
- `decode_text`: 是否解码文本（默认 True）
- `tokenizer`: 分词器
- `generate_audio`: 是否生成音频
- `temperature`, `top_k`, `top_p`: 采样参数

---

### 3. duplex_chat() - 双工离线推理

对完整音频文件进行离线双工对话，一站式处理。

```python
import librosa

# 加载用户音频
user_audio, _ = librosa.load("user_speech.wav", sr=16000, mono=True)

# 执行离线双工推理
result = model.duplex_chat(
    user_audio=user_audio,
    system_prompt="Streaming Duplex Conversation! You are a helpful assistant.",
    ref_audio_path="/path/to/ref_voice.wav",  # TTS 参考音色
    chunk_ms=1000,  # 每个 chunk 时长（毫秒）
    force_listen_count=3,  # 强制 listen 的 chunk 数
)

print(f"成功: {result['success']}")
print(f"完整文本: {result['full_text']}")
print(f"分块数: {len(result['chunks'])}")

# 播放生成的音频
for audio_chunk in result["audio_chunks"]:
    play_audio(audio_chunk)
```

**关键参数**：
- `user_audio`: 用户音频波形（16kHz numpy array）
- `system_prompt`: 系统提示
- `ref_audio_path` 或 `ref_audio`: TTS 参考音频
- `chunk_ms`: 每个 chunk 的时长（默认 1000ms）
- `force_listen_count`: 强制 listen 的 chunk 数（默认 3）
- `image_list`: 图像列表（视频双工场景）

**返回值**：
```python
{
    "success": bool,
    "full_text": str,  # 完整输出文本
    "chunks": List[dict],  # 每个 chunk 的详情
    "audio_chunks": List[np.ndarray],  # 生成的音频块
    "error": Optional[str],
}
```

---

### 4. duplex_prepare() / duplex_prefill() / duplex_generate() - 双工在线流式

用于实时双工对话，支持边听边说。

```python
# Step 1: 准备会话
model.duplex_prepare(
    prefix_system_prompt="Streaming Duplex Conversation! You are a helpful assistant.",
    prompt_wav_path="/path/to/ref_voice.wav",  # TTS 参考音色
)

# Step 2: 循环处理音频块
for audio_chunk in realtime_audio_stream:  # 实时音频流
    # 预填充当前音频块
    model.duplex_prefill(
        audio_waveform=audio_chunk,  # 16kHz numpy array, 1秒
        frame_list=None,  # 或 [PIL.Image] 用于视频双工
    )
    
    # 生成响应
    result = model.duplex_generate(
        decode_mode="greedy",  # 或 "sample"
        temperature=0.7,
        top_k=20,
        top_p=0.8,
    )
    
    if result["is_listen"]:
        print("[听]")
    else:
        print(f"[说] {result['text']}")
        if result.get("audio"):
            play_audio(result["audio"])
    
    if result["end_of_turn"]:
        print("[轮次结束]")
        break

# Step 3: 停止会话
model.duplex_stop()
```

**duplex_prepare 关键参数**：
- `prefix_system_prompt`: 系统提示
- `prompt_wav_path` 或 `ref_audio`: TTS 参考音频

**duplex_prefill 关键参数**：
- `audio_waveform`: 音频块（16kHz numpy array，通常 1 秒）
- `frame_list`: 图像帧列表（视频双工）

**duplex_generate 关键参数**：
- `decode_mode`: "greedy" 或 "sample"
- `temperature`, `top_k`, `top_p`: 采样参数
- `listen_prob_scale`: listen 概率缩放

**duplex_generate 返回值**：
```python
{
    "is_listen": bool,  # True=在听, False=在说
    "text": str,  # 生成的文本（is_listen=False 时有效）
    "audio": np.ndarray,  # 生成的音频（可选）
    "end_of_turn": bool,  # 是否轮次结束
}
```

---

### 5. duplex_set_break() / duplex_clear_break() - 打断控制

用于实现用户打断功能。

```python
# 检测到用户打断时
model.duplex_set_break()
# 之后的 duplex_generate() 会返回 is_listen=True

# 清除打断状态
model.duplex_clear_break()
```

---

### 6. duplex_stop() - 停止会话

```python
# 结束双工会话
model.duplex_stop()
```

---

## API 总览表

### 单工模式

| 方法 | 说明 | 输入 | 输出 |
|------|------|------|------|
| `chat()` | 离线对话 | 消息列表 | 文本 (+音频) |
| `streaming_prefill()` | 流式预填充 | 消息列表 | 无 |
| `streaming_generate()` | 流式生成 | 无 | Generator[chunk] |

### 双工模式

| 方法 | 说明 | 输入 | 输出 |
|------|------|------|------|
| `duplex_chat()` | 离线双工 | 完整音频 | dict |
| `duplex_prepare()` | 准备会话 | system_prompt, ref_audio | 无 |
| `duplex_prefill()` | 预填充 | 音频块 (+图像) | 无 |
| `duplex_generate()` | 生成 | 采样参数 | dict |
| `duplex_set_break()` | 设置打断 | 无 | 无 |
| `duplex_clear_break()` | 清除打断 | 无 | 无 |
| `duplex_stop()` | 停止会话 | 无 | 无 |

## 与原始实现的关系

| 原始类 | 统一后 |
|--------|--------|
| `MiniCPMOForCausalLM` | `MiniCPMO`（继承） |
| `MiniCPMODuplex` | `DuplexCapability`（组合） |

## 初始化流程

```python
model = MiniCPMO.from_pretrained(model_path)

# 初始化统一模式（必须调用）
model.init_unified(
    device="cuda",
    ref_audio_path="/path/to/ref.wav",  # 默认参考音频
    preload_both_tts=True,  # 预加载两个 TTS
)
```

## 模式切换

模型支持毫秒级模式切换，无需重新加载参数：

```python
# 单工 -> 双工：直接调用双工方法即可
model.chat(...)  # 单工
model.duplex_prepare(...)  # 自动切换到双工

# 双工 -> 单工：直接调用单工方法即可
model.duplex_generate(...)  # 双工
model.chat(...)  # 自动切换到单工
```

## KV Cache 与会话管理

### KV Cache 清理时机

| 操作 | 清理 KV Cache | 说明 |
|------|---------------|------|
| `chat()` | ✅ 每次清理 | 每次调用都是独立对话 |
| `streaming_prefill(session_id=新ID)` | ✅ 清理 | 新 session 开始新对话 |
| `streaming_prefill(session_id=相同ID)` | ❌ 不清理 | 继续多轮对话 |
| `duplex_prepare()` | ✅ 清理 | 每次 prepare 都是新会话 |
| `set_mode()` / `set_xxx_mode()` | ✅ 清理 | 切换模式时重置 |

---

### Streaming 多轮对话（保留历史）

```python
streaming = processor.set_streaming_mode()

# ========== 第一轮对话 ==========
streaming.prefill(StreamingRequest(
    session_id="user_001",  # 新 session → 清理 cache，开始新对话
    messages=[Message(role=Role.USER, content="你好")],
    is_last_chunk=True,
))
for chunk in streaming.generate(session_id="user_001"):
    print(chunk.text, end="")

# ========== 第二轮对话（相同 session，保留历史）==========
streaming.prefill(StreamingRequest(
    session_id="user_001",  # 相同 session → 保留 cache，继续对话
    messages=[Message(role=Role.USER, content="继续讲")],
    is_last_chunk=True,
))
for chunk in streaming.generate(session_id="user_001"):
    print(chunk.text, end="")

# ========== 新用户对话（新 session）==========
streaming.prefill(StreamingRequest(
    session_id="user_002",  # 新 session → 清理 cache，新对话
    messages=[Message(role=Role.USER, content="你好")],
    is_last_chunk=True,
))
for chunk in streaming.generate(session_id="user_002"):
    print(chunk.text, end="")
```

**关键**：`session_id` 相同时保留历史，不同时重新开始。

---

### Streaming 手动重置会话

```python
# 强制重置某个会话（清理 KV cache）
streaming.reset_session(session_id="user_001")

# 之后再 prefill 会重新开始
streaming.prefill(StreamingRequest(
    session_id="user_001",  # 即使相同 ID，也是新对话
    messages=[...],
))
```

---

### Duplex 会话管理

```python
duplex = processor.set_duplex_mode()

# ========== 第一次会话 ==========
duplex.prepare(...)  # 清理 cache，开始新会话
for audio_chunk in audio_stream:
    duplex.prefill(audio_waveform=audio_chunk)
    result = duplex.generate()
    if result.end_of_turn:
        break
duplex.stop()

# ========== 第二次会话 ==========
duplex.prepare(...)  # 再次清理 cache，又是新会话
# ...
```

**关键**：每次 `prepare()` 都会清理 KV cache 开始新会话。

---

### 模式切换时的清理

```python
# Chat 模式
chat = processor.set_chat_mode()
response = chat.chat(request)

# 切换到 Streaming 模式 → 自动清理所有 cache
streaming = processor.set_streaming_mode()  # < 1ms，同时清理 cache
streaming.prefill(...)

# 切换到 Duplex 模式 → 自动清理所有 cache
duplex = processor.set_duplex_mode()  # < 1ms，同时清理 cache
duplex.prepare(...)
```

**关键**：`set_xxx_mode()` 会调用 `model.set_mode()`，自动清理所有 KV cache。

---

## 与 Processor 层的配合

推荐使用 `UnifiedProcessor` 作为高层接口：

```python
from core.processors import UnifiedProcessor
from core.schemas import ChatRequest, StreamingRequest, Message, Role

processor = UnifiedProcessor(model_path=..., preload_both_tts=True)

# ========== Chat 模式 ==========
chat = processor.set_chat_mode()
response = chat.chat(ChatRequest(
    messages=[Message(role=Role.USER, content="你好")]
))
print(response.content)

# ========== Streaming 模式 ==========
streaming = processor.set_streaming_mode()
streaming.prefill(StreamingRequest(
    session_id="user_001",
    messages=[Message(role=Role.USER, content="讲个故事")],
    is_last_chunk=True,
))
for chunk in streaming.generate(session_id="user_001"):
    print(chunk.text, end="", flush=True)

# ========== Duplex 模式 ==========
duplex = processor.set_duplex_mode()
duplex.prepare(
    system_prompt_text="你是一个友好的助手。",
    ref_audio_path="/path/to/ref.wav",
)
for audio_chunk in audio_stream:
    duplex.prefill(audio_waveform=audio_chunk)
    result = duplex.generate()
    if not result.is_listen:
        print(result.text)
duplex.stop()

# ========== 离线双工 ==========
from core.schemas.duplex import DuplexOfflineInput
output = duplex.offline_inference(DuplexOfflineInput(
    system_prompt="你是助手",
    user_audio_path="/path/to/audio.wav",
    ref_audio_path="/path/to/ref.wav",
))
print(output.full_text)
```

## 文件结构

```
MiniCPMO45/
├── modeling_minicpmo.py          # 原始实现（保留兼容）
├── modeling_minicpmo_unified.py  # 统一实现（推荐使用）
├── utils.py                      # 工具类（StreamDecoder 等）
└── ...
```

## 迁移指南

### 从 MiniCPMOForCausalLM 迁移

```python
# 旧代码
from MiniCPMO45.modeling_minicpmo import MiniCPMOForCausalLM
model = MiniCPMOForCausalLM.from_pretrained(...)

# 新代码
from MiniCPMO45.modeling_minicpmo_unified import MiniCPMO
model = MiniCPMO.from_pretrained(...)
model.init_unified(device="cuda")
```

### 从 MiniCPMODuplex 迁移

```python
# 旧代码
from MiniCPMO45.modeling_minicpmo import MiniCPMODuplex
duplex = MiniCPMODuplex(model_path=...)

# 新代码
from MiniCPMO45.modeling_minicpmo_unified import MiniCPMO
model = MiniCPMO.from_pretrained(model_path)
model.init_unified(device="cuda")
# 直接使用 model.duplex_* 方法
```

## 注意事项

1. **必须调用 init_unified()**：加载模型后必须调用此方法初始化双工能力
2. **ref_audio 用于 TTS**：如需生成语音，需要提供参考音频
3. **preload_both_tts=True 推荐**：预加载两个 TTS 可减少首次生成延迟
