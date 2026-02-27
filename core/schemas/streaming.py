"""流式对话（Streaming）Schema 定义 —— 有状态模式

本模块定义流式对话模式的请求和响应格式。

流式对话特点：
============

1. **流式返回**：模型边生成边返回，用户可以实时看到输出（单轮内）
2. **有状态（session_id）**：通过会话 ID 管理状态，**复用 KV Cache**
3. **增量预填充**：多轮对话只需 prefill 新消息，不重复计算历史
4. **分步操作**：prefill（预填充）→ generate（生成）
5. **支持回溯**：可以保存快照并恢复（用于 VAD 抢跑）

**[CRITICAL] 有状态 & KV Cache 复用**：
=====================================

- 通过 `session_id` 标识会话，同一 session 内 **复用 KV Cache**
- 多轮对话只需 prefill **新的一条消息**，历史已在 KV Cache 中
- 相比 Chat 模式，多轮对话效率**大幅提升**

```
Turn 1: prefill(user1)  → KV Cache 保存
Turn 2: prefill(user2)  → 只计算新消息，复用 Turn 1 的 KV
Turn 3: prefill(user3)  → 只计算新消息，复用 Turn 1+2 的 KV
```

**需要多轮高效对话时，推荐使用 Streaming 而非 Chat**。

**session_id 机制与限制**：
========================

模型层内部逻辑：

```python
# 每次 streaming_prefill() 时判断
is_first = (self.session_id is None) or (session_id != self.session_id)

if is_first:
    reset_session()              # 清空 KV Cache
    self.session_id = session_id # 记录新 session
    # 完整 prefill
else:
    # 增量 prefill（复用 KV Cache）
```

**[CRITICAL] 关键限制**：

| 特性 | 说明 |
|------|------|
| 单 session | 模型实例只能同时维护 **一个** session 的 KV Cache |
| 切换丢失 | 切换 session_id 会 **清空并丢失** 旧 session 的 KV Cache |
| 不支持并发 | 多用户需要 **多个模型实例** 或 **会话调度器** |

```python
# ❌ 不支持多 session 交替（A 的 KV 会丢失）
prefill(session_id="A", msg)
prefill(session_id="B", msg)  # reset！A 的 KV 丢失
prefill(session_id="A", msg)  # 需要重新 prefill

# ✅ 正确用法：同一 session 连续调用
prefill(session_id="A", msg1)  # 建立
prefill(session_id="A", msg2)  # 增量
prefill(session_id="A", msg3)  # 继续增量
```

工作流程：
=========

```
┌─────────────────────────────────────────────────────────────┐
│                 流式对话流程（有状态）                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. init_streaming_mode()     初始化流式模式                 │
│           ↓                                                 │
│  2. reset_session(session_id) 重置/创建会话                  │
│           ↓                   （清空 KV Cache）              │
│  3. streaming_prefill()       预填充用户消息                 │
│           ↓                   （增量，复用 KV Cache）        │
│  4. streaming_generate()      流式生成响应                   │
│           ↓                   （返回迭代器，逐块产出）       │
│  5. [可选] rollback()         回溯到上一个快照点              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

对比单工模式：
=============

| 特性 | 单工 (Chat) | 流式 (Streaming) |
|------|-------------|------------------|
| 返回方式 | 一次性完整 | 逐块返回 |
| 状态管理 | ❌ 无状态 | ✅ session_id |
| KV Cache 复用 | ❌ 每次重算 | ✅ 增量 prefill |
| 多轮效率 | 低（O(n²)） | 高（O(n)） |
| 首字延迟 | 高（等完整生成） | 低（立即开始） |
| 内存占用 | 较低 | 较高（维护状态） |
| 适用场景 | 单轮简单问答 | 多轮对话、实时 TTS |

回溯机制（Speculative Snapshot）：
=================================

流式模式支持**回溯**功能，用于 VAD（语音活动检测）场景：

1. **场景**：用户说完后，模型开始生成，但用户又说话了
2. **问题**：已生成的内容需要丢弃，回到用户说话前的状态
3. **方案**：
   - 生成前保存快照（enable_speculative_snapshot=True）
   - 检测到用户继续说话时，调用 rollback() 恢复状态
   - 继续预填充用户的新输入

使用示例：
=========

```python
from core.schemas.streaming import (
    StreamingConfig,
    StreamingRequest,
    StreamingChunk,
    StreamingResponse,
)

# 1. 基本流式对话
request = StreamingRequest(
    session_id="user_001",
    messages=[Message(role=Role.USER, content="讲个故事")],
    is_last_chunk=True
)

# 预填充
processor.streaming_prefill(request)

# 流式生成
for chunk in processor.streaming_generate(session_id="user_001"):
    print(chunk.text_delta, end="", flush=True)
    if chunk.audio_data:
        play_audio_chunk(chunk.audio_data)

# 2. 带回溯的流式对话（VAD 场景）
processor.streaming_prefill(request)
for chunk in processor.streaming_generate(
    session_id="user_001",
    enable_speculative_snapshot=True  # 启用回溯
):
    if vad_detected_user_speaking():
        processor.rollback()  # 回溯到 prefill 后的状态
        break
    process(chunk)
```

音频输出说明：
=============

流式模式的音频输出是 **24kHz float32 PCM** 格式：
- 每个 StreamingChunk 包含一小段音频
- 需要按顺序拼接才能得到完整音频
- 采样率固定 24000 Hz（与单工模式一致）
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from core.schemas.common import (
    Message,
    GenerationConfig,
    ImageConfig,
    TTSSamplingParams,
)


# =============================================================================
# 流式配置
# =============================================================================

class StreamingConfig(BaseModel):
    """流式推理配置
    
    控制流式生成的行为和参数。
    
    **音频生成**：
    
    - generate_audio=True：生成文本+音频
    - generate_audio=False：只生成文本（更快）
    
    **分块大小**：
    
    - audio_token_chunk_size：每次生成的音频 token 数
    - 默认 25 token/s，即每秒生成约 1 秒音频
    - 较小值：延迟更低，但效率较低
    - 较大值：效率更高，但延迟更高
    
    **回溯功能**：
    
    - enable_speculative_snapshot=True：生成前保存状态快照
    - 用于 VAD 场景，检测到用户继续说话时可回溯
    
    Attributes:
        generate_audio: 是否生成音频
        audio_token_chunk_size: 音频 token 块大小
        ref_audio_path: 参考音频路径（TTS 时必须）
        ref_audio_data: 参考音频 Base64 数据
        enable_speculative_snapshot: 是否启用回溯快照
        tts_sampling: TTS 采样参数
    """
    generate_audio: bool = Field(
        True, 
        description="是否生成音频"
    )
    audio_token_chunk_size: int = Field(
        25, 
        ge=1,
        description="音频 token 块大小（默认 25 token/s）"
    )
    ref_audio_path: Optional[str] = Field(
        None, 
        description="参考音频路径（TTS 时必须）"
    )
    ref_audio_data: Optional[str] = Field(
        None, 
        description="参考音频 Base64 数据"
    )
    enable_speculative_snapshot: bool = Field(
        False,
        description="是否启用回溯快照（VAD 场景）"
    )
    tts_sampling: TTSSamplingParams = Field(
        default_factory=TTSSamplingParams,
        description="TTS 采样参数"
    )


# =============================================================================
# 流式请求
# =============================================================================

class StreamingRequest(BaseModel):
    """流式预填充请求
    
    用于向模型上下文中预填充消息。可以多次调用以逐块预填充。
    
    **预填充流程**：
    
    1. 第一次预填充：发送用户消息，is_last_chunk=False
    2. 继续预填充：发送更多内容（可选）
    3. 最后一块：is_last_chunk=True，表示预填充完成
    
    **单次预填充（常见场景）**：
    
    ```python
    request = StreamingRequest(
        session_id="user_001",
        messages=[Message(role=Role.USER, content="你好")],
        is_last_chunk=True  # 只有一块
    )
    processor.streaming_prefill(request)
    ```
    
    **分块预填充（长音频场景）**：
    
    ```python
    # 第 1 块
    processor.streaming_prefill(StreamingRequest(
        session_id="user_001",
        messages=[Message(role=Role.USER, content=[AudioContent(data=chunk1)])],
        is_last_chunk=False
    ))
    
    # 第 2 块（最后一块）
    processor.streaming_prefill(StreamingRequest(
        session_id="user_001", 
        messages=[Message(role=Role.USER, content=[AudioContent(data=chunk2)])],
        is_last_chunk=True
    ))
    ```
    
    Attributes:
        session_id: 会话 ID（必须唯一标识一个对话）
        messages: 消息列表（单次预填充通常只有一条）
        is_last_chunk: 是否是最后一块预填充
        generation: 生成参数
        streaming: 流式配置
        omni_mode: Omni 模式
        use_tts_template: 使用 TTS 模板
        enable_thinking: 启用思考模式
    """
    session_id: str = Field(
        ..., 
        description="会话 ID（唯一标识一个对话）"
    )
    messages: List[Message] = Field(
        ..., 
        min_length=1, 
        description="消息列表（单次预填充通常只有一条）"
    )
    is_last_chunk: bool = Field(
        False, 
        description="是否是最后一块预填充"
    )
    
    # 配置
    generation: GenerationConfig = Field(
        default_factory=GenerationConfig, 
        description="生成参数"
    )
    streaming: StreamingConfig = Field(
        default_factory=StreamingConfig, 
        description="流式配置"
    )
    
    # 高级选项
    omni_mode: bool = Field(
        True, 
        description="Omni 模式"
    )
    use_tts_template: bool = Field(
        True, 
        description="使用 TTS 模板"
    )
    enable_thinking: bool = Field(
        False, 
        description="启用思考模式"
    )
    
    # 图像配置
    image: ImageConfig = Field(
        default_factory=ImageConfig,
        description="图像处理配置（max_slice_nums 等）"
    )


# =============================================================================
# 流式响应块
# =============================================================================

class StreamingChunk(BaseModel):
    """流式响应块
    
    streaming_generate() 每次 yield 返回一个 StreamingChunk。
    
    **块内容**：
    
    - text_delta: 本块生成的**增量**文本（需要累加）
    - audio_data: 本块生成的**增量**音频（Base64，需要拼接）
    - is_final: 是否是最后一块
    
    **[CRITICAL] 增量语义**：
    
    - `text_delta` 是**增量**的，必须累加才能得到完整文本
    - `audio_data` 是**增量**的，必须按顺序拼接
    - 命名带 `_delta` 后缀，明确表示是增量而非完整内容
    
    **正确使用示例**：
    
    ```python
    full_text = ""
    audio_chunks = []
    
    for chunk in streaming_view.generate(session_id):
        # [CRITICAL] 累加增量文本
        if chunk.text_delta:
            full_text += chunk.text_delta
            print(chunk.text_delta, end="", flush=True)
        
        # 收集音频块
        if chunk.audio_data:
            audio_chunks.append(chunk.audio_data)
        
        if chunk.is_final:
            print("\\n[Done]")
            break
    
    # full_text 现在是完整的回复文本
    ```
    
    **便捷方法**：如果只需要完整结果，使用 `HalfDuplexView.complete_turn()`。
    
    Attributes:
        chunk_index: 块索引（从 0 开始）
        text_delta: 增量文本（本块新增的文本，需要累加）
        audio_data: 音频块 Base64 数据（24kHz float32）
        audio_sample_rate: 音频采样率（固定 24000）
        is_final: 是否是最后一块
        duration_ms: 本块生成耗时
    """
    chunk_index: int = Field(
        ..., 
        description="块索引（从 0 开始）"
    )
    text_delta: Optional[str] = Field(
        None, 
        description="增量文本（本块新增的文本，需要累加得到完整回复）"
    )
    audio_data: Optional[str] = Field(
        None, 
        description="音频块 Base64 数据（24kHz float32）"
    )
    audio_sample_rate: int = Field(
        24000, 
        description="音频采样率（固定 24000 Hz）"
    )
    is_final: bool = Field(
        False, 
        description="是否是最后一块"
    )
    
    # 元信息
    duration_ms: Optional[float] = Field(
        None, 
        description="本块生成耗时（毫秒）"
    )


# =============================================================================
# 流式完成响应
# =============================================================================

class StreamingResponse(BaseModel):
    """流式推理完成后的汇总响应
    
    当 streaming_generate() 迭代完成后，或 complete_turn() 调用完成后返回。
    
    **内容**：
    
    - full_text: 完整的生成文本（所有 chunk.text_delta 拼接）
    - audio_path: 合并后的音频路径（如果保存了）
    - audio_data: 合并后的音频 Base64 数据（如果生成了）
    - audio_duration_ms: 音频时长（毫秒）
    - total_chunks: 总块数
    - total_duration_ms: 总耗时
    
    **用途**：
    
    - 获取完整文本（避免手动拼接）
    - 获取合并后的音频（Base64 或文件路径）
    - 统计信息
    - 错误检查
    
    Attributes:
        session_id: 会话 ID
        full_text: 完整生成文本
        audio_path: 合并后的音频路径（如果 output_audio_path 指定了）
        audio_data: 合并后的音频 Base64 数据（24kHz float32）
        audio_sample_rate: 音频采样率（固定 24000）
        audio_duration_ms: 音频时长（毫秒）
        total_chunks: 总块数
        total_duration_ms: 总耗时
        success: 是否成功
        error: 错误信息
    """
    session_id: str = Field(
        ..., 
        description="会话 ID"
    )
    full_text: str = Field(
        ..., 
        description="完整生成文本"
    )
    audio_path: Optional[str] = Field(
        None, 
        description="合并后的音频路径（如果 output_audio_path 指定了）"
    )
    audio_data: Optional[str] = Field(
        None, 
        description="合并后的音频 Base64 数据（24kHz float32）"
    )
    audio_sample_rate: int = Field(
        24000, 
        description="音频采样率（固定 24000 Hz）"
    )
    audio_duration_ms: Optional[float] = Field(
        None, 
        description="音频时长（毫秒）"
    )
    total_chunks: int = Field(
        ..., 
        description="总块数"
    )
    total_duration_ms: float = Field(
        ..., 
        description="总耗时（毫秒）"
    )
    success: bool = Field(
        True, 
        description="是否成功"
    )
    error: Optional[str] = Field(
        None, 
        description="错误信息"
    )


# =============================================================================
# 回溯相关
# =============================================================================

class RollbackResult(BaseModel):
    """回溯操作结果
    
    调用 processor.rollback() 后返回的结果。
    
    **成功条件**：
    
    1. 之前启用了 enable_speculative_snapshot=True
    2. 有可用的快照
    3. 尚未调用过 rollback()（每个快照只能恢复一次）
    
    **失败原因**：
    
    - 未启用快照功能
    - 没有保存快照
    - 快照已被使用
    
    Attributes:
        success: 是否成功
        reason: 失败原因（成功时为 None）
        restored_position: 恢复到的位置信息
    """
    success: bool = Field(
        ...,
        description="是否成功回溯"
    )
    reason: Optional[str] = Field(
        None,
        description="失败原因"
    )
    restored_position: Optional[str] = Field(
        None,
        description="恢复到的位置信息（调试用）"
    )
