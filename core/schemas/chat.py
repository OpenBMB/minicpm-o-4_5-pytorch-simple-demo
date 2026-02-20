"""单工对话（Chat）Schema 定义 —— 无状态模式

本模块定义单工对话模式的请求和响应格式。

单工对话特点：
============

1. **一问一答**：用户发送消息，等待模型完整回复
2. **同步等待**：调用者阻塞直到收到完整响应
3. **支持多轮**：通过传完整消息历史实现（⚠️ 每次重新 prefill）
4. **无实时交互**：不支持打断、不支持流式返回

**[CRITICAL] 无状态 & 不复用 KV Cache**：
=======================================

- 每次调用都是独立的，**不复用前轮的 KV Cache**
- 多轮对话需要传入**完整历史** `messages`，会重新计算所有消息
- 多轮效率随轮数线性下降（Turn N 需要 prefill N*2-1 条消息）

**如果需要 KV Cache 复用，请使用 StreamingProcessor**。

适用场景：
=========

- 单轮简单问答（推荐）
- 图像理解
- 音频理解（非实时）
- 多轮对话（可用，但效率不如 Streaming）

对比其他模式：
=============

| 特性 | 单工 (Chat) | 流式 (Streaming) | 双工 (Duplex) |
|------|-------------|------------------|---------------|
| 返回方式 | 一次性完整 | 逐块返回 | 实时双向 |
| 是否阻塞 | 是 | 否（迭代器） | 否 |
| KV Cache 复用 | ❌ | ✅ | ✅ |
| 打断支持 | ❌ | ❌ | ✅ |
| 回溯支持 | ✅ | ✅ | ❌ |
| 资源占用 | 短暂 | 持续 | 独占 |

使用示例：
=========

```python
from core.schemas.chat import ChatRequest, ChatResponse
from core.schemas.common import Message, Role, TTSConfig, TTSMode

# 1. 纯文本对话
request = ChatRequest(
    messages=[Message(role=Role.USER, content="1+1等于几？")]
)

# 2. 图像理解
from core.schemas.common import ImageContent, TextContent
request = ChatRequest(
    messages=[
        Message(role=Role.USER, content=[
            ImageContent(data="<base64 encoded image>"),
            TextContent(text="描述这张图片")
        ])
    ]
)

# 3. 带 TTS 输出
request = ChatRequest(
    messages=[Message(role=Role.USER, content="请用语音回复我")],
    tts=TTSConfig(
        enabled=True,
        mode=TTSMode.AUDIO_ASSISTANT,
        ref_audio_path="/path/to/ref.wav",
        output_path="/path/to/output.wav"
    )
)

# 4. 多轮对话
messages = [
    Message(role=Role.USER, content="我叫小明"),
    Message(role=Role.ASSISTANT, content="你好小明！"),
    Message(role=Role.USER, content="我叫什么？"),
]
request = ChatRequest(messages=messages)

# 5. 批量推理
from core.schemas.chat import BatchChatRequest
batch = BatchChatRequest(requests=[request1, request2, request3])
```
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from core.schemas.common import (
    Message,
    TTSConfig,
    GenerationConfig,
    ImageConfig,
    TTSSamplingParams,
)


# =============================================================================
# Chat 请求
# =============================================================================

class ChatRequest(BaseModel):
    """单工对话请求
    
    用于发起一次完整的对话请求，等待模型返回完整响应。
    
    **核心参数**：
    
    - messages: 对话消息列表（必填）
    - generation: 生成参数（控制输出长度、采样策略）
    - tts: TTS 配置（是否输出语音）
    
    **高级参数**：
    
    - image: 图像处理配置
    - use_tts_template: 是否使用 TTS 模板（有音频输入时自动启用）
    - omni_mode: Omni 模式（视频理解）
    - enable_thinking: 思考模式（显示推理过程）
    - return_prompt: 是否返回完整 prompt（调试用）
    
    **omni_mode 说明**：
    
    Omni 模式用于处理视频输入（图像帧序列 + 音频），启用时：
    - 图像和音频会以特殊方式拼接
    - 适合视频理解、实时视频对话
    - 需要配合 TTS 使用
    
    **enable_thinking 说明**：
    
    思考模式会让模型显示推理过程（类似 Chain-of-Thought）：
    - 输出格式：<think>推理过程</think>最终答案
    - 适合需要解释的场景
    - 会增加输出长度
    
    Attributes:
        messages: 对话消息列表
        generation: 生成参数
        tts: TTS 配置
        image: 图像处理配置
        use_tts_template: 是否使用 TTS 模板
        omni_mode: 是否启用 Omni 模式
        enable_thinking: 是否启用思考模式
        return_prompt: 是否返回 prompt（调试用）
    
    示例：
        >>> # 最简单的请求
        >>> request = ChatRequest(
        ...     messages=[Message(role=Role.USER, content="你好")]
        ... )
        
        >>> # 带配置的请求
        >>> request = ChatRequest(
        ...     messages=[Message(role=Role.USER, content="写一首诗")],
        ...     generation=GenerationConfig(
        ...         max_new_tokens=200,
        ...         temperature=0.9
        ...     )
        ... )
    """
    # 核心参数
    messages: List[Message] = Field(
        ..., 
        min_length=1, 
        description="对话消息列表（至少包含一条用户消息）"
    )
    generation: GenerationConfig = Field(
        default_factory=GenerationConfig, 
        description="生成参数配置"
    )
    tts: TTSConfig = Field(
        default_factory=TTSConfig, 
        description="TTS 配置"
    )
    
    # 图像配置
    image: ImageConfig = Field(
        default_factory=ImageConfig,
        description="图像处理配置"
    )
    
    # 高级选项
    use_tts_template: bool = Field(
        False, 
        description="是否使用 TTS 模板（有音频输入时自动启用）"
    )
    omni_mode: bool = Field(
        False, 
        description="是否启用 Omni 模式（视频输入）"
    )
    enable_thinking: bool = Field(
        False, 
        description="是否启用思考模式（显示推理过程）"
    )
    return_prompt: bool = Field(
        False,
        description="是否返回完整 prompt（调试用）"
    )


# =============================================================================
# Chat 响应
# =============================================================================

class ChatResponse(BaseModel):
    """单工对话响应
    
    模型对 ChatRequest 的响应，包含生成的文本和可选的音频。
    
    **核心字段**：
    
    - text: 生成的文本内容（总是存在，即使失败也会有空字符串）
    - success: 是否成功
    - error: 错误信息（仅失败时有值）
    
    **音频字段**（仅当 TTS 启用时）：
    
    - audio_path: 生成的音频文件路径（如果指定了 output_path）
    - audio_data: 生成的音频 Base64 数据
    - audio_sample_rate: 音频采样率（固定 24000 Hz）
    
    **元信息**：
    
    - tokens_generated: 生成的 token 数量
    - duration_ms: 推理耗时（毫秒）
    - prompt: 完整的 prompt（仅当 return_prompt=True）
    
    **错误处理**：
    
    即使推理失败，也会返回 ChatResponse 对象：
    - success=False
    - error 包含错误信息
    - text 为空字符串
    
    Attributes:
        text: 生成的文本内容
        audio_path: 音频文件路径
        audio_data: 音频 Base64 数据
        audio_sample_rate: 音频采样率
        tokens_generated: 生成的 token 数量
        duration_ms: 推理耗时
        prompt: 完整 prompt（调试用）
        error: 错误信息
        success: 是否成功
    
    示例：
        >>> response = processor.chat(request)
        >>> if response.success:
        ...     print(response.text)
        ...     if response.audio_path:
        ...         play_audio(response.audio_path)
        ... else:
        ...     print(f"Error: {response.error}")
    """
    # 核心输出
    text: str = Field(..., description="生成的文本内容")
    
    # 音频输出（TTS）
    audio_path: Optional[str] = Field(
        None, 
        description="生成的音频文件路径"
    )
    audio_data: Optional[str] = Field(
        None, 
        description="生成的音频 Base64 数据（24kHz, float32）"
    )
    audio_sample_rate: int = Field(
        24000,
        description="音频采样率（固定 24000 Hz）"
    )
    
    # 元信息
    tokens_generated: Optional[int] = Field(
        None, 
        description="生成的 token 数量（旧字段，建议使用 token_stats）"
    )
    duration_ms: Optional[float] = Field(
        None, 
        description="推理耗时（毫秒）"
    )
    prompt: Optional[str] = Field(
        None,
        description="完整的 prompt（仅当 return_prompt=True）"
    )
    token_stats: Optional[Dict[str, int]] = Field(
        None,
        description=(
            "LLM 主干 token 统计。"
            "包含 cached_tokens（缓存命中）、input_tokens（总输入）、"
            "generated_tokens（生成）、total_tokens（最终 KV cache 长度）。"
            "Chat 模式下 cached_tokens 始终为 0。"
        ),
    )
    
    # 录制
    recording_session_id: Optional[str] = Field(
        None,
        description="后端录制的 session ID（用于 /s/{id} 回看页面）",
    )

    # 状态
    error: Optional[str] = Field(
        None, 
        description="错误信息（仅失败时有值）"
    )
    success: bool = Field(
        True, 
        description="是否成功"
    )
