"""处理器能力声明

本模块定义每种处理器的能力，使调用者无需阅读实现代码即可了解：
- 支持哪些输入类型（文本/图像/音频）
- 支持哪些输出类型（文本/音频）
- 支持哪些交互特性（打断/回溯）
- 资源占用特性

设计理念：
=========

能力声明是一种"契约"，让调用者（Gateway、前端）知道：
1. 某个 Processor 能做什么
2. 不能做什么
3. 需要什么资源

这样 Gateway 可以根据能力进行路由和调度：
- 双工请求 → 路由到 Duplex Worker（独占）
- 普通请求 → 路由到 Chat Worker（可共享）

使用示例：
=========

```python
from core.capabilities import ProcessorMode, CAPABILITIES

# 查询某模式的能力
cap = CAPABILITIES[ProcessorMode.DUPLEX]
print(f"双工模式支持打断: {cap.supports_interrupt}")  # True
print(f"双工模式需要独占: {cap.requires_exclusive_worker}")  # True

# 在 Gateway 中使用
def route_request(request):
    mode = determine_mode(request)
    cap = CAPABILITIES[mode]
    
    if cap.requires_exclusive_worker:
        return allocate_exclusive_worker()
    else:
        return allocate_shared_worker()
```

三种模式的能力对比：
==================

| 能力 | Chat | Streaming | Duplex |
|------|------|-----------|--------|
| 文本输入 | ✅ | ✅ | ✅ |
| 图像输入 | ✅ | ✅ | ✅ |
| 音频输入 | ✅ | ✅ | ✅ |
| 文本输出 | ✅ | ✅ | ✅ |
| 音频输出 (TTS) | ✅ | ✅ | ✅ |
| 流式输出 | ❌ | ✅ | ✅ |
| 多轮对话 | ✅ | ✅ | ✅ |
| 打断支持 | ❌ | ❌ | ✅ |
| 回溯支持 | ✅ | ✅ | ❌ |
| KV Cache 复用 | ✅ | ✅ | ❌ |
| 独占 Worker | ❌ | ❌ | ✅ |
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict


class ProcessorMode(Enum):
    """处理器模式
    
    MiniCPMO45 支持三种处理模式，每种模式有不同的特性。
    
    - CHAT: 单工对话
      - 一问一答，同步等待完整响应
      - 适合简单问答、批量处理
      - 使用 MiniCPMO 类
      
    - STREAMING: 流式对话
      - 一问一答，流式返回响应
      - 适合实时 TTS、快速首字显示
      - 使用 MiniCPMO 类（streaming=True）
      
    - DUPLEX: 双工对话
      - 全双工，用户和模型可同时说话
      - 适合语音助手、实时交互
      - 使用 MiniCPMODuplex 类（不同的类！）
    
    **[CRITICAL] CHAT/STREAMING vs DUPLEX**：
    
    - CHAT 和 STREAMING 使用同一个模型实例（MiniCPMO）
    - DUPLEX 使用不同的模型类（MiniCPMODuplex）
    - 切换 CHAT/STREAMING：只需 init_tts(streaming=True/False)
    - 切换到 DUPLEX：需要重新加载模型
    
    这意味着 Worker 可以分为两类：
    - MiniCPMO Worker：处理 CHAT 和 STREAMING
    - MiniCPMODuplex Worker：处理 DUPLEX
    """
    CHAT = auto()
    STREAMING = auto()
    DUPLEX = auto()


@dataclass(frozen=True)
class ProcessorCapabilities:
    """处理器能力声明
    
    定义一个处理器支持的所有能力和特性。
    
    **输入能力**：
    
    - supports_text: 支持文本输入
    - supports_image: 支持图像输入
    - supports_audio: 支持音频输入
    - supports_video: 支持视频输入（图像帧序列）
    
    **输出能力**：
    
    - supports_text_output: 支持文本输出
    - supports_audio_output: 支持音频输出（TTS）
    - supports_streaming_output: 支持流式输出
    
    **交互能力**：
    
    - supports_multi_turn: 支持多轮对话
    - supports_interrupt: 支持打断（用户打断模型）
    - supports_rollback: 支持回溯（恢复到之前的状态）
    
    **资源特性**：
    
    - requires_exclusive_worker: 是否需要独占 Worker
    - supports_kv_cache_reuse: 是否支持 KV Cache 复用
    
    Attributes:
        mode: 处理器模式
        supports_text: 支持文本输入
        supports_image: 支持图像输入
        supports_audio: 支持音频输入
        supports_video: 支持视频输入
        supports_text_output: 支持文本输出
        supports_audio_output: 支持音频输出
        supports_streaming_output: 支持流式输出
        supports_multi_turn: 支持多轮对话
        supports_interrupt: 支持打断
        supports_rollback: 支持回溯
        requires_exclusive_worker: 需要独占 Worker
        supports_kv_cache_reuse: 支持 KV Cache 复用
    """
    mode: ProcessorMode
    
    # 输入能力
    supports_text: bool = True
    supports_image: bool = True
    supports_audio: bool = True
    supports_video: bool = False  # 图像帧序列
    
    # 输出能力
    supports_text_output: bool = True
    supports_audio_output: bool = True  # TTS
    supports_streaming_output: bool = False
    
    # 交互能力
    supports_multi_turn: bool = True
    supports_interrupt: bool = False      # 打断
    supports_rollback: bool = False       # 回溯
    
    # 资源特性
    requires_exclusive_worker: bool = False
    supports_kv_cache_reuse: bool = True
    
    def __str__(self) -> str:
        """返回能力摘要"""
        features = []
        if self.supports_streaming_output:
            features.append("streaming")
        if self.supports_interrupt:
            features.append("interrupt")
        if self.supports_rollback:
            features.append("rollback")
        if self.requires_exclusive_worker:
            features.append("exclusive")
        
        return f"{self.mode.name}: [{', '.join(features) or 'basic'}]"


# =============================================================================
# 预定义能力
# =============================================================================

CHAT_CAPABILITIES = ProcessorCapabilities(
    mode=ProcessorMode.CHAT,
    
    # 输入：全支持
    supports_text=True,
    supports_image=True,
    supports_audio=True,
    supports_video=False,  # Chat 不支持视频（图像帧序列）
    
    # 输出：文本+音频，非流式
    supports_text_output=True,
    supports_audio_output=True,
    supports_streaming_output=False,  # 非流式
    
    # 交互：多轮+回溯
    supports_multi_turn=True,
    supports_interrupt=False,  # 不支持打断
    supports_rollback=True,    # 支持回溯（通过 speculative_snapshot）
    
    # 资源：共享、KV Cache 复用
    requires_exclusive_worker=False,
    supports_kv_cache_reuse=True,
)
"""单工对话能力

特点：
- 一问一答，完整返回
- 不支持打断
- 支持回溯（speculative_snapshot）
- 可以共享 Worker
- 支持 KV Cache 复用（多轮对话）
"""


STREAMING_CAPABILITIES = ProcessorCapabilities(
    mode=ProcessorMode.STREAMING,
    
    # 输入：全支持
    supports_text=True,
    supports_image=True,
    supports_audio=True,
    supports_video=False,
    
    # 输出：文本+音频，流式
    supports_text_output=True,
    supports_audio_output=True,
    supports_streaming_output=True,  # 流式！
    
    # 交互：多轮+回溯
    supports_multi_turn=True,
    supports_interrupt=False,  # 不支持打断
    supports_rollback=True,    # 支持回溯
    
    # 资源：共享、KV Cache 复用
    requires_exclusive_worker=False,
    supports_kv_cache_reuse=True,
)
"""流式对话能力

特点：
- 一问一答，流式返回（边生成边返回）
- 首字延迟低
- 不支持打断
- 支持回溯（speculative_snapshot）
- 可以共享 Worker（但连接期间占用）
- 支持 KV Cache 复用
"""


DUPLEX_CAPABILITIES = ProcessorCapabilities(
    mode=ProcessorMode.DUPLEX,
    
    # 输入：全支持，包括视频
    supports_text=True,
    supports_image=True,
    supports_audio=True,
    supports_video=True,  # 支持视频（图像帧序列）！
    
    # 输出：文本+音频，流式
    supports_text_output=True,
    supports_audio_output=True,
    supports_streaming_output=True,
    
    # 交互：多轮+打断
    supports_multi_turn=True,
    supports_interrupt=True,   # 支持打断！
    supports_rollback=False,   # 不支持回溯
    
    # 资源：独占！
    requires_exclusive_worker=True,  # 独占！
    supports_kv_cache_reuse=False,   # 不复用
)
"""双工对话能力

特点：
- 全双工，用户和模型可同时说话
- 支持打断（用户打断模型）
- 支持视频输入（图像帧序列）
- 需要独占 Worker（实时性要求）
- 不支持 KV Cache 复用（状态复杂）
"""


# 能力字典（按模式索引）
CAPABILITIES: Dict[ProcessorMode, ProcessorCapabilities] = {
    ProcessorMode.CHAT: CHAT_CAPABILITIES,
    ProcessorMode.STREAMING: STREAMING_CAPABILITIES,
    ProcessorMode.DUPLEX: DUPLEX_CAPABILITIES,
}
"""按模式索引的能力字典

使用示例：
    >>> from core.capabilities import CAPABILITIES, ProcessorMode
    >>> cap = CAPABILITIES[ProcessorMode.DUPLEX]
    >>> cap.supports_interrupt
    True
"""


def get_capabilities(mode: ProcessorMode) -> ProcessorCapabilities:
    """获取指定模式的能力声明
    
    Args:
        mode: 处理器模式
        
    Returns:
        对应模式的能力声明
        
    Raises:
        KeyError: 未知的模式
        
    示例：
        >>> cap = get_capabilities(ProcessorMode.CHAT)
        >>> print(cap)
        CHAT: [rollback]
    """
    return CAPABILITIES[mode]


def supports_feature(mode: ProcessorMode, feature: str) -> bool:
    """检查指定模式是否支持某特性
    
    Args:
        mode: 处理器模式
        feature: 特性名称（如 "interrupt", "rollback", "streaming_output"）
        
    Returns:
        是否支持该特性
        
    Raises:
        AttributeError: 特性名称不存在
        
    示例：
        >>> supports_feature(ProcessorMode.DUPLEX, "interrupt")
        True
        >>> supports_feature(ProcessorMode.CHAT, "interrupt")
        False
    """
    cap = CAPABILITIES[mode]
    attr_name = f"supports_{feature}"
    if hasattr(cap, attr_name):
        return getattr(cap, attr_name)
    raise AttributeError(f"Unknown feature: {feature}")
