"""处理器工厂

本模块提供统一的处理器创建入口，简化处理器的实例化过程。

设计理念：
=========

工厂模式的优势：
1. **统一入口**：不需要知道具体的处理器类
2. **参数校验**：根据模式自动校验必需参数
3. **配置管理**：支持从配置文件创建处理器
4. **便于扩展**：新增处理器类型时，只需修改工厂

使用示例：
=========

**推荐：直接使用 UnifiedProcessor**

```python
from core.processors import UnifiedProcessor

processor = UnifiedProcessor(
    model_path="/path/to/base_model",
    pt_path="/path/to/custom.pt",  # 可选：覆盖权重
)

# 模式切换（毫秒级）
chat = processor.set_chat_mode()
half_duplex = processor.set_half_duplex_mode()
duplex = processor.set_duplex_mode()
```

**使用工厂创建特定模式的 View**

```python
from core.factory import ProcessorFactory
from core.capabilities import ProcessorMode

# 创建 UnifiedProcessor 并返回指定模式的 View
chat_view = ProcessorFactory.create(
    mode=ProcessorMode.CHAT,
    model_path="/path/to/base_model",
    pt_path="/path/to/custom.pt",  # 可选
)
```

从配置创建：
===========

```python
config = {
    "mode": "DUPLEX",
    "model_path": "/path/to/base_model",
    "pt_path": "/path/to/custom.pt",  # 可选
    "ref_audio_path": "/path/to/ref.wav",
}
duplex_view = ProcessorFactory.from_config(config)
```
"""

from typing import Union, Dict, Any, Optional

from core.capabilities import ProcessorMode
from core.processors import UnifiedProcessor, ChatView, HalfDuplexView, DuplexView


# 缓存 UnifiedProcessor 实例，避免重复加载模型
_processor_cache: Dict[str, UnifiedProcessor] = {}


class ProcessorFactory:
    """处理器工厂
    
    统一创建处理器的入口，根据模式返回对应的 View 实例。
    
    **支持的模式**：
    
    - CHAT: ChatView
    - HALF_DUPLEX: HalfDuplexView
    - DUPLEX: DuplexView
    
    **参数校验**：
    
    - TTS 相关功能需要 ref_audio_path
    
    类方法：
        create(): 创建 View
        from_config(): 从配置字典创建
        get_processor(): 获取缓存的 UnifiedProcessor
    """
    
    @staticmethod
    def get_processor(
        model_path: str,
        pt_path: Optional[str] = None,
        device: str = "cuda",
        ref_audio_path: Optional[str] = None,
        **kwargs
    ) -> UnifiedProcessor:
        """获取或创建 UnifiedProcessor（带缓存）
        
        如果相同 model_path + pt_path 的 processor 已存在，直接复用。
        
        Args:
            model_path: 基础模型路径（HuggingFace 格式目录）
            pt_path: 额外的 .pt 权重路径（可选，用于覆盖基础模型权重）
            device: 运行设备
            ref_audio_path: 参考音频路径
            **kwargs: 其他参数
            
        Returns:
            UnifiedProcessor 实例
        """
        cache_key = f"{model_path}:{pt_path or ''}"
        if cache_key not in _processor_cache:
            _processor_cache[cache_key] = UnifiedProcessor(
                model_path=model_path,
                pt_path=pt_path,
                device=device,
                ref_audio_path=ref_audio_path,
                **kwargs
            )
        return _processor_cache[cache_key]
    
    @staticmethod
    def create(
        mode: ProcessorMode,
        model_path: str,
        pt_path: Optional[str] = None,
        device: str = "cuda",
        ref_audio_path: Optional[str] = None,
        **kwargs
    ) -> Union[ChatView, HalfDuplexView, DuplexView]:
        """创建 View
        
        根据指定的模式创建对应的 View 实例。
        
        Args:
            mode: 处理器模式（CHAT/HALF_DUPLEX/DUPLEX）
            model_path: 基础模型路径（HuggingFace 格式目录）
            pt_path: 额外的 .pt 权重路径（可选，用于覆盖基础模型权重）
            device: 运行设备，默认 "cuda"
            ref_audio_path: 参考音频路径（TTS 用）
            **kwargs: 其他参数传递给处理器构造函数
            
        Returns:
            对应模式的 View 实例
            
        Raises:
            ValueError: 未知的模式
        
        示例：
            >>> # Chat
            >>> chat = ProcessorFactory.create(
            ...     ProcessorMode.CHAT,
            ...     model_path="/path/to/model",
            ...     pt_path="/path/to/weights.pt"
            ... )
            
            >>> # Duplex
            >>> duplex = ProcessorFactory.create(
            ...     ProcessorMode.DUPLEX,
            ...     model_path="/path/to/model",
            ...     pt_path="/path/to/weights.pt"
            ... )
        """
        processor = ProcessorFactory.get_processor(
            model_path=model_path,
            pt_path=pt_path,
            device=device,
            ref_audio_path=ref_audio_path,
            **kwargs
        )
        
        if mode == ProcessorMode.CHAT:
            return processor.set_chat_mode()
        elif mode == ProcessorMode.HALF_DUPLEX:
            return processor.set_half_duplex_mode()
        elif mode == ProcessorMode.DUPLEX:
            return processor.set_duplex_mode()
        else:
            raise ValueError(f"未知的处理器模式: {mode}")
    
    @staticmethod
    def from_config(config: Dict[str, Any]) -> Union[ChatView, HalfDuplexView, DuplexView]:
        """从配置字典创建 View
        
        便于从配置文件（YAML/JSON）加载处理器。
        
        Args:
            config: 配置字典，必须包含 "mode" 和 "model_path"
            
        Returns:
            View 实例
            
        Raises:
            KeyError: 缺少必需的配置项
            ValueError: mode 值无效
        
        配置格式：
            ```python
            config = {
                "mode": "CHAT" | "HALF_DUPLEX" | "DUPLEX",
                "model_path": "/path/to/model",
                "pt_path": "/path/to/weights.pt",  # 可选，覆盖权重
                "device": "cuda",  # 可选，默认 cuda
                "ref_audio_path": "/path/to/ref.wav",  # 可选
                # ... 其他参数
            }
            ```
        
        示例：
            >>> import yaml
            >>> with open("config.yaml") as f:
            ...     config = yaml.safe_load(f)
            >>> view = ProcessorFactory.from_config(config)
        """
        # 复制 config 避免修改原始参数
        config = config.copy()
        
        # 提取必需参数
        mode_str = config.pop("mode")
        model_path = config.pop("model_path")
        
        # 解析模式
        try:
            mode = ProcessorMode[mode_str.upper()]
        except KeyError:
            raise ValueError(
                f"无效的模式: {mode_str}，"
                f"有效值: {[m.name for m in ProcessorMode]}"
            )
        
        # 提取可选参数
        pt_path = config.pop("pt_path", None)
        device = config.pop("device", "cuda")
        ref_audio_path = config.pop("ref_audio_path", None)
        
        # 创建 View
        return ProcessorFactory.create(
            mode=mode,
            model_path=model_path,
            pt_path=pt_path,
            device=device,
            ref_audio_path=ref_audio_path,
            **config  # 剩余参数传递给处理器
        )


# 便捷函数
def create_processor(
    mode: ProcessorMode,
    model_path: str,
    **kwargs
) -> Union[ChatView, HalfDuplexView, DuplexView]:
    """创建 View（便捷函数）
    
    ProcessorFactory.create() 的简写形式。
    
    Args:
        mode: 处理器模式
        model_path: 模型路径
        **kwargs: 其他参数
        
    Returns:
        View 实例
    
    示例:
        >>> from core.factory import create_processor
        >>> from core.capabilities import ProcessorMode
        >>> chat = create_processor(ProcessorMode.CHAT, "/path/to/model")
    """
    return ProcessorFactory.create(mode=mode, model_path=model_path, **kwargs)
