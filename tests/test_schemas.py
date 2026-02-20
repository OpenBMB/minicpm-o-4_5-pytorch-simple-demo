"""Pydantic Schema 单元测试

测试 core/schemas.py 中定义的所有 Schema 的验证逻辑。
这些测试不需要 GPU，可以快速运行。

运行命令：
cd /user/sunweiyue/lib/swy-dev/minicpmo45_service
PYTHONPATH=. .venv/base/bin/python -m pytest tests/test_schemas.py -v
"""

import pytest
from pydantic import ValidationError

from core.schemas import (
    # 枚举
    Role, TTSMode, ContentType,
    # 内容类型
    TextContent, ImageContent, AudioContent,
    # 消息
    Message,
    # 配置
    TTSConfig, GenerationConfig,
    # 请求/响应
    ChatRequest, ChatResponse,
)


# =============================================================================
# 测试枚举类型
# =============================================================================

class TestEnums:
    """测试枚举类型"""
    
    def test_role_values(self):
        """Role 枚举值"""
        assert Role.SYSTEM == "system"
        assert Role.USER == "user"
        assert Role.ASSISTANT == "assistant"
    
    def test_tts_mode_values(self):
        """TTSMode 枚举值"""
        assert TTSMode.DEFAULT == "default"
        assert TTSMode.AUDIO_ASSISTANT == "audio_assistant"
        assert TTSMode.OMNI == "omni"
        assert TTSMode.AUDIO_ROLEPLAY == "audio_roleplay"
        assert TTSMode.VOICE_CLONING == "voice_cloning"


# =============================================================================
# 测试内容类型
# =============================================================================

class TestContentTypes:
    """测试多模态内容类型"""
    
    def test_text_content_valid(self):
        """TextContent 正常创建"""
        content = TextContent(text="你好")
        assert content.type == "text"
        assert content.text == "你好"
    
    def test_text_content_empty(self):
        """TextContent 允许空字符串"""
        content = TextContent(text="")
        assert content.text == ""
    
    def test_image_content_with_data(self):
        """ImageContent 使用 base64 数据"""
        content = ImageContent(data="base64encodeddata")
        assert content.type == "image"
        assert content.data == "base64encodeddata"
    
    def test_image_content_requires_data(self):
        """ImageContent 必须提供 data"""
        with pytest.raises(ValidationError):
            ImageContent()
    
    def test_audio_content_with_data(self):
        """AudioContent 使用 base64 数据"""
        content = AudioContent(data="base64pcmdata")
        assert content.type == "audio"
        assert content.data == "base64pcmdata"
        assert content.sample_rate == 16000
    
    def test_audio_content_invalid_sample_rate(self):
        """AudioContent 采样率必须为 16000"""
        with pytest.raises(ValidationError) as exc_info:
            AudioContent(data="base64pcmdata", sample_rate=44100)
        assert "采样率必须为 16000" in str(exc_info.value)
    
    def test_audio_content_requires_data(self):
        """AudioContent 必须提供 data"""
        with pytest.raises(ValidationError):
            AudioContent()


# =============================================================================
# 测试消息类型
# =============================================================================

class TestMessage:
    """测试 Message 类型"""
    
    def test_message_text_content(self):
        """Message 使用字符串内容"""
        msg = Message(role=Role.USER, content="你好")
        assert msg.role == Role.USER
        assert msg.content == "你好"
    
    def test_message_multimodal_content(self):
        """Message 使用多模态内容"""
        msg = Message(
            role=Role.USER,
            content=[
                TextContent(text="描述这张图片"),
                ImageContent(data="base64encodedimage"),
            ]
        )
        assert msg.role == Role.USER
        assert len(msg.content) == 2
        assert isinstance(msg.content[0], TextContent)
        assert isinstance(msg.content[1], ImageContent)
    
    def test_message_system_role(self):
        """Message 支持 system 角色"""
        msg = Message(role=Role.SYSTEM, content="你是一个助手")
        assert msg.role == Role.SYSTEM
    
    def test_message_assistant_role(self):
        """Message 支持 assistant 角色"""
        msg = Message(role=Role.ASSISTANT, content="好的，我来帮你")
        assert msg.role == Role.ASSISTANT


# =============================================================================
# 测试 TTS 配置
# =============================================================================

class TestTTSConfig:
    """测试 TTSConfig"""
    
    def test_tts_disabled_by_default(self):
        """TTS 默认禁用"""
        config = TTSConfig()
        assert config.enabled is False
    
    def test_tts_default_mode(self):
        """TTS 默认模式为 audio_assistant"""
        config = TTSConfig()
        assert config.mode == TTSMode.AUDIO_ASSISTANT
    
    def test_tts_enabled_without_ref_audio_allowed(self):
        """启用 TTS 不提供参考音频也允许构造（Processor 层运行时补上）

        v1.0.0 改动：服务场景下前端不传 ref_audio，由 Worker 默认提供。
        TTSConfig 不再强制校验 ref_audio，改为 Processor 层处理。
        """
        config = TTSConfig(enabled=True, mode=TTSMode.AUDIO_ASSISTANT)
        assert config.enabled is True
        assert config.ref_audio_path is None
        assert config.ref_audio_data is None
    
    def test_tts_enabled_with_ref_audio_path(self):
        """启用 TTS 提供参考音频路径"""
        config = TTSConfig(
            enabled=True,
            mode=TTSMode.AUDIO_ASSISTANT,
            ref_audio_path="/path/to/ref.wav"
        )
        assert config.enabled is True
        assert config.ref_audio_path == "/path/to/ref.wav"
    
    def test_tts_enabled_with_ref_audio_data(self):
        """启用 TTS 提供参考音频数据"""
        config = TTSConfig(
            enabled=True,
            mode=TTSMode.AUDIO_ASSISTANT,
            ref_audio_data="base64data"
        )
        assert config.enabled is True
        assert config.ref_audio_data == "base64data"
    
    def test_tts_default_mode_no_ref_audio_required(self):
        """default 模式不需要参考音频（但会被忽略）"""
        config = TTSConfig(enabled=True, mode=TTSMode.DEFAULT)
        assert config.enabled is True
        # 注意：default 模式会忽略 ref_audio，这是模型的行为


# =============================================================================
# 测试生成配置
# =============================================================================

class TestGenerationConfig:
    """测试 GenerationConfig"""
    
    def test_default_values(self):
        """默认值"""
        config = GenerationConfig()
        assert config.max_new_tokens == 512
        assert config.do_sample is True
        assert config.temperature == 0.7
        assert config.top_p == 0.8
        assert config.top_k == 100
    
    def test_max_new_tokens_range(self):
        """max_new_tokens 范围验证"""
        config = GenerationConfig(max_new_tokens=1)
        assert config.max_new_tokens == 1
        
        config = GenerationConfig(max_new_tokens=4096)
        assert config.max_new_tokens == 4096
        
        with pytest.raises(ValidationError):
            GenerationConfig(max_new_tokens=0)
        
        with pytest.raises(ValidationError):
            GenerationConfig(max_new_tokens=5000)
    
    def test_temperature_range(self):
        """temperature 范围验证"""
        config = GenerationConfig(temperature=0.0)
        assert config.temperature == 0.0
        
        config = GenerationConfig(temperature=2.0)
        assert config.temperature == 2.0
        
        with pytest.raises(ValidationError):
            GenerationConfig(temperature=-0.1)
        
        with pytest.raises(ValidationError):
            GenerationConfig(temperature=2.1)


# =============================================================================
# 测试 ChatRequest
# =============================================================================

class TestChatRequest:
    """测试 ChatRequest"""
    
    def test_minimal_request(self):
        """最小请求"""
        request = ChatRequest(
            messages=[Message(role=Role.USER, content="你好")]
        )
        assert len(request.messages) == 1
        assert request.generation.max_new_tokens == 512  # 默认值
        assert request.tts.enabled is False  # 默认禁用
    
    def test_request_with_generation_config(self):
        """带生成配置的请求"""
        request = ChatRequest(
            messages=[Message(role=Role.USER, content="你好")],
            generation=GenerationConfig(max_new_tokens=100, temperature=0.5)
        )
        assert request.generation.max_new_tokens == 100
        assert request.generation.temperature == 0.5
    
    def test_request_with_tts(self):
        """带 TTS 配置的请求"""
        request = ChatRequest(
            messages=[Message(role=Role.USER, content="你好")],
            tts=TTSConfig(
                enabled=True,
                mode=TTSMode.AUDIO_ASSISTANT,
                ref_audio_path="/path/to/ref.wav"
            )
        )
        assert request.tts.enabled is True
    
    def test_request_requires_messages(self):
        """请求必须包含消息"""
        with pytest.raises(ValidationError):
            ChatRequest(messages=[])
    
    def test_multi_turn_request(self):
        """多轮对话请求"""
        request = ChatRequest(
            messages=[
                Message(role=Role.USER, content="我叫小明"),
                Message(role=Role.ASSISTANT, content="你好小明"),
                Message(role=Role.USER, content="我叫什么名字？"),
            ]
        )
        assert len(request.messages) == 3
    
    def test_multimodal_request(self):
        """多模态请求"""
        request = ChatRequest(
            messages=[
                Message(role=Role.USER, content=[
                    ImageContent(data="base64encodedimage"),
                    TextContent(text="描述这张图片"),
                ])
            ]
        )
        assert len(request.messages) == 1
        assert len(request.messages[0].content) == 2


# =============================================================================
# 测试 ChatResponse
# =============================================================================

class TestChatResponse:
    """测试 ChatResponse"""
    
    def test_minimal_response(self):
        """最小响应"""
        response = ChatResponse(text="你好")
        assert response.text == "你好"
        assert response.success is True
        assert response.audio_path is None
        assert response.error is None
    
    def test_response_with_audio(self):
        """带音频的响应"""
        response = ChatResponse(
            text="你好",
            audio_path="/path/to/output.wav",
            duration_ms=1234.5
        )
        assert response.audio_path == "/path/to/output.wav"
        assert response.duration_ms == 1234.5
    
    def test_error_response(self):
        """错误响应"""
        response = ChatResponse(
            text="",
            error="推理失败",
            success=False
        )
        assert response.success is False
        assert response.error == "推理失败"


# =============================================================================
# 测试序列化/反序列化
# =============================================================================

class TestSerialization:
    """测试 JSON 序列化/反序列化"""
    
    def test_chat_request_to_json(self):
        """ChatRequest 序列化为 JSON"""
        request = ChatRequest(
            messages=[Message(role=Role.USER, content="你好")]
        )
        json_str = request.model_dump_json()
        assert "你好" in json_str
        assert "user" in json_str
    
    def test_chat_request_from_json(self):
        """ChatRequest 从 JSON 反序列化"""
        json_data = {
            "messages": [{"role": "user", "content": "你好"}]
        }
        request = ChatRequest.model_validate(json_data)
        assert request.messages[0].content == "你好"
    
    def test_multimodal_request_serialization(self):
        """多模态请求序列化"""
        request = ChatRequest(
            messages=[
                Message(role=Role.USER, content=[
                    TextContent(text="描述"),
                    ImageContent(data="base64encodedimg"),
                ])
            ]
        )
        json_str = request.model_dump_json()
        assert "描述" in json_str
        assert "base64encodedimg" in json_str
    
    def test_chat_response_serialization(self):
        """ChatResponse 序列化"""
        response = ChatResponse(
            text="回答",
            audio_path="/path/to/audio.wav",
            duration_ms=1234.5
        )
        json_str = response.model_dump_json()
        assert "回答" in json_str
        assert "1234.5" in json_str


# =============================================================================
# 测试流式请求/响应
# =============================================================================

class TestStreamingSchemas:
    """测试流式推理 Schema"""
    
    def test_streaming_config_defaults(self):
        """StreamingConfig 默认值"""
        from core.schemas import StreamingConfig
        config = StreamingConfig()
        assert config.generate_audio is True
        assert config.audio_token_chunk_size == 25
    
    def test_streaming_request_minimal(self):
        """StreamingRequest 最小请求"""
        from core.schemas import StreamingRequest
        request = StreamingRequest(
            session_id="test_001",
            messages=[Message(role=Role.USER, content="你好")],
        )
        assert request.session_id == "test_001"
        assert request.is_last_chunk is False
        assert request.omni_mode is True
    
    def test_streaming_request_with_config(self):
        """StreamingRequest 带配置"""
        from core.schemas import StreamingRequest, StreamingConfig
        request = StreamingRequest(
            session_id="test_002",
            messages=[Message(role=Role.USER, content="你好")],
            is_last_chunk=True,
            streaming=StreamingConfig(generate_audio=False),
        )
        assert request.is_last_chunk is True
        assert request.streaming.generate_audio is False
    
    def test_streaming_chunk(self):
        """StreamingChunk"""
        from core.schemas import StreamingChunk
        chunk = StreamingChunk(
            chunk_index=0,
            text_delta="你好",
            audio_data="base64data",
            is_final=False,
        )
        assert chunk.chunk_index == 0
        assert chunk.text_delta == "你好"
        assert chunk.audio_sample_rate == 24000
    
    def test_streaming_chunk_final(self):
        """StreamingChunk 最终块"""
        from core.schemas import StreamingChunk
        chunk = StreamingChunk(
            chunk_index=5,
            is_final=True,
            duration_ms=1000.0,
        )
        assert chunk.is_final is True
        assert chunk.text_delta is None
        assert chunk.audio_data is None
    
    def test_streaming_response(self):
        """StreamingResponse"""
        from core.schemas import StreamingResponse
        response = StreamingResponse(
            session_id="test_001",
            full_text="你好世界",
            total_chunks=3,
            total_duration_ms=1500.0,
        )
        assert response.session_id == "test_001"
        assert response.success is True


# =============================================================================
# 测试双工请求/响应
# =============================================================================

class TestDuplexSchemas:
    """测试双工对话 Schema"""
    
    def test_duplex_config_defaults(self):
        """DuplexConfig 默认值"""
        from core.schemas import DuplexConfig
        config = DuplexConfig()
        assert config.generate_audio is True
        assert config.ls_mode == "explicit"
        assert config.max_new_speak_tokens_per_chunk == 20
        assert config.temperature == 0.7
    
    def test_duplex_config_custom(self):
        """DuplexConfig 自定义值"""
        from core.schemas import DuplexConfig
        config = DuplexConfig(
            generate_audio=False,
            temperature=0.5,
            top_k=50,
        )
        assert config.generate_audio is False
        assert config.temperature == 0.5
        assert config.top_k == 50
    
    def test_duplex_prepare_request(self):
        """DuplexPrepareRequest"""
        from core.schemas import DuplexPrepareRequest
        request = DuplexPrepareRequest(
            prefix_system_prompt="你是助手",
            suffix_system_prompt="请简短回复",
            ref_audio_path="/path/to/audio.wav",
        )
        assert request.prefix_system_prompt == "你是助手"
        assert request.suffix_system_prompt == "请简短回复"
    
    def test_duplex_prefill_request(self):
        """DuplexPrefillRequest"""
        from core.schemas import DuplexPrefillRequest
        request = DuplexPrefillRequest(
            audio_path="/path/to/user_audio.wav",
            max_slice_nums=2,
        )
        assert request.audio_path == "/path/to/user_audio.wav"
        assert request.max_slice_nums == 2
    
    def test_duplex_generate_result(self):
        """DuplexGenerateResult"""
        from core.schemas import DuplexGenerateResult
        result = DuplexGenerateResult(
            is_listen=False,
            text="你好",
            audio_data="base64data",
            end_of_turn=False,
            current_time=5,
        )
        assert result.is_listen is False
        assert result.text == "你好"
        assert result.end_of_turn is False
    
    def test_duplex_generate_result_listen(self):
        """DuplexGenerateResult listen 状态"""
        from core.schemas import DuplexGenerateResult
        result = DuplexGenerateResult(
            is_listen=True,
            end_of_turn=False,
        )
        assert result.is_listen is True
        assert result.text == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
