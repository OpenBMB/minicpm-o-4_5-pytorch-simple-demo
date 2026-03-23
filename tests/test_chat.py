"""ChatView 集成测试（数据驱动）

测试 Chat 模式的各种输入类型。

**设计原则**：
- 测试数据存放在 resources/cases/chat/*.json
- 每个 JSON 文件包含 description、input、expected
- 测试代码只负责加载数据、执行、验证
- 无状态模式，不需要状态测试

运行命令：
cd /user/sunweiyue/lib/swy-dev/minicpmo45_service
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/base/bin/python -m pytest tests/test_chat.py -v -s
"""

import sys
import base64
from pathlib import Path

import pytest

# 添加 tests 目录到 path
_tests_dir = Path(__file__).parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))

from conftest import (
    CaseSaver,
    MODEL_PATH,
    get_cases,
    load_case,
    assert_expected,
    skip_if_placeholder_model_path,
)

from core.schemas import (
    ChatRequest,
    Message,
    Role,
    TextContent,
    ImageContent,
    AudioContent,
    TTSConfig,
    TTSMode,
    GenerationConfig,
)
from core.processors import UnifiedProcessor, ChatView


# =============================================================================
# Fixture: 共享的 Processor 实例（避免重复加载模型）
# =============================================================================

@pytest.fixture(scope="module")
def processor():
    """创建共享的 ChatView 实例"""
    skip_if_placeholder_model_path()
    from conftest import PT_PATH
    print(f"\n[Setup] 加载模型: {MODEL_PATH}")
    print(f"[Setup] 额外权重: {PT_PATH}")
    unified = UnifiedProcessor(model_path=MODEL_PATH, pt_path=PT_PATH)
    chat_view = unified.set_chat_mode()
    yield chat_view
    print("\n[Teardown] 释放模型")
    del unified


# =============================================================================
# 数据测试（Data-Driven）
# =============================================================================

class TestChatData:
    """Chat 数据测试 - 所有 case 共用一个测试方法"""
    
    @pytest.mark.parametrize("case_name", get_cases("chat"))
    def test_chat(self, processor, case_saver, case_name: str):
        """数据驱动的 Chat 测试"""
        saver: CaseSaver = case_saver(case_name, "chat")
        
        # 加载 case 数据
        case = load_case("chat", case_name, output_dir=saver.base_dir)
        print(f"\n📋 {case['description']}")
        
        # 构造请求
        request = self._build_request(case["input"], saver)
        
        # 保存输入
        saver.save_input(request)
        
        # 执行推理
        response = processor.chat(request)
        
        # 保存输出
        saver.save_output(response)
        saver.finalize({"case_name": case_name, "description": case["description"]})
        
        # 验证
        assert_expected(response, case["expected"], output_dir=saver.base_dir)
        
        # 打印结果
        text = response.text or ""
        print(f"✅ {case_name}: {text[:80]}{'...' if len(text) > 80 else ''}")
    
    def _build_request(self, input_data: dict, saver: CaseSaver) -> ChatRequest:
        """从 JSON input 构造 ChatRequest"""
        messages = []
        
        for msg_data in input_data.get("messages", []):
            content = msg_data["content"]
            
            # 处理复合内容（图像、音频）
            if isinstance(content, list):
                content_items = []
                for item in content:
                    if item["type"] == "text":
                        content_items.append(TextContent(text=item["text"]))
                    elif item["type"] == "image":
                        src_path = Path(item["path"])
                        if src_path.exists():
                            saver.copy_input_file(src_path, src_path.name)
                        img_b64 = base64.b64encode(src_path.read_bytes()).decode()
                        content_items.append(ImageContent(data=img_b64))
                    elif item["type"] == "audio":
                        import librosa, numpy as np
                        src_path = Path(item["path"])
                        if src_path.exists():
                            saver.copy_input_file(src_path, src_path.name)
                        audio, _ = librosa.load(str(src_path), sr=16000, mono=True)
                        audio_b64 = base64.b64encode(audio.astype(np.float32).tobytes()).decode()
                        content_items.append(AudioContent(data=audio_b64))
                content = content_items
            
            messages.append(Message(
                role=Role(msg_data["role"]),
                content=content,
            ))
        
        # 构造生成配置
        generation = None
        if "generation" in input_data:
            generation = GenerationConfig(**input_data["generation"])
        
        # 构造 TTS 配置
        tts = None
        if "tts" in input_data:
            tts_data = input_data["tts"]
            # 复制参考音频
            if "ref_audio_path" in tts_data:
                ref_path = Path(tts_data["ref_audio_path"])
                if ref_path.exists():
                    saver.copy_input_file(ref_path, "ref_audio.wav")
            tts = TTSConfig(
                enabled=tts_data.get("enabled", True),
                mode=TTSMode(tts_data.get("mode", "audio_assistant")),
                ref_audio_path=tts_data.get("ref_audio_path"),
                output_path=tts_data.get("output_path"),
            )
        
        # 只传有值的参数
        kwargs = {"messages": messages}
        if generation is not None:
            kwargs["generation"] = generation
        if tts is not None:
            kwargs["tts"] = tts
        
        return ChatRequest(**kwargs)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
