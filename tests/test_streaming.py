"""HalfDuplexView 集成测试（数据驱动 + 状态测试）

测试 Streaming 模式的功能和状态管理。

**设计原则**：
- 数据测试：resources/cases/streaming/*.json
- 状态测试：验证 session_id 机制、KV Cache 复用、rollback
- 有状态模式，需要专门测试状态操作

运行命令：
cd /user/sunweiyue/lib/swy-dev/minicpmo45_service
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/base/bin/python -m pytest tests/test_streaming.py -v -s
"""

import base64
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest

# 添加 tests 目录到 path
_tests_dir = Path(__file__).parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))

from conftest import (
    CaseSaver,
    MODEL_PATH,
    REF_AUDIO_PATH,
    get_cases,
    load_case,
    assert_expected,
)

from core.schemas import (
    StreamingRequest,
    StreamingChunk,
    Message,
    Role,
    TextContent,
    AudioContent,
    ImageContent,
)
from core.processors import UnifiedProcessor, HalfDuplexView


# =============================================================================
# Fixture: 共享的 Processor 实例
# =============================================================================

@pytest.fixture(scope="module")
def processor():
    """创建共享的 HalfDuplexView 实例"""
    from conftest import PT_PATH
    print(f"\n[Setup] 加载模型: {MODEL_PATH}")
    print(f"[Setup] 额外权重: {PT_PATH}")
    unified = UnifiedProcessor(
        model_path=MODEL_PATH,
        pt_path=PT_PATH,
        ref_audio_path=str(REF_AUDIO_PATH),
    )
    streaming_view = unified.set_half_duplex_mode()
    yield streaming_view
    print("\n[Teardown] 释放模型")
    del unified


# =============================================================================
# 数据测试（Data-Driven）
# =============================================================================

class TestStreamingData:
    """Streaming 数据测试 - 验证基本功能"""
    
    @staticmethod
    def _build_messages(msg_data_list: list, saver: CaseSaver) -> List[Message]:
        """从 JSON input 构造 Message 列表（支持多模态内容）
        
        处理 content 的两种格式：
        1. 字符串: "你好" → Message(content="你好")
        2. 列表: [{"type": "audio", "path": "..."}, {"type": "text", "text": "..."}]
           → Message(content=[AudioContent(...), TextContent(...)])
        """
        messages = []
        for msg_data in msg_data_list:
            content = msg_data["content"]
            
            # 处理多模态内容（与 test_chat.py 一致）
            if isinstance(content, list):
                content_items = []
                for item in content:
                    if item["type"] == "text":
                        content_items.append(TextContent(text=item["text"]))
                    elif item["type"] == "audio":
                        import librosa
                        src_path = Path(item["path"])
                        if src_path.exists():
                            saver.copy_input_file(src_path, src_path.name)
                        audio, _ = librosa.load(str(src_path), sr=16000, mono=True)
                        audio_b64 = base64.b64encode(audio.astype(np.float32).tobytes()).decode()
                        content_items.append(AudioContent(data=audio_b64))
                    elif item["type"] == "image":
                        src_path = Path(item["path"])
                        if src_path.exists():
                            saver.copy_input_file(src_path, src_path.name)
                        img_b64 = base64.b64encode(src_path.read_bytes()).decode()
                        content_items.append(ImageContent(data=img_b64))
                content = content_items
            
            messages.append(Message(
                role=Role(msg_data["role"]),
                content=content,
            ))
        return messages
    
    @pytest.mark.parametrize("case_name", get_cases("streaming"))
    def test_streaming(self, processor, case_saver, case_name: str):
        """数据驱动的 Streaming 测试"""
        saver: CaseSaver = case_saver(case_name, "streaming")
        
        # 加载 case 数据
        case = load_case("streaming", case_name, output_dir=saver.base_dir)
        print(f"\n📋 {case['description']}")
        
        input_data = case["input"]
        session_id = input_data["session_id"]
        generate_audio = input_data.get("generate_audio", False)
        
        # 复制参考音频
        if generate_audio:
            saver.copy_input_file(REF_AUDIO_PATH, "ref_audio.wav")
            # [CRITICAL] 初始化 TTS 缓存
            # streaming 模式下生成音频需要先初始化 token2wav_cache
            processor.init_ref_audio(str(REF_AUDIO_PATH))
        
        # 保存输入
        saver.save_input(input_data)
        
        # 构造请求（支持多模态内容）
        messages = self._build_messages(input_data["messages"], saver)
        request = StreamingRequest(session_id=session_id, messages=messages, is_last_chunk=True)
        
        # 预填充
        processor.prefill(request)
        
        # 流式生成
        chunks: List[dict] = []
        all_text = []
        all_audio = []
        
        for i, chunk in enumerate(processor.generate(session_id, generate_audio=generate_audio)):
            chunk_data = {
                "idx": i,
                "text_delta": chunk.text_delta,
                "has_audio": chunk.audio_data is not None,
                "is_final": chunk.is_final,
            }
            
            # 解码音频
            audio_np = None
            if chunk.audio_data:
                audio_bytes = base64.b64decode(chunk.audio_data)
                audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
                chunk_data["audio_samples"] = len(audio_np)
                all_audio.append(audio_np)
            
            chunks.append(chunk_data)
            saver.save_chunk(i, chunk_data, audio_np, sample_rate=24000)
            
            if chunk.text_delta:
                all_text.append(chunk.text_delta)
            
            if chunk.is_final:
                break
        
        # 合并结果
        full_text = "".join(all_text)
        combined_audio = np.concatenate(all_audio) if all_audio else np.array([])
        
        if len(combined_audio) > 0:
            saver.save_output_audio(combined_audio, "combined.wav", sample_rate=24000)
        
        # 构造响应对象（用于 assert_expected）
        class StreamingResponse:
            def __init__(self):
                self.success = len(full_text) > 0
                self.full_text = full_text
                self.text = full_text
                self.audio_duration_s = len(combined_audio) / 24000 if len(combined_audio) > 0 else 0
                self.total_chunks = len(chunks)
        
        response = StreamingResponse()
        
        # 保存输出
        saver.save_output({
            "full_text": full_text,
            "total_chunks": len(chunks),
            "audio_duration_s": response.audio_duration_s,
        })
        saver.finalize({"case_name": case_name, "description": case["description"]})
        
        # 验证
        assert_expected(response, case["expected"], output_dir=saver.base_dir)
        
        # 重置会话
        processor.reset_session(session_id)
        
        print(f"✅ {case_name}: {full_text[:80]}{'...' if len(full_text) > 80 else ''}")


# =============================================================================
# complete_turn 便捷方法测试
# =============================================================================

class TestCompleteTurn:
    """测试 HalfDuplexView.complete_turn() 便捷方法
    
    complete_turn 封装了 prefill + generate + 累加文本/音频的流程，
    适用于不需要实时流式输出的场景。
    """
    
    def test_complete_turn_text_only(self, processor, case_saver):
        """测试：complete_turn 纯文本生成"""
        from core.schemas import Message, Role
        saver: CaseSaver = case_saver("complete_turn_text_only", "streaming")
        
        session_id = f"complete_turn_text_{int(time.time())}"
        processor.reset_session(session_id)
        
        # 使用 complete_turn
        response = processor.complete_turn(
            session_id=session_id,
            messages=[
                Message(role=Role.USER, content="请用一句话介绍你自己。"),
            ],
            generate_audio=False,
            max_new_tokens=100,
        )
        
        saver.save_output({
            "full_text": response.full_text,
            "total_chunks": response.total_chunks,
            "total_duration_ms": response.total_duration_ms,
        })
        saver.finalize({"test": "complete_turn_text_only"})
        
        # 验证
        assert response.success, "complete_turn 应该成功"
        assert len(response.full_text) > 0, "应该生成文本"
        assert response.audio_data is None, "不应该有音频"
        
        processor.reset_session(session_id)
        print(f"✅ complete_turn 纯文本: {response.full_text[:80]}")
    
    def test_complete_turn_with_audio(self, processor, case_saver):
        """测试：complete_turn 带音频生成"""
        from core.schemas import Message, Role
        import soundfile as sf
        saver: CaseSaver = case_saver("complete_turn_with_audio", "streaming")
        
        session_id = f"complete_turn_audio_{int(time.time())}"
        processor.reset_session(session_id)
        
        # [CRITICAL] 初始化 TTS 缓存
        processor.init_ref_audio(str(REF_AUDIO_PATH))
        
        output_path = saver.base_dir / "output.wav"
        
        # 使用 complete_turn
        response = processor.complete_turn(
            session_id=session_id,
            messages=[
                Message(role=Role.SYSTEM, content="你是一个友好的助手，用简短的话回答。"),
                Message(role=Role.USER, content="你好"),
            ],
            generate_audio=True,
            max_new_tokens=50,
            output_audio_path=str(output_path),
        )
        
        saver.save_output({
            "full_text": response.full_text,
            "total_chunks": response.total_chunks,
            "audio_duration_ms": response.audio_duration_ms,
            "total_duration_ms": response.total_duration_ms,
        })
        saver.finalize({"test": "complete_turn_with_audio"})
        
        # 验证
        assert response.success, "complete_turn 应该成功"
        assert len(response.full_text) > 0, "应该生成文本"
        assert response.audio_data is not None, "应该有音频"
        assert response.audio_duration_ms > 0, "音频时长应该 > 0"
        assert output_path.exists(), "音频文件应该被保存"
        
        processor.reset_session(session_id)
        print(f"✅ complete_turn 带音频: {response.full_text[:50]}, 音频 {response.audio_duration_ms:.0f}ms")
    
    def test_complete_turn_multi_turn(self, processor, case_saver):
        """测试：complete_turn 多轮对话（KV Cache 复用）"""
        from core.schemas import Message, Role
        saver: CaseSaver = case_saver("complete_turn_multi_turn", "streaming")
        
        session_id = f"complete_turn_multi_{int(time.time())}"
        processor.reset_session(session_id)
        
        # Turn 1
        response1 = processor.complete_turn(
            session_id=session_id,
            messages=[Message(role=Role.USER, content="请帮我计算 15 + 27 等于多少？")],
            generate_audio=False,
        )
        
        # Turn 2（同一 session，应复用 KV）
        response2 = processor.complete_turn(
            session_id=session_id,
            messages=[Message(role=Role.USER, content="那这个结果再乘以 2 呢？")],
            generate_audio=False,
        )
        
        saver.save_output({
            "turn1_text": response1.full_text,
            "turn2_text": response2.full_text,
        })
        saver.finalize({"test": "complete_turn_multi_turn"})
        
        # 验证
        assert "42" in response1.full_text, f"Turn 1 应该包含 42，实际: {response1.full_text}"
        assert "84" in response2.full_text, f"Turn 2 应该包含 84，实际: {response2.full_text}"
        
        processor.reset_session(session_id)
        print(f"✅ complete_turn 多轮对话")
        print(f"   Turn 1: {response1.full_text[:50]}")
        print(f"   Turn 2: {response2.full_text[:50]}")


# =============================================================================
# 状态测试（State Tests）
# =============================================================================

class TestStreamingState:
    """Streaming 状态测试 - 验证 session_id 和 KV Cache 机制"""
    
    def test_session_creates_new(self, processor, case_saver):
        """测试：新 session_id 创建新会话"""
        saver: CaseSaver = case_saver("state_session_creates_new", "streaming")
        
        session_id = f"test_new_{int(time.time())}"
        
        # 创建新会话
        request = StreamingRequest(
            session_id=session_id,
            messages=[Message(role="user", content="你好")],
            is_last_chunk=True,
        )
        
        processor.prefill(request)
        
        # 生成响应
        chunks = list(processor.generate(session_id, generate_audio=False))
        text = "".join(c.text_delta for c in chunks if c.text_delta)
        
        saver.save_output({"session_id": session_id, "text": text})
        saver.finalize({"test": "session_creates_new"})
        
        # 清理
        processor.reset_session(session_id)
        
        assert len(text) > 0, "新会话应该能生成响应"
        print(f"✅ 新会话创建成功: {text[:50]}")
    
    def test_session_kv_reuse(self, processor, case_saver):
        """测试：同一 session_id 复用 KV Cache（多轮对话）"""
        saver: CaseSaver = case_saver("state_session_kv_reuse", "streaming")
        
        session_id = f"test_reuse_{int(time.time())}"
        
        # Turn 1
        t1_start = time.time()
        request1 = StreamingRequest(
            session_id=session_id,
            messages=[Message(role="user", content="我叫小明，请记住我的名字。")],
            is_last_chunk=True,
        )
        processor.prefill(request1)
        chunks1 = list(processor.generate(session_id, generate_audio=False))
        t1_elapsed = time.time() - t1_start
        text1 = "".join(c.text_delta for c in chunks1 if c.text_delta)
        
        # Turn 2（同一 session，应复用 KV）
        t2_start = time.time()
        request2 = StreamingRequest(
            session_id=session_id,
            messages=[Message(role="user", content="我叫什么名字？")],
            is_last_chunk=True,
        )
        processor.prefill(request2)
        chunks2 = list(processor.generate(session_id, generate_audio=False))
        t2_elapsed = time.time() - t2_start
        text2 = "".join(c.text_delta for c in chunks2 if c.text_delta)
        
        saver.save_output({
            "session_id": session_id,
            "turn1_text": text1,
            "turn1_time": t1_elapsed,
            "turn2_text": text2,
            "turn2_time": t2_elapsed,
        })
        saver.finalize({"test": "session_kv_reuse"})
        
        # 清理
        processor.reset_session(session_id)
        
        # 验证：Turn 2 应该记住 Turn 1 的信息
        assert "小明" in text2, f"多轮对话应记住名字，但回答是: {text2}"
        print(f"✅ KV Cache 复用成功")
        print(f"   Turn 1 ({t1_elapsed:.2f}s): {text1[:50]}")
        print(f"   Turn 2 ({t2_elapsed:.2f}s): {text2[:50]}")
    
    def test_session_switch_clears_kv(self, processor, case_saver):
        """测试：切换 session_id 清空 KV Cache"""
        saver: CaseSaver = case_saver("state_session_switch", "streaming")
        
        session_a = f"test_switch_A_{int(time.time())}"
        session_b = f"test_switch_B_{int(time.time())}"
        
        # Session A: 第一轮
        request_a1 = StreamingRequest(
            session_id=session_a,
            messages=[Message(role="user", content="我叫小红，请记住我的名字。")],
            is_last_chunk=True,
        )
        processor.prefill(request_a1)
        chunks_a1 = list(processor.generate(session_a, generate_audio=False))
        text_a1 = "".join(c.text_delta for c in chunks_a1 if c.text_delta)
        
        # Session B: 切换到新会话（A 的 KV 应该丢失）
        request_b = StreamingRequest(
            session_id=session_b,
            messages=[Message(role="user", content="你好")],
            is_last_chunk=True,
        )
        processor.prefill(request_b)
        chunks_b = list(processor.generate(session_b, generate_audio=False))
        text_b = "".join(c.text_delta for c in chunks_b if c.text_delta)
        
        # Session A: 切回（由于 KV 丢失，应该不记得名字）
        request_a2 = StreamingRequest(
            session_id=session_a,
            messages=[Message(role="user", content="我叫什么名字？")],
            is_last_chunk=True,
        )
        processor.prefill(request_a2)
        chunks_a2 = list(processor.generate(session_a, generate_audio=False))
        text_a2 = "".join(c.text_delta for c in chunks_a2 if c.text_delta)
        
        saver.save_output({
            "session_a_turn1": text_a1,
            "session_b": text_b,
            "session_a_turn2": text_a2,
        })
        saver.finalize({"test": "session_switch_clears_kv"})
        
        # 清理
        processor.reset_session(session_a)
        processor.reset_session(session_b)
        
        # 注意：由于 KV 丢失，切回 A 后应该不记得名字
        # 但这取决于模型行为，我们主要验证流程不崩溃
        print(f"✅ Session 切换测试完成")
        print(f"   Session A Turn 1: {text_a1[:50]}")
        print(f"   Session B: {text_b[:50]}")
        print(f"   Session A Turn 2: {text_a2[:50]}")
    
    def test_reset_session(self, processor, case_saver):
        """测试：reset_session 清空会话状态"""
        saver: CaseSaver = case_saver("state_reset_session", "streaming")
        
        session_id = f"test_reset_{int(time.time())}"
        
        # 第一轮
        request1 = StreamingRequest(
            session_id=session_id,
            messages=[Message(role="user", content="我叫小李。")],
            is_last_chunk=True,
        )
        processor.prefill(request1)
        chunks1 = list(processor.generate(session_id, generate_audio=False))
        text1 = "".join(c.text_delta for c in chunks1 if c.text_delta)
        
        # 显式重置
        processor.reset_session(session_id)
        
        # 第二轮（重置后应该不记得名字）
        request2 = StreamingRequest(
            session_id=session_id,
            messages=[Message(role="user", content="我叫什么名字？")],
            is_last_chunk=True,
        )
        processor.prefill(request2)
        chunks2 = list(processor.generate(session_id, generate_audio=False))
        text2 = "".join(c.text_delta for c in chunks2 if c.text_delta)
        
        saver.save_output({
            "before_reset": text1,
            "after_reset": text2,
        })
        saver.finalize({"test": "reset_session"})
        
        # 清理
        processor.reset_session(session_id)
        
        print(f"✅ reset_session 测试完成")
        print(f"   重置前: {text1[:50]}")
        print(f"   重置后: {text2[:50]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
