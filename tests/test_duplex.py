"""DuplexView 集成测试（数据驱动 + 状态测试）

测试 Duplex 模式的功能和状态管理。

**设计原则**：
- 数据测试：resources/cases/duplex/*.json
- 特殊构造逻辑在测试准备阶段完成，生成标准 DuplexOfflineInput
- input.json / output.json 始终是标准 Schema
- 状态测试：验证 prepare/prefill/generate 流程

运行命令：
cd /user/sunweiyue/lib/swy-dev/minicpmo45_service
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/base/bin/python -m pytest tests/test_duplex.py -v -s
"""

import sys
import time
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np
import pytest
import soundfile as sf
import torch
from PIL import Image

# 添加 tests 目录到 path
_tests_dir = Path(__file__).parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))

from conftest import (
    CaseSaver,
    MODEL_PATH,
    REF_AUDIO_PATH,
    INPUT_DIR,
    get_cases,
    load_case,
    assert_expected,
    skip_if_placeholder_model_path,
)

from core.schemas import (
    DuplexConfig,
    DuplexOfflineInput,
    DuplexOfflineOutput,
    DuplexChunkResult,
)
from core.processors import UnifiedProcessor, DuplexView


# =============================================================================
# Fixture: 共享的 Processor 实例
# =============================================================================

@pytest.fixture(scope="module")
def processor():
    """创建共享的 DuplexView 实例"""
    skip_if_placeholder_model_path()
    from conftest import PT_PATH
    print(f"\n[Setup] 加载 Duplex 模型: {MODEL_PATH}")
    print(f"[Setup] 额外权重: {PT_PATH}")
    unified = UnifiedProcessor(model_path=MODEL_PATH, pt_path=PT_PATH)
    duplex_view = unified.set_duplex_mode()
    yield duplex_view
    print("\n[Teardown] 释放模型")
    del unified
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# 数据构造辅助函数
# =============================================================================

def build_duplex_input(case_data: dict, saver: CaseSaver) -> DuplexOfflineInput:
    """从 case JSON 构造标准 DuplexOfflineInput
    
    支持的构造模式：
    1. 基础模式：直接使用 user_audio_path
    2. 图像过渡模式：根据 total_duration_s 和 image_transition_at_s 构造
    """
    input_data = case_data["input"]
    
    # 复制参考音频
    saver.copy_input_file(Path(input_data["ref_audio_path"]), "ref_audio.wav")
    
    # 处理用户音频
    if "total_duration_s" in input_data:
        # 图像过渡模式：需要构造指定时长的音频
        user_audio_path = _build_extended_audio(input_data, saver)
    else:
        # 基础模式：直接使用
        user_audio_path = input_data["user_audio_path"]
        saver.copy_input_file(Path(user_audio_path), "user_audio.wav")
    
    # 处理图像列表
    image_paths = None
    if "image_paths" in input_data:
        # 直接指定图像列表
        image_paths = input_data["image_paths"]
        for i, p in enumerate(image_paths):
            src = Path(p)
            if src.exists():
                saver.copy_input_file(src, f"image_{i}.png")
    elif "image_transition_at_s" in input_data:
        # 图像过渡模式：构造黑屏→真实图像的列表
        image_paths = _build_transition_images(input_data, saver)
    
    # 构造标准 DuplexOfflineInput
    return DuplexOfflineInput(
        system_prompt=input_data["system_prompt"],
        user_audio_path=str(user_audio_path),
        ref_audio_path=input_data["ref_audio_path"],
        image_paths=image_paths,
        config=DuplexConfig(**input_data.get("config", {})),
    )


def _build_extended_audio(input_data: dict, saver: CaseSaver) -> Path:
    """构造指定时长的用户音频（原音频 + 静音填充）"""
    user_audio_path = Path(input_data["user_audio_path"])
    total_duration = input_data["total_duration_s"]
    
    # 加载原始音频
    audio_start, _ = librosa.load(str(user_audio_path), sr=16000, mono=True)
    
    # 构造完整音频
    total_samples = total_duration * 16000
    audio_full = np.zeros(total_samples, dtype=np.float32)
    audio_full[:len(audio_start)] = audio_start
    
    # 保存
    output_path = saver.base_dir / "user_audio_extended.wav"
    sf.write(str(output_path), audio_full, 16000)
    saver.copy_input_file(user_audio_path, "user_audio_original.wav")
    
    return output_path


def _build_transition_images(input_data: dict, saver: CaseSaver) -> List[str]:
    """构造黑屏→真实图像的过渡列表"""
    image_path = Path(input_data["image_path"])
    total_duration = input_data["total_duration_s"]
    transition_at = input_data["image_transition_at_s"]
    
    # 加载真实图像
    real_image = Image.open(image_path)
    saver.copy_input_file(image_path, "real_image.png")
    
    # 创建黑屏图像
    black_image = Image.new("RGB", real_image.size, (0, 0, 0))
    black_path = saver.base_dir / "black_image.png"
    black_image.save(black_path)
    
    # 构造图像列表
    image_paths = []
    for i in range(total_duration):
        if i < transition_at:
            image_paths.append(str(black_path))
        else:
            image_paths.append(str(image_path))
    
    return image_paths


# =============================================================================
# 辅助函数
# =============================================================================

def run_duplex_offline(
    processor: DuplexView,
    task_input: DuplexOfflineInput,
    saver: CaseSaver,
) -> DuplexOfflineOutput:
    """执行双工离线推理并保存 chunks
    
    使用 processor.offline_inference() 执行推理，
    然后将 chunks 保存到 CaseSaver。
    """
    import base64
    
    # 执行离线推理
    result = processor.offline_inference(task_input)
    
    # 收集所有音频
    all_audio = []
    
    # 保存 chunks（如果有）
    if result.chunks:
        for chunk in result.chunks:
            # 解码音频数据
            audio_np = None
            if chunk.audio_data:
                audio_bytes = base64.b64decode(chunk.audio_data)
                audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
                all_audio.append(audio_np)
            
            # 保存 chunk（包含音频）
            saver.save_chunk(
                chunk.chunk_idx, 
                chunk.model_dump(exclude={"audio_data"}),  # 排除 base64 数据避免 JSON 过大
                audio_data=audio_np,
                sample_rate=24000
            )
    
    # 合并并保存所有音频
    if all_audio:
        combined_audio = np.concatenate(all_audio)
        saver.save_output_audio(combined_audio, "combined.wav", sample_rate=24000)
    
    return result


# =============================================================================
# 数据测试（Data-Driven）
# =============================================================================

class TestDuplexData:
    """Duplex 数据测试 - 所有 case 共用一个测试方法"""
    
    @pytest.mark.parametrize("case_name", get_cases("duplex"))
    def test_duplex(self, processor, case_saver, case_name: str):
        """数据驱动的 Duplex 测试"""
        saver: CaseSaver = case_saver(case_name, "duplex")
        
        # 加载 case 数据
        case = load_case("duplex", case_name, output_dir=saver.base_dir)
        print(f"\n📋 {case['description']}")
        
        # 构造标准输入（特殊构造逻辑在 build_duplex_input 中处理）
        task_input = build_duplex_input(case, saver)
        
        # 保存输入（标准 Schema）
        saver.save_input(task_input)
        
        # 执行推理
        response = run_duplex_offline(processor, task_input, saver)
        
        # 保存输出（标准 Schema）
        saver.save_output(response)
        saver.finalize({"case_name": case_name, "description": case["description"]})
        
        # 验证
        assert_expected(response, case["expected"], output_dir=saver.base_dir)
        
        print(f"✅ duplex_{case_name}: {response.full_text[:80]}{'...' if len(response.full_text) > 80 else ''}")


# =============================================================================
# 状态测试（State Tests）
# =============================================================================

class TestDuplexState:
    """Duplex 状态测试 - 验证 prepare/prefill/generate 流程"""
    
    def test_prepare_prefill_generate_cycle(self, processor, case_saver):
        """测试：prepare → prefill → generate 完整流程"""
        saver: CaseSaver = case_saver("state_prepare_prefill_generate", "duplex")
        
        # prepare（使用正确的参数名）
        processor.prepare(
            system_prompt_text="你是一个助手。",
            ref_audio_path=str(REF_AUDIO_PATH),
        )
        
        # prefill（静音 chunk，使用正确的参数名）
        silent_chunk = np.zeros(16000, dtype=np.float32)
        processor.prefill(audio_waveform=silent_chunk)
        
        # generate（无参数）
        result = processor.generate()
        
        saver.save_output({
            "is_listen": result.is_listen,
            "text": result.text or "",
            "has_audio": result.audio_data is not None,
        })
        saver.finalize({"test": "prepare_prefill_generate_cycle"})
        
        print(f"✅ prepare/prefill/generate 流程测试完成")
        print(f"   is_listen: {result.is_listen}")
        print(f"   text: {(result.text or '')[:50]}")
    
    def test_offline_inference(self, processor, case_saver):
        """测试：offline_inference 便捷方法"""
        saver: CaseSaver = case_saver("state_offline_inference", "duplex")
        
        # 使用 offline_inference
        task_input = DuplexOfflineInput(
            system_prompt="你是一个助手。",
            user_audio_path=str(INPUT_DIR / "user_audio" / "000_user_audio0.wav"),
            ref_audio_path=str(REF_AUDIO_PATH),
            config=DuplexConfig(force_listen_count=3),
        )
        
        saver.copy_input_file(REF_AUDIO_PATH, "ref_audio.wav")
        saver.save_input(task_input)
        
        response = processor.offline_inference(task_input)
        
        saver.save_output(response)
        saver.finalize({"test": "offline_inference"})
        
        assert response.success, f"offline_inference 应该成功，但返回: {response.error}"
        print(f"✅ offline_inference 测试完成")
        print(f"   text: {response.full_text[:80]}")
        print(f"   audio: {response.audio_duration_s:.2f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
