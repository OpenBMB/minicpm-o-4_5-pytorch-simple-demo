"""Pytest 配置和共享 fixtures

提供测试所需的共享配置、路径和工具函数。

数据驱动测试架构：
- cases/ 目录按处理器类型存放 JSON 测试用例
- load_case() 加载 JSON 并替换路径变量
- get_cases() 获取某类型的所有 case 名称
- assert_expected() 验证响应符合预期
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# ============================================================
# 路径配置
# ============================================================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 模型路径（开源仓库使用环境变量，避免硬编码本地路径）
MODEL_PATH = os.environ.get("MINICPMO45_MODEL_PATH", "/path/to/MiniCPM-o-4_5")
PT_PATH = os.environ.get("MINICPMO45_PT_PATH")  # 可选，Duplex 微调权重

# 测试目录
TESTS_DIR = Path(__file__).parent
CASES_DIR = TESTS_DIR / "cases"           # git 维护：测试用例定义
RESULTS_DIR = TESTS_DIR / "results"       # gitignore：测试输出结果

# 共享输入资源（位于 cases/common/）
COMMON_DIR = CASES_DIR / "common"
REF_AUDIO_PATH = COMMON_DIR / "ref_audio" / "BH-Ref-HT-F224-Ref06_82_U001_话题_3_348s-355s.wav"
USER_AUDIO_PATH = COMMON_DIR / "user_audio" / "000_user_audio0.wav"

# 向后兼容
INPUT_DIR = COMMON_DIR
OUTPUT_DIR = RESULTS_DIR


# ============================================================
# 数据驱动测试工具
# ============================================================

def get_cases(processor_type: str) -> List[str]:
    """获取某处理器类型的所有测试用例名称
    
    Args:
        processor_type: 处理器类型（chat/streaming/duplex）
    
    Returns:
        case 名称列表（不含 .json 后缀）
        
    Note:
        以 _skip_ 开头的文件会被跳过（用于保留但不执行的 case）
    """
    case_dir = CASES_DIR / processor_type
    if not case_dir.exists():
        return []
    # 过滤掉 _skip_ 前缀的文件
    return [f.stem for f in case_dir.glob("*.json") if not f.stem.startswith("_skip_")]


def load_case(processor_type: str, case_name: str, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """加载测试用例 JSON 文件，并替换路径变量
    
    支持的变量：
    - ${INPUT_DIR}: 输入资源目录
    - ${REF_AUDIO_PATH}: 参考音频路径
    - ${OUTPUT_PATH}: 输出路径（需要提供 output_dir）
    
    Args:
        processor_type: 处理器类型（chat/streaming/duplex）
        case_name: case 名称（不含 .json 后缀）
        output_dir: 输出目录（用于 TTS 输出等）
    
    Returns:
        解析后的 case 字典
    """
    case_path = CASES_DIR / processor_type / f"{case_name}.json"
    with open(case_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 替换路径变量
    content = content.replace("${INPUT_DIR}", str(INPUT_DIR))
    content = content.replace("${REF_AUDIO_PATH}", str(REF_AUDIO_PATH))
    if output_dir:
        content = content.replace("${OUTPUT_PATH}", str(output_dir / "output_audio.wav"))
    
    return json.loads(content)


def assert_expected(response, expected: Dict[str, Any], output_dir: Optional[Path] = None):
    """验证响应符合预期
    
    支持的 expected 字段：
    - success: bool - 预期是否成功
    - text_contains: list[str] - 响应文本应包含的关键词
    - text_min_length: int - 响应文本最小长度
    - has_audio: bool - 是否应产生音频
    - has_error: bool - 是否应有错误信息
    - has_chunks: bool - 是否应有流式 chunks
    
    Args:
        response: Processor 返回的响应对象
        expected: 预期结果字典
        output_dir: 输出目录（检查音频文件）
    """
    # 检查成功/失败
    if "success" in expected:
        assert response.success == expected["success"], \
            f"Expected success={expected['success']}, got {response.success}"
    
    # 检查文本包含关键词
    if "text_contains" in expected:
        text = getattr(response, "text", "") or getattr(response, "full_text", "") or ""
        for keyword in expected["text_contains"]:
            assert keyword in text, f"Expected text to contain '{keyword}', got: {text[:100]}"
    
    # 检查文本最小长度
    if "text_min_length" in expected:
        text = getattr(response, "text", "") or getattr(response, "full_text", "") or ""
        assert len(text) >= expected["text_min_length"], \
            f"Expected text length >= {expected['text_min_length']}, got {len(text)}"
    
    # 检查是否有音频
    if "has_audio" in expected and expected["has_audio"]:
        # 检查响应对象中的音频属性
        has_audio = (
            getattr(response, "audio_data", None) is not None or
            getattr(response, "audio_duration_s", 0) > 0 or
            (output_dir and (output_dir / "output_audio.wav").exists())
        )
        assert has_audio, "Expected response to have audio"
    
    # 检查是否有错误
    if "has_error" in expected and expected["has_error"]:
        assert response.error is not None, "Expected response to have error"


# ============================================================
# 测试结果保存工具
# ============================================================

class CaseSaver:
    """测试用例结果保存器
    
    保存完整的输入输出到 resources/output/<processor_type>/case_<name>/，包括：
    - input.json: 输入 schema
    - output.json: 输出 schema
    - input_audio.wav: 输入音频（如有）
    - output_audio.wav: 输出音频（如有）
    - chunks/: 分块输出（如有）
    
    使用示例：
    ```python
    saver = TestCaseSaver("duplex_basic", processor_type="duplex")
    saver.save_input({"system_prompt": "...", "user_audio": "..."})
    saver.save_input_audio(audio_data, "user_audio.wav")
    saver.save_chunk(0, {"text": "..."}, audio_chunk)
    saver.save_output({"full_text": "...", "duration_s": 5.0})
    saver.save_output_audio(combined_audio, "combined.wav")
    saver.finalize()
    ```
    """
    
    def __init__(self, case_name: str, processor_type: str, output_dir: Optional[Path] = None):
        """初始化保存器
        
        Args:
            case_name: 测试用例名称
            processor_type: 处理器类型（chat/streaming/duplex）
            output_dir: 输出目录，默认为 resources/output/
        """
        self.case_name = case_name
        self.processor_type = processor_type
        # 按 processor_type 保存到子目录
        self.base_dir = (output_dir or OUTPUT_DIR) / processor_type / f"case_{case_name}"
        self.chunks_dir = self.base_dir / "chunks"
        
        # 创建目录
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(exist_ok=True)
        
        # 元数据
        self.metadata = {
            "case_name": case_name,
            "timestamp": datetime.now().isoformat(),
            "model_path": MODEL_PATH,
        }
    
    def save_input(self, input_data, filename: str = "input.json") -> Path:
        """保存输入 schema
        
        Args:
            input_data: 输入数据，可以是 Pydantic BaseModel 或 dict
            filename: 文件名
        """
        from pydantic import BaseModel
        
        # 如果是 Pydantic 对象，自动转换为 dict
        if isinstance(input_data, BaseModel):
            data = input_data.model_dump()
        else:
            data = input_data
        
        path = self.base_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path
    
    def save_output(self, output_data, filename: str = "output.json") -> Path:
        """保存输出 schema
        
        Args:
            output_data: 输出数据，可以是 Pydantic BaseModel 或 dict
            filename: 文件名
        """
        from pydantic import BaseModel
        
        # 如果是 Pydantic 对象，自动转换为 dict
        if isinstance(output_data, BaseModel):
            data = output_data.model_dump()
        else:
            data = output_data
        
        data["_metadata"] = self.metadata
        path = self.base_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path
    
    def save_input_audio(self, audio_data, filename: str, sample_rate: int = 16000) -> Path:
        """保存输入音频"""
        import soundfile as sf
        path = self.base_dir / filename
        sf.write(str(path), audio_data, sample_rate)
        return path
    
    def save_output_audio(self, audio_data, filename: str, sample_rate: int = 24000) -> Path:
        """保存输出音频"""
        import soundfile as sf
        path = self.base_dir / filename
        sf.write(str(path), audio_data, sample_rate)
        return path
    
    def save_chunk(
        self,
        chunk_idx: int,
        chunk_data: Dict[str, Any],
        audio_data=None,
        sample_rate: int = 24000
    ) -> Path:
        """保存单个 chunk 的输出
        
        Args:
            chunk_idx: chunk 索引
            chunk_data: chunk 数据
            audio_data: 音频数据（可选）
            sample_rate: 音频采样率
        """
        chunk_dir = self.chunks_dir / f"chunk_{chunk_idx:03d}"
        chunk_dir.mkdir(exist_ok=True)
        
        # 保存 JSON
        json_path = chunk_dir / "data.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=2)
        
        # 保存音频
        if audio_data is not None:
            import soundfile as sf
            audio_path = chunk_dir / "audio.wav"
            sf.write(str(audio_path), audio_data, sample_rate)
        
        return chunk_dir
    
    def copy_input_file(self, src_path: Path, dest_filename: str) -> Path:
        """复制输入文件到 case 目录"""
        import shutil
        dest_path = self.base_dir / dest_filename
        shutil.copy(src_path, dest_path)
        return dest_path
    
    def finalize(self, summary: Optional[Dict[str, Any]] = None) -> Path:
        """完成保存，生成 summary.json"""
        summary_data = {
            **self.metadata,
            "summary": summary or {},
            "files": [str(p.relative_to(self.base_dir)) for p in self.base_dir.rglob("*") if p.is_file()]
        }
        path = self.base_dir / "summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        return path


# ============================================================
# Pytest Fixtures
# ============================================================

@pytest.fixture(scope="session")
def model_path() -> str:
    """基础模型路径"""
    return MODEL_PATH


@pytest.fixture(scope="session")
def pt_path() -> Optional[str]:
    """额外权重路径（可选，Duplex 微调用）"""
    return PT_PATH


@pytest.fixture(scope="session")
def ref_audio_path() -> Path:
    """参考音频路径"""
    return REF_AUDIO_PATH


@pytest.fixture(scope="session")
def user_audio_path() -> Path:
    """用户音频路径"""
    return USER_AUDIO_PATH


@pytest.fixture
def case_saver():
    """创建测试用例保存器的工厂函数
    
    使用示例：
        saver = case_saver("simple_chat", "chat")
        saver = case_saver("streaming_text_only", "streaming")
        saver = case_saver("duplex_basic", "duplex")
    """
    def _create_saver(case_name: str, processor_type: str) -> CaseSaver:
        return CaseSaver(case_name, processor_type)
    return _create_saver
