"""参考音频管理

管理参考音频的上传、存储、查询、删除。
不使用数据库，使用内存 dict + JSON 文件持久化。

存储结构：
    data/
    └── assets/
        └── ref_audio/
            ├── registry.json      # 注册表索引
            ├── {uuid}.wav         # 上传的音频文件
            └── ...
"""

import io
import os
import json
import uuid
import base64
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("gateway.ref_audio")


# ============ 数据模型 ============

class RefAudioInfo(BaseModel):
    """参考音频信息"""
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    filename: str  # 存储文件名（不含路径）
    source_type: str = "upload"
    duration_ms: Optional[int] = None
    sample_rate: int = 16000
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class RefAudioListResponse(BaseModel):
    """参考音频列表响应"""
    total: int
    ref_audios: List[RefAudioInfo]


class UploadRefAudioRequest(BaseModel):
    """上传参考音频请求"""
    name: str = Field(..., description="显示名称")
    audio_base64: str = Field(..., description="Base64 编码的 WAV 音频")


class RefAudioResponse(BaseModel):
    """参考音频操作响应"""
    success: bool
    id: Optional[str] = None
    name: Optional[str] = None
    message: str = ""


# ============ 注册表 ============

class RefAudioRegistry:
    """参考音频注册表

    管理参考音频的元数据和文件存储。
    """

    def __init__(self, storage_dir: str):
        """
        Args:
            storage_dir: 存储目录路径（如 data/assets/ref_audio/）
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.storage_dir / "registry.json"

        # 内存索引
        self._registry: Dict[str, RefAudioInfo] = {}

        # 加载已有注册表
        self._load_registry()

    def _load_registry(self) -> None:
        """从 JSON 文件加载注册表"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    info = RefAudioInfo(**item)
                    self._registry[info.id] = info
                logger.info(f"Loaded {len(self._registry)} ref audios from registry")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
        else:
            logger.info("No existing registry, starting fresh")

    def _save_registry(self) -> None:
        """保存注册表到 JSON 文件"""
        data = [info.model_dump() for info in self._registry.values()]
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ========== CRUD ==========

    def upload(self, name: str, audio_base64: str) -> RefAudioInfo:
        """上传参考音频（自动校验 + 归一化为 16kHz mono 16-bit WAV）

        Args:
            name: 显示名称
            audio_base64: Base64 编码的音频（任意格式，会自动转换）

        Returns:
            RefAudioInfo

        Raises:
            ValueError: 无法解析为音频文件
        """
        import librosa
        import soundfile as sf

        audio_bytes = base64.b64decode(audio_base64)

        try:
            audio_data, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        except Exception as e:
            raise ValueError(f"无法解析音频文件: {e}")

        ref_id = str(uuid.uuid4())
        filename = f"{ref_id}.wav"
        filepath = self.storage_dir / filename

        sf.write(str(filepath), audio_data, 16000, subtype="PCM_16")

        duration_ms = int(len(audio_data) / 16000 * 1000)

        ref_info = RefAudioInfo(
            id=ref_id,
            name=name,
            filename=filename,
            duration_ms=duration_ms,
        )
        self._registry[ref_id] = ref_info
        self._save_registry()

        logger.info(f"Uploaded ref audio: {name} ({ref_id}), {duration_ms}ms, 16kHz mono PCM_16")
        return ref_info

    def get(self, ref_id: str) -> Optional[RefAudioInfo]:
        """获取参考音频信息"""
        return self._registry.get(ref_id)

    def get_file_path(self, ref_id: str) -> Optional[str]:
        """获取参考音频的文件路径

        Args:
            ref_id: 参考音频 ID

        Returns:
            文件路径（绝对路径），None 表示不存在
        """
        info = self._registry.get(ref_id)
        if info is None:
            return None

        return str(self.storage_dir / info.filename)

    def get_base64(self, ref_id: str) -> Optional[str]:
        """获取参考音频的 Base64 数据

        Args:
            ref_id: 参考音频 ID

        Returns:
            Base64 编码的音频数据，None 表示不存在
        """
        filepath = self.get_file_path(ref_id)
        if filepath is None or not os.path.exists(filepath):
            return None

        with open(filepath, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def list_all(self) -> List[RefAudioInfo]:
        """列出所有参考音频"""
        return list(self._registry.values())

    def delete(self, ref_id: str) -> bool:
        """删除参考音频

        Args:
            ref_id: 参考音频 ID

        Returns:
            是否删除成功
        """
        info = self._registry.get(ref_id)
        if info is None:
            return False

        filepath = self.storage_dir / info.filename
        if filepath.exists():
            filepath.unlink()

        del self._registry[ref_id]
        self._save_registry()

        logger.info(f"Deleted ref audio: {info.name} ({ref_id})")
        return True

    def exists(self, ref_id: str) -> bool:
        """检查参考音频是否存在"""
        return ref_id in self._registry

    @property
    def count(self) -> int:
        return len(self._registry)
