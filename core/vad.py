"""流式 VAD（Voice Activity Detection）

基于 SileroVAD 的流式语音活动检测器，用于 Half-Duplex Audio 模式。
逐 chunk 喂入音频，检测用户语音的开始和结束。

当检测到用户说完话（足够长的静音），返回累积的语音段。
"""

import logging
from typing import Optional, NamedTuple

import numpy as np

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000


class VadOptions(NamedTuple):
    threshold: float = 0.8
    min_speech_duration_ms: int = 128
    min_silence_duration_ms: int = 800
    window_size_samples: int = 1024
    speech_pad_ms: int = 30


class StreamingVAD:
    """流式 VAD：逐 chunk 喂入，检测语音段落

    使用 SileroVAD ONNX 模型，在每个窗口上计算语音概率。
    当检测到足够长的静音后（说明用户说完），返回完整的语音段。

    用法:
        vad = StreamingVAD()
        for audio_chunk in audio_stream:
            speech = vad.feed(audio_chunk)
            if speech is not None:
                # 用户说完了，speech 是完整的语音 ndarray
                process(speech)
    """

    def __init__(self, options: Optional[VadOptions] = None):
        from public.vad.vad_utils import get_vad_model

        self.options = options or VadOptions()
        self.model = get_vad_model()
        self.state = self.model.get_initial_state(batch_size=1)

        self._threshold = self.options.threshold
        self._neg_threshold = self._threshold - 0.15
        self._window = self.options.window_size_samples
        self._min_speech_samples = int(SAMPLING_RATE * self.options.min_speech_duration_ms / 1000)
        self._min_silence_samples = int(SAMPLING_RATE * self.options.min_silence_duration_ms / 1000)
        self._speech_pad_samples = int(SAMPLING_RATE * self.options.speech_pad_ms / 1000)

        self._triggered = False
        self._speech_buffer: list[np.ndarray] = []
        self._speech_start_sample = 0
        self._current_sample = 0
        self._silence_start_sample = 0

        self._leftover = np.array([], dtype=np.float32)

    @property
    def is_speaking(self) -> bool:
        return self._triggered

    def reset(self) -> None:
        """重置 VAD 状态"""
        self.state = self.model.get_initial_state(batch_size=1)
        self._triggered = False
        self._speech_buffer = []
        self._speech_start_sample = 0
        self._current_sample = 0
        self._silence_start_sample = 0
        self._leftover = np.array([], dtype=np.float32)

    def feed(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """喂入一段音频 chunk，返回完整语音段或 None

        Args:
            audio_chunk: float32 音频数据，16kHz

        Returns:
            当检测到语音结束时返回完整的语音段 ndarray，否则 None
        """
        audio = np.concatenate([self._leftover, audio_chunk]) if self._leftover.size > 0 else audio_chunk
        self._leftover = np.array([], dtype=np.float32)

        offset = 0
        result = None

        while offset + self._window <= len(audio):
            window_data = audio[offset:offset + self._window]
            prob, self.state = self.model(window_data, self.state, SAMPLING_RATE)
            speech_prob = float(prob.squeeze())

            if speech_prob >= self._threshold and not self._triggered:
                self._triggered = True
                self._speech_start_sample = self._current_sample
                self._speech_buffer = []
                self._silence_start_sample = 0

            if self._triggered:
                self._speech_buffer.append(window_data)

            if speech_prob < self._neg_threshold and self._triggered:
                if self._silence_start_sample == 0:
                    self._silence_start_sample = self._current_sample

                silence_duration = self._current_sample - self._silence_start_sample + self._window
                if silence_duration >= self._min_silence_samples:
                    speech_duration = self._current_sample - self._speech_start_sample
                    if speech_duration >= self._min_speech_samples:
                        result = np.concatenate(self._speech_buffer)
                    self._triggered = False
                    self._speech_buffer = []
                    self._silence_start_sample = 0
            else:
                if self._triggered:
                    self._silence_start_sample = 0

            offset += self._window
            self._current_sample += self._window

        if offset < len(audio):
            self._leftover = audio[offset:]

        return result

    def flush(self) -> Optional[np.ndarray]:
        """强制结束当前语音段（用于 session 结束时）"""
        if self._triggered and self._speech_buffer:
            speech_duration = self._current_sample - self._speech_start_sample
            if speech_duration >= self._min_speech_samples:
                result = np.concatenate(self._speech_buffer)
                self._triggered = False
                self._speech_buffer = []
                return result
        self._triggered = False
        self._speech_buffer = []
        return None
