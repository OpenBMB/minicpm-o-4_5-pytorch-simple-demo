"""VAD（Voice Activity Detection）模块

包含：
- SileroVADModel: ONNX 模型封装
- get_vad_model(): 单例模型加载
- get_speech_timestamps(): 批量音频语音段检测
- collect_chunks(): 音频段拼接
- run_vad(): 批量 VAD 处理入口
- StreamingVAD: 流式 VAD（逐 chunk 喂入，用于 Half-Duplex Audio）
- VadOptions / StreamingVadOptions: 配置
"""

import functools
import logging
import os
import time
import traceback
import warnings
from typing import List, NamedTuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000


# ============================================================
# ONNX Model
# ============================================================

class SileroVADModel:
    def __init__(self, path):
        try:
            import onnxruntime
        except ImportError as e:
            raise RuntimeError(
                "Applying the VAD filter requires the onnxruntime package"
            ) from e

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4

        self.session = onnxruntime.InferenceSession(
            path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )

    def get_initial_state(self, batch_size: int):
        h = np.zeros((2, batch_size, 64), dtype=np.float32)
        c = np.zeros((2, batch_size, 64), dtype=np.float32)
        return h, c

    def __call__(self, x, state, sr: int):
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)
        if len(x.shape) > 2:
            raise ValueError(
                f"Too many dimensions for input audio chunk {len(x.shape)}"
            )
        if sr / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        h, c = state
        ort_inputs = {
            "input": x,
            "h": h,
            "c": c,
            "sr": np.array(sr, dtype="int64"),
        }
        out, h, c = self.session.run(None, ort_inputs)
        state = (h, c)
        return out, state


@functools.lru_cache
def get_vad_model():
    """Returns the VAD model instance."""
    path = os.path.join(os.path.dirname(__file__), "..", "assets", "vad_model", "silero_vad.onnx")
    return SileroVADModel(path)


# ============================================================
# Batch VAD (offline, process entire audio)
# ============================================================

class BatchVadOptions(NamedTuple):
    """Batch VAD options (for offline processing)."""
    threshold: float = 0.7
    min_speech_duration_ms: int = 128
    max_speech_duration_s: float = float("inf")
    min_silence_duration_ms: int = 500
    window_size_samples: int = 1024
    speech_pad_ms: int = 30


def get_speech_timestamps(
    audio: np.ndarray,
    vad_options: Optional[BatchVadOptions] = None,
    **kwargs,
) -> List[dict]:
    """Split long audio into speech chunks using silero VAD."""
    if vad_options is None:
        vad_options = BatchVadOptions(**kwargs)

    threshold = vad_options.threshold
    min_speech_duration_ms = vad_options.min_speech_duration_ms
    max_speech_duration_s = vad_options.max_speech_duration_s
    min_silence_duration_ms = vad_options.min_silence_duration_ms
    window_size_samples = vad_options.window_size_samples
    speech_pad_ms = vad_options.speech_pad_ms

    if window_size_samples not in [512, 1024, 1536]:
        warnings.warn(
            "Unusual window_size_samples! Supported: [512, 1024, 1536] for 16000 sampling_rate"
        )

    sampling_rate = SAMPLING_RATE
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = (
        sampling_rate * max_speech_duration_s
        - window_size_samples
        - 2 * speech_pad_samples
    )
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate * 98 / 1000

    audio_length_samples = len(audio)

    model = get_vad_model()
    state = model.get_initial_state(batch_size=1)

    speech_probs = []
    for current_start_sample in range(0, audio_length_samples, window_size_samples):
        chunk = audio[current_start_sample: current_start_sample + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = np.pad(chunk, (0, int(window_size_samples - len(chunk))))
        speech_prob, state = model(chunk, state, sampling_rate)
        speech_probs.append(speech_prob)

    triggered = False
    speeches = []
    current_speech = {}
    neg_threshold = threshold - 0.15
    temp_end = 0
    prev_end = next_start = 0

    for i, speech_prob in enumerate(speech_probs):
        if (speech_prob >= threshold) and temp_end:
            temp_end = 0
            if next_start < prev_end:
                next_start = window_size_samples * i

        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech["start"] = window_size_samples * i
            continue

        if (
            triggered
            and (window_size_samples * i) - current_speech["start"] > max_speech_samples
        ):
            if prev_end:
                current_speech["end"] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                if next_start < prev_end:
                    triggered = False
                else:
                    current_speech["start"] = next_start
                prev_end = next_start = temp_end = 0
            else:
                current_speech["end"] = window_size_samples * i
                speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue

        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = window_size_samples * i
            if (window_size_samples * i) - temp_end > min_silence_samples_at_max_speech:
                prev_end = temp_end
            if (window_size_samples * i) - temp_end < min_silence_samples:
                continue
            else:
                current_speech["end"] = temp_end
                if (current_speech["end"] - current_speech["start"]) > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue

    if (
        current_speech
        and (audio_length_samples - current_speech["start"]) > min_speech_samples
    ):
        current_speech["end"] = audio_length_samples
        speeches.append(current_speech)

    for i, speech in enumerate(speeches):
        if i == 0:
            speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i + 1]["start"] - speech["end"]
            if silence_duration < 2 * speech_pad_samples:
                speech["end"] += int(silence_duration // 2)
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - silence_duration // 2)
                )
            else:
                speech["end"] = int(min(audio_length_samples, speech["end"] + speech_pad_samples))
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - speech_pad_samples)
                )
        else:
            speech["end"] = int(min(audio_length_samples, speech["end"] + speech_pad_samples))

    return speeches


def collect_chunks(audio: np.ndarray, chunks: List[dict]) -> np.ndarray:
    """Collects and concatenates audio chunks."""
    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate([audio[chunk["start"]: chunk["end"]] for chunk in chunks])


def run_vad(ori_audio, sr, vad_options=None):
    """Batch VAD processing entry point."""
    _st = time.time()
    try:
        import librosa
        audio = np.frombuffer(ori_audio, dtype=np.int16)
        audio = audio.astype(np.float32) / 32768.0
        sampling_rate = SAMPLING_RATE
        if sr != sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)
        if vad_options is None:
            vad_options = BatchVadOptions()
        speech_chunks = get_speech_timestamps(audio, vad_options=vad_options)
        audio = collect_chunks(audio, speech_chunks)
        duration_after_vad = audio.shape[0] / sampling_rate
        return duration_after_vad, ori_audio, round(time.time() - _st, 4)
    except Exception as e:
        msg = f"[asr vad error] audio_len: {len(ori_audio)/(sr*2):.3f} s, trace: {traceback.format_exc()}"
        print(msg)
        return -1, ori_audio, round(time.time() - _st, 4)


# ============================================================
# Streaming VAD (online, chunk-by-chunk for Half-Duplex Audio)
# ============================================================

class StreamingVadOptions(NamedTuple):
    """Streaming VAD options (for real-time half-duplex)."""
    threshold: float = 0.8
    min_speech_duration_ms: int = 128
    min_silence_duration_ms: int = 800
    window_size_samples: int = 1024
    speech_pad_ms: int = 30


class StreamingVAD:
    """Streaming VAD: feed audio chunks, detect speech segments.

    Usage:
        vad = StreamingVAD()
        for audio_chunk in audio_stream:
            speech = vad.feed(audio_chunk)
            if speech is not None:
                process(speech)
    """

    def __init__(self, options: Optional[StreamingVadOptions] = None):
        self.options = options or StreamingVadOptions()
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
        self.state = self.model.get_initial_state(batch_size=1)
        self._triggered = False
        self._speech_buffer = []
        self._speech_start_sample = 0
        self._current_sample = 0
        self._silence_start_sample = 0
        self._leftover = np.array([], dtype=np.float32)

    def feed(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """Feed an audio chunk (float32, 16kHz). Returns speech segment or None."""
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
        """Force-end current speech segment (for session end)."""
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
