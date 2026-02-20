
# Deps

- text / vision / audio
```shell
pip install \
  "transformers==4.51.0" accelerate \
  "torch<=2.8.0" "torchaudio<=2.8.0" \
  "minicpmo-utils>=1.0.1"
```

- tts / streaming
```shell
pip install \
  "transformers==4.51.0" accelerate \
  "torch<=2.8.0" "torchaudio<=2.8.0" \
  "minicpmo-utils[all]>=1.0.1"
```

- flash attention
```shell
MAX_JOBS=16 FLASH_ATTENTION_FORCE_BUILD=TRUE pip install "flash-attn<=2.8.2,>=2.7.1" --no-build-isolation
```


# Example

## Chat

- text / vision / audio

```python
from transformers import AutoModel
from PIL import Image
import librosa

name_or_path = ""
model = AutoModel.from_pretrained(name_or_path, trust_remote_code=True, attn_implementation="sdpa")
model.bfloat16().eval().cuda()

msgs_text = [{"role": "user", "content": ["Who are you?"]}]

image_path = ...
msgs_vision = [
    {
        "role": "user",
        "content": [
            Image.open(image_path).convert("RGB"),
            "请描述这张图片的内容。",
        ],
    }
]

audio_path = ...
msgs_audio = [
    {
        "role": "user",
        "content": [
            librosa.load(audio_path, sr=16000, mono=True)[0],
            "Determine what is producing the sound in the audio\nA. Owl\nB. Robot\nC. Rooster\nD. Parrot\nFollowing your assessment of the audio's primary features, identify the option that best satisfies the question requirements. Only respond with the letter of the correct option.\n",
        ],
    }
]

msgs_omni = [
    {
        "role": "user",
        "content": [
            "Determine what is producing the sound in the audio\nA. Owl\nB. Robot\nC. Rooster\nD. Parrot\nFollowing your assessment of the audio's primary features, identify the option that best satisfies the question requirements. Only respond with the letter of the correct option.\n",
            Image.open(image_path).convert("RGB"),
            librosa.load(audio_path, sr=16000, mono=True)[0],
        ],
    }
]

msgs = msgs_audio
omni_mode = False  # set True if msgs is omni

res, prompt = model.chat(
    image=None,
    msgs=msgs,
    omni_mode=omni_mode,
    use_tts_template=True,
    enable_thinking=False,
    do_sample=False,
    num_beams=1,
    return_prompt=True
)

print(f"answer: {res}")
```


## Simplex Streaming

```python
from transformers import AutoModel
import librosa
import torch
import soundfile as sf
import numpy as np
import uuid
from minicpmo.utils import get_video_frame_audio_segments

name_or_path = ""
model = AutoModel.from_pretrained(name_or_path, trust_remote_code=True, attn_implementation="sdpa")
model.bfloat16().eval().cuda()
model.init_tts(streaming=True)

# 可选：初始化 TTS 音色
ref_audio_path = ...
ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)

# 从视频中提取帧和音频 chunks
video_path = ...
video_frames, audio_chunks, stacked_frames = get_video_frame_audio_segments(video_path)

# 构建 omni_contents: [frame1, audio1, frame2, audio2, ...]
omni_contents = []
for i in range(len(video_frames)):
    omni_contents.append(video_frames[i])
    omni_contents.append(audio_chunks[i])
    if stacked_frames is not None and stacked_frames[i] is not None:
        omni_contents.append(stacked_frames[i])

session_id = str(uuid.uuid4())  # 每次新 session 使用不同的 id
model.reset_session()
model.init_token2wav_cache(prompt_speech_16k=ref_audio)

# 1. streaming_prefill: 逐个 chunk 填充上下文
audio_indices = [i for i, c in enumerate(omni_contents) if isinstance(c, np.ndarray)]
last_audio_idx = audio_indices[-1] if audio_indices else -1

prompts = []
for idx, content in enumerate(omni_contents):
    is_last_audio = (idx == last_audio_idx)
    prompt = model.streaming_prefill(
        session_id=session_id,
        msgs=[{"role": "user", "content": [content]}],
        omni_mode=True,
        is_last_chunk=is_last_audio,
    )
    prompts.append(prompt)
print(f"prefilled prompt: {''.join(prompts)}")

# 2. streaming_generate
iter_gen = model.streaming_generate(
    session_id=session_id,
    do_sample=False,
    generate_audio=True,
    use_tts_template=True,
    enable_thinking=False,
)

wav_chunks = []
text_chunks = []
for waveform_chunk, new_text in iter_gen:
    wav_chunks.append(waveform_chunk)
    if new_text:
        text_chunks.append(new_text)
        print(f"{new_text}", end="", flush=True)


if wav_chunks:
    generated_waveform = torch.cat(wav_chunks, dim=-1)[0]
    sf.write("output.wav", generated_waveform.cpu().numpy(), samplerate=24000)

print(f"answer: {''.join(text_chunks)}")
```


## Duplex Streaming

```python
import librosa
import numpy as np
import soundfile as sf

from MiniCPMO45.modeling_minicpmo import MiniCPMODuplex
from minicpmo.utils import get_video_frame_audio_segments

# 1. 加载模型
name_or_path = ""
model = MiniCPMODuplex.from_pretrained(
    name_or_path,
    trust_remote_code=True,
    attn_implementation="sdpa",
    audio_pool_step=5,
    audio_chunk_length=1,
    ls_mode="explicit",
    n_timesteps=5,
)

# 2. 准备 session
ref_audio_path = ...
ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)

model.prepare(
    prefix_system_prompt="<|im_start|>system\nYou are a helpful assistant.\n<|audio_start|>",
    suffix_system_prompt="<|audio_end|><|im_end|>",
    ref_audio=ref_audio,
    prompt_wav_path=ref_audio_path,
)

# 3. 从视频中提取帧和音频 chunks
video_path = ...
video_frames, audio_chunks, stacked_frames = get_video_frame_audio_segments(video_path, use_ffmpeg=True)

all_output_audio = []

for chunk_idx in range(len(audio_chunks)):
    audio_chunk = audio_chunks[chunk_idx]
    frame = video_frames[chunk_idx] if chunk_idx < len(video_frames) else None
    frame_list = [frame] if frame is not None else None
    if stacked_frames is not None and chunk_idx < len(stacked_frames) and stacked_frames[chunk_idx] is not None:
        frame_list.append(stacked_frames[chunk_idx])

    # 3.1 streaming_prefill: 填充当前 chunk
    prefill_result = model.streaming_prefill(
        audio_waveform=audio_chunk,
        frame_list=frame_list,
        # text_list=["请仔细理解"]
    )
    if not prefill_result.get("success", False):
        continue

    # 3.2 streaming_generate: 模型决定 listen 或 speak
    result = model.streaming_generate(
        prompt_wav_path=ref_audio_path,
        max_new_speak_tokens_per_chunk=20,
        decode_mode="greedy",
    )

    # 处理结果
    if result["is_listen"]:
        print(f"[chunk {chunk_idx}] listen...")
    else:
        print(f"[chunk {chunk_idx}] speak: {result['text']}")
        if result["audio_waveform"] is not None:
            all_output_audio.append(result["audio_waveform"])

    if result["end_of_turn"]:
        break

# 4. 保存输出音频
if all_output_audio:
    output_audio = np.concatenate(all_output_audio)
    sf.write("duplex_output.wav", (output_audio * 32768).astype(np.int16), 24000, subtype="PCM_16")

print(f"完整回复: {model.get_generated_text()}")
```


#### 双工滑窗

在模型初始化时配置，并在 prepare 时传入合适的 prompt

- 不带 context 滑窗
```python
from MiniCPMO45.modeling_minicpmo import MiniCPMODuplex

model = MiniCPMODuplex(
    ...,
    sliding_window_mode="basic",  # off-无滑窗, basic-不带 context 的滑窗, context-带 context 的滑窗
    # basic 参数
    basic_window_high_tokens=8000,
    basic_window_low_tokens=6000,
)

...

model.prepare(
    prefix_system_prompt="<|im_start|>system\nStreaming Duplex Conversation! You are a helpful assistant.\n<|audio_start|>",
    suffix_system_prompt="<|audio_end|><|im_end|>",
    ref_audio=...,
    prompt_wav_path=...,
)
```

- 带 context 滑窗
```python
from MiniCPMO45.modeling_minicpmo import MiniCPMODuplex

model = MiniCPMODuplex(
    ...,
    sliding_window_mode="context",  # off-无滑窗, basic-不带 context 的滑窗, context-带 context 的滑窗
    # context 参数
    context_previous_max_tokens=500,  # 滑入的模型生成内容最多保留 500 token, 右侧滑入左侧截断
    context_max_units=24  # 每 24 个 unit 触发一次滑窗操作
)

...

model.prepare(
    prefix_system_prompt="<|im_start|>system\nStreaming Vision Caption.\nTitle: ???",
    suffix_system_prompt="<|im_end|>",
    ref_audio=...,
    prompt_wav_path=...,
    context_previous_marker="\n\nprevious: ",  # 可选，滑窗时用于标记历史内容
)
```

#### 强制 Listen

在每个 session 的前 N 次 `streaming_generate` 调用中强制生成 listen token，跳过模型 decode，让模型先"听"一段时间再决定是否回复。

**初始化配置**：

```python
from MiniCPMO45.modeling_minicpmo import MiniCPMODuplex

model = MiniCPMODuplex.from_pretrained(
    ...,
    force_listen_count=3,  # 每个 session 前 3 次 streaming_generate 强制 listen（0=禁用）
)
```

注意：计数器会在每次调用 `prepare()` 时重置，每个 session 独立计数。

#### 单工流式 VAD 抢跑

在 VAD 检测到用户停顿时提前调用 `streaming_generate` 以减少响应延迟, 支持回滚并继续 prefill。

**关键参数**：
- `enable_speculative_snapshot`：控制是否在 `streaming_generate` 开始时保存快照
  - `True`：启用抢跑回滚支持
  - `False`：（默认）禁用快照保存

```python
model = ...

session_id = "user_123"
omni_segments = [...]

snapshot = None

# 1. 流式 prefill 用户输入（image + audio 交替）
for idx, content in enumerate(omni_segments):
    # snapshot
    snapshot = model.save_speculative_snapshot()
     
    model.streaming_prefill(
        session_id=session_id,
        msgs=[{"role": "user", "content": [content]}],
        omni_mode=True,
        is_last_chunk=False,
    )

# 2. VAD 尝试抢跑
gen = model.streaming_generate(
    ...,
)

# 3. 抢跑失败 - 回滚
model.restore_speculative_snapshot(snapshot)
...
```
