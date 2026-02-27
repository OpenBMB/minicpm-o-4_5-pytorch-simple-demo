import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from moviepy.video.io.VideoFileClip import VideoFileClip
from vad import vad as vad_utils

# file = '/cache/ninghongbo/code/data/20250730-162824.mp4'
# file = '/cache/ninghongbo/code/data/流畅-视频-短-热点3.mp4'
file = '/cache/ninghongbo/code/data/20250730163453_rec_.mp4'


if file.endswith('.wav'):
    samplerate, ori_audio = wavfile.read(file)
elif file.endswith('.mp4'):
    video = VideoFileClip(file)
    samplerate = video.audio.fps
    ori_audio = video.audio.to_soundarray(fps=samplerate)
    if ori_audio.ndim == 2:
        ori_audio = ori_audio.mean(axis=1)
    ori_audio = np.clip(ori_audio * 32768, -32768, 32767).astype(np.int16)
    video.close()


vad_options = vad_utils.VadOptions(
)

print(f'Sample rate: {samplerate}')
print(ori_audio.shape[0] / samplerate)

window_size_samples = int(samplerate)
samples_per_loop = int(samplerate * 10 / 1000)


# time_axis = []
# dur_vad_values = []
# for s in range(0, ori_audio.shape[0] - window_size_samples, samples_per_loop):
#     res = vad_utils.run_vad(ori_audio[s: s+window_size_samples], samplerate, vad_options)
#     dur_vad = res["dur_vad"]
    
#     time_axis.append(s / samplerate)
#     dur_vad_values.append(dur_vad)


# # 绘图
# plt.figure()
# plt.plot(time_axis, dur_vad_values, label='dur_vad', color='blue')
# plt.xlabel('Time')
# plt.ylabel('dur_vad')
# plt.title('dur_vad over time')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig(f'/cache/ninghongbo/code/MiniCPM-o-demo-web/server/vad/dur_vad_plot_{window_size_samples}.png')


full_time_axis = np.arange(0, ori_audio.shape[0] / samplerate * 16000)
vad_mask = np.zeros_like(full_time_axis)

res = vad_utils.run_vad_streaming(ori_audio, samplerate, vad_options, chunk_size_ms=32)
# res = vad_utils.run_vad(ori_audio, samplerate, vad_options)
speech_chunks, chunk_probs = res["speech_chunks"], res["chunk_probs"]
audio = res["audio"]

for chunk in speech_chunks:
    start_frame = chunk["start"]
    end_frame = chunk["end"]

    # 标记有语音的时间段为 1
    vad_mask[start_frame:end_frame] = 1

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6), gridspec_kw={'height_ratios': [1, 2]})

# 可视化语音活动
ax1.plot(full_time_axis / 16000, vad_mask, drawstyle='steps-post', label="finally result")
ax1.plot([512 * t / 16000 for t in range(0, len(chunk_probs))], chunk_probs, linestyle='--', marker='*', alpha=0.3, label="speech probability")
ax1.set_xlabel("Time (sample)")
ax1.set_ylabel("Speech (1=Yes, 0=No)")
ax1.set_title("Voice Activity Over Time")
ax1.grid(True)
ax1.set_ylim(-0.1, 1.1)
ax1.legend(loc="upper right")

ax2.plot(np.arange(len(audio)) / 16000, audio, color='gray', linewidth=0.5)
ax2.set_xlabel("Time (sample)")
ax2.set_ylabel("Amplitude")
ax2.set_title("Audio Waveform")
ax2.grid(True)

plt.tight_layout()
plt.savefig(f'/cache/ninghongbo/code/data/dur_vad_plot__.png')


# 按speech_chunks分割并保存每个语音段为wav文件
import os
import shutil
shutil.rmtree('/cache/ninghongbo/code/data/test', ignore_errors=True)
os.makedirs('/cache/ninghongbo/code/data/test', exist_ok=True)
audio = (audio * 32767).astype(np.int16)
wavfile.write('/cache/ninghongbo/code/data/test/summary.wav',
      16000,
      vad_utils.collect_chunks(audio, speech_chunks)
)
for i, chunk in enumerate(speech_chunks):
    segment = audio[chunk["start"]:chunk["end"]]
    out_path = f"/cache/ninghongbo/code/data/test/segment_{i+1}.wav"
    wavfile.write(out_path, 16000, segment)
    print(f"Saved: {out_path}")


