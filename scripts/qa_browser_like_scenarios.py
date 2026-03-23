#!/usr/bin/env python3
"""
按「真实浏览器」协议压测 Gateway，用于复现 / 回归测试员反馈的问题。

**默认与测试员打开的网页对齐**；Half-duplex **默认开 TTS**（与页面一致），加快跑可加 `--half-duplex-no-tts`。`test_case` 下多段 WAV 默认 **合并成一条** 再发送（与「多段语言素材合成一问」一致）。C++ 每秒切片流：`--half-duplex-cpp-one-second-stream`。

- Turn-based：`/ws/chat` 首帧 JSON 对齐 `static/turnbased.html` 的 `_buildChatWSBody`（纯文本、关 TTS 时：无 `tts` 字段，`streaming`/`generation`/`image`/`omni_mode` 与页面一致）。
- Half-duplex：`prepare` **默认 TTS 开**；系统提示支持 **按用户语言作答**；用户音频默认将 `audio_test_case`（或回退目录）中 **从 `--half-duplex-merge-start` 起连续 `--half-duplex-merge-count` 段** 拼成一条 float32 流（段间短静音），再按 0.5s/块发送；合并结果写入 `data/qa_cache/half_duplex_merged_probe.wav` 便于复现。
- Duplex：`replay_duplex_session.py` 走 `/ws/duplex/{id}`，与 `duplex-session.js` 一致；输入为 C++ `test_case` 按秒切的 WAV（浏览器实时 chunk 大小还取决于采集/文件源，见 `--half-duplex-cpp-one-second-stream`）。

局限（脚本内说明）：
- Half-duplex 使用 Silero VAD：**正弦/白噪声等合成音频通常不会判成语音**。默认 WAV 解析顺序（`--llamacpp-root`）：`audio_test_case/*.wav`（与 omni-cli 默认 audio 测例一致）→ `duplex_omni_test_case` → `omni_test_case` → 本仓库 `assets/ref_audio/ref_minicpm_signature.wav`；均在末尾自动拼 1.5s 静音以便 VAD 收尾。自定义请用 `--half-duplex-wav`。
- 「中文重复用户问题」「陈述句不答」「教我做炒面只回好的」等**语义**问题，自动化只能做启发式检查；
  精确复现需测试员提供 **16kHz mono WAV**（或浏览器录屏 + 导出），用 --half-duplex-wav / 自建 session 回放。
- 句尾 / 尾音截断：可做「流式文本 vs done 文本」一致性、输出 WAV 尾部能量等**弱信号**，不能替代人耳。

用法（服务已启动）：
  PYTHONPATH=. .venv/base/bin/python scripts/qa_browser_like_scenarios.py
  PYTHONPATH=. .venv/base/bin/python scripts/qa_browser_like_scenarios.py --gateway http://127.0.0.1:8020 --skip-duplex-replay
  PYTHONPATH=. .venv/base/bin/python scripts/qa_browser_like_scenarios.py --half-duplex-wav /path/to/user_16k.wav
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
VENV_PY = PROJECT / ".venv" / "base" / "bin" / "python"

try:
    import soundfile as sf
except ImportError:
    sf = None  # type: ignore

try:
    import websockets
except ImportError:
    print("需要: pip install websockets", file=sys.stderr)
    raise


def _http_to_ws_base(gateway: str) -> str:
    g = gateway.rstrip("/")
    if g.startswith("https://"):
        return "wss://" + g[len("https://") :]
    if g.startswith("http://"):
        return "ws://" + g[len("http://") :]
    return g


# Half-duplex 系统提示：覆盖测试员关心的「语言一致、勿截断、勿复读」
HDX_SYSTEM_MULTILANG = (
    "你是多语言语音助手。请根据用户实际使用的语言（中文、英文等）用同一语言作答；"
    "表述完整、不要无故截断，不要机械重复用户原话。"
)


def _test_case_wav_triples() -> List[tuple[str, str, str]]:
    """(子目录, 文件名前缀, 说明) — 按优先级尝试合并/单段。"""
    return [
        ("audio_test_case", "audio_test_case", "C++ omni-cli 默认 audio 测例"),
        ("duplex_omni_test_case", "duplex_omni_test_case", "duplex 测例"),
        ("omni_test_case", "omni_test_case", "omni 图+音测例"),
    ]


def _resolve_default_half_duplex_wav(llamacpp_root: Path, asset_index: int) -> Path:
    """与 tools/omni 下 C++ 测试数据对齐：audio_test_case → duplex_omni_test_case → omni_test_case → demo ref。"""
    base = (llamacpp_root / "tools/omni/assets/test_case").resolve()
    idx = asset_index
    candidates = [
        base / sub / f"{prefix}_{idx:04d}.wav"
        for sub, prefix, _ in _test_case_wav_triples()
    ]
    for p in candidates:
        if p.is_file():
            return p
    ref = (PROJECT / "assets/ref_audio/ref_minicpm_signature.wav").resolve()
    if ref.is_file():
        return ref
    raise FileNotFoundError(
        f"未找到 half-duplex 探针 WAV（已查 {base} 下 audio/duplex/omni 测例及 {ref}）。"
        "请指定 --llamacpp-root、安装 omni test_case 资源，或使用 --half-duplex-wav。"
    )


def _half_duplex_silence_tail(min_silence_ms: int = 800) -> np.ndarray:
    """句末静音长度 ≥ half-duplex 页默认 `vadMinSilence`（800ms），流式 VAD 才会收尾。"""
    # 略长于 800ms，避免边界抖动
    return np.zeros(int(16000 * max(1.2, (min_silence_ms / 1000.0) * 1.1)), dtype=np.float32)


def _load_wav_16k_mono(path: Path) -> np.ndarray:
    if sf is None:
        raise RuntimeError("读取 WAV 需要 soundfile: pip install soundfile")
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32)
    if sr != 16000:
        old_x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        new_len = max(1, int(round(len(audio) * 16000 / sr)))
        new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        audio = np.interp(new_x, old_x, audio.astype(np.float64)).astype(np.float32)
    return audio


def _merge_test_case_wavs(
    llamacpp_root: Path,
    start: int,
    count: int,
    gap_ms: float = 200.0,
) -> tuple[np.ndarray, str, List[str]]:
    """
    将 test_case 中连续多段 WAV 合并为 16k mono float32（段间 gap_ms 静音）。
    返回 (audio, 描述标签, 源文件路径列表)。
    """
    if count < 1:
        raise ValueError("merge_count 至少为 1")
    base = (llamacpp_root / "tools/omni/assets/test_case").resolve()
    gap = np.zeros(max(0, int(16000 * gap_ms / 1000.0)), dtype=np.float32)

    for sub, prefix, _desc in _test_case_wav_triples():
        paths: List[Path] = []
        for i in range(count):
            p = base / sub / f"{prefix}_{start + i:04d}.wav"
            if not p.is_file():
                paths = []
                break
            paths.append(p)
        if len(paths) != count:
            continue
        parts: List[np.ndarray] = []
        for i, p in enumerate(paths):
            parts.append(_load_wav_16k_mono(p))
            if i < len(paths) - 1:
                parts.append(gap.copy())
        audio = np.concatenate(parts) if parts else np.array([], dtype=np.float32)
        label = f"merged:{sub}:{start:04d}-{start + count - 1:04d}"
        return audio, label, [str(p.resolve()) for p in paths]

    if count == 1:
        p = _resolve_default_half_duplex_wav(llamacpp_root, start)
        return _load_wav_16k_mono(p), str(p.resolve()), [str(p.resolve())]

    raise FileNotFoundError(
        f"无法在 {base} 找到连续 {count} 段 WAV（起始编号 {start:04d}），"
        "请减小 --half-duplex-merge-count 或指定 --half-duplex-wav。"
    )


def _chunks_f32pcm_b64(audio: np.ndarray, chunk_samples: int) -> List[str]:
    out: List[str] = []
    for i in range(0, len(audio), chunk_samples):
        sl = audio[i : i + chunk_samples]
        if sl.size == 0:
            continue
        out.append(base64.b64encode(sl.tobytes()).decode("ascii"))
    return out


async def scenario_turnbased_stream_vs_done(gateway_http: str, timeout_s: float) -> Dict[str, Any]:
    """
    对应测试员反馈：turn_based「句尾截断」。
    检查：流式 text_delta 拼接结果是否与最终 done.text 一致（不一致常表示客户端展示或流式合并 bug）。
    """
    uri = f"{_http_to_ws_base(gateway_http)}/ws/chat"
    # 与 turnbased.html::_buildChatWSBody 一致：纯文本、无视频 → omni_mode false，image.max_slice_nums=1（未勾选 HD）
    payload: Dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": "请用中文写一段 80～120 字的说明，介绍春天公园里的花草与游人，最后一句要以句号结尾。",
            }
        ],
        "streaming": True,
        "generation": {
            "max_new_tokens": 256,
            "do_sample": True,
            "length_penalty": 1.1,
        },
        "image": {"max_slice_nums": 1},
        "omni_mode": False,
    }
    stream_parts: List[str] = []
    done_text = ""
    done_tokens: Optional[int] = None
    async with websockets.connect(uri, max_size=50_000_000, open_timeout=30) as ws:
        await ws.send(json.dumps(payload, ensure_ascii=False))
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            raw = await asyncio.wait_for(ws.recv(), timeout=max(5.0, deadline - time.monotonic()))
            msg = json.loads(raw)
            t = msg.get("type")
            if t == "chunk" and msg.get("text_delta"):
                stream_parts.append(msg["text_delta"])
            elif t == "done":
                done_text = (msg.get("text") or "").strip()
                done_tokens = msg.get("generated_tokens")
                break
            elif t == "error":
                return {"ok": False, "error": msg, "scenario": "turnbased_stream_vs_done"}
            elif t == "prefill_done":
                continue
            elif t == "heartbeat":
                continue
        else:
            return {"ok": False, "error": "timeout", "scenario": "turnbased_stream_vs_done"}

    joined = "".join(stream_parts).strip()
    mismatch = joined != done_text
    tail_ok = done_text.endswith(("。", ".", "！", "？", "!", "?"))
    return {
        "ok": not mismatch,
        "scenario": "turnbased_stream_vs_done",
        "stream_len": len(joined),
        "done_len": len(done_text),
        "stream_done_match": not mismatch,
        "done_ends_with_sentence_punct": tail_ok,
        "generated_tokens": done_tokens,
        "preview_done": done_text[:120] + ("…" if len(done_text) > 120 else ""),
    }


async def scenario_half_duplex_like_browser(
    gateway_http: str,
    session_id: str,
    audio: np.ndarray,
    *,
    chunk_samples: int,
    chunk_interval_s: float,
    tts_enabled: bool,
    max_new_tokens: int,
    timeout_s: float,
    system_text: str = HDX_SYSTEM_MULTILANG,
) -> Dict[str, Any]:
    """
    Half-duplex：默认与 static/half-duplex/half-duplex-app.js 中 DEFAULTS + getSettings() 一致。
    """
    uri = f"{_http_to_ws_base(gateway_http)}/ws/half_duplex/{session_id}"
    system_content: List[Dict[str, Any]] = [{"type": "text", "text": system_text}]
    # half-duplex-app.js DEFAULTS / getSettings()
    config = {
        "vad": {
            "threshold": 0.8,
            "min_speech_duration_ms": 128,
            "min_silence_duration_ms": 800,
            "speech_pad_ms": 30,
        },
        "generation": {
            "max_new_tokens": max_new_tokens,
            "length_penalty": 1.1,
            "temperature": 0.7,
        },
        "tts": {"enabled": tts_enabled},
        "session": {"timeout_s": 300},
    }

    assistant_text_parts: List[str] = []
    prepare_payload = json.dumps(
        {"type": "prepare", "system_content": system_content, "config": config},
        ensure_ascii=False,
    )

    async with websockets.connect(uri, max_size=50_000_000, open_timeout=60) as ws:
        # 与 half-duplex-app.js 一致：onopen 立刻发 prepare（Gateway 先排队再接通 Worker 时会缓存该帧）
        await ws.send(prepare_payload)

        deadline = time.monotonic() + 120
        prepared = False
        while time.monotonic() < deadline:
            raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
            msg = json.loads(raw)
            typ = msg.get("type")
            if typ == "prepared":
                prepared = True
                break
            if typ == "error":
                return {"ok": False, "error": msg, "scenario": "half_duplex"}
            # queued / queue_update / queue_done / vad_state 等在排队或异常时可能出现，忽略直至 prepared
        if not prepared:
            return {"ok": False, "error": "no prepared", "scenario": "half_duplex"}

        await asyncio.sleep(0.55)  # 对齐 worker INITIAL_GUARD_S 0.5s
        for b64 in _chunks_f32pcm_b64(audio, chunk_samples):
            await ws.send(json.dumps({"type": "audio_chunk", "audio_base64": b64}))
            if chunk_interval_s > 0:
                await asyncio.sleep(chunk_interval_s)

        # 收完一轮 turn_done 或超时
        turn_done = False
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            msg = json.loads(raw)
            typ = msg.get("type")
            if typ == "chunk" and msg.get("text_delta"):
                assistant_text_parts.append(msg["text_delta"])
            elif typ == "turn_done":
                turn_done = True
                break
            elif typ == "error":
                return {"ok": False, "error": msg, "scenario": "half_duplex"}

        await ws.send(json.dumps({"type": "stop"}))

    full = "".join(assistant_text_parts).strip()
    too_short = len(full) < 4
    return {
        "ok": turn_done and not too_short,
        "scenario": "half_duplex",
        "turn_done": turn_done,
        "assistant_text_len": len(full),
        "assistant_preview": full[:200] + ("…" if len(full) > 200 else ""),
        "too_short_suspect_truncation": too_short,
        "half_duplex_chunk_samples": chunk_samples,
        "half_duplex_send_interval_s": chunk_interval_s,
        "half_duplex_tts_enabled": tts_enabled,
    }


def run_duplex_replay_omni(
    gateway_http: str,
    llamacpp_root: Path,
    count: int,
) -> Dict[str, Any]:
    """
    对应 Omni / Audio duplex 的浏览器侧：DuplexSession 发 prepare + audio_chunk。
    这里用录制回放逼近真实节奏；「尾音截断」可对人耳听 merged_output.wav。
    """
    env = {**__import__("os").environ, "PYTHONPATH": str(PROJECT)}
    mat = subprocess.run(
        [
            str(VENV_PY),
            str(PROJECT / "scripts" / "materialize_duplex_session_from_cpp_assets.py"),
            "--llamacpp-root",
            str(llamacpp_root),
            "--count",
            str(count),
            "--system-prompt",
            "请用中文简短交流。",
        ],
        cwd=str(PROJECT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if mat.returncode != 0:
        return {"ok": False, "scenario": "duplex_replay", "error": mat.stderr[-1500:]}

    session_dir = Path(mat.stdout.strip().splitlines()[-1].strip())
    if not session_dir.is_dir():
        return {"ok": False, "scenario": "duplex_replay", "error": f"bad session {session_dir}"}

    ws_base = _http_to_ws_base(gateway_http)
    rep = subprocess.run(
        [
            str(VENV_PY),
            str(PROJECT / "replay_duplex_session.py"),
            "--session-dir",
            str(session_dir),
            "--gateway-ws-base",
            ws_base,
            "--chunk-duration-s",
            "1.0",
            "--timing-mode",
            "fixed",
            "--send-interval-s",
            "1.0",
            "--pre-stop-wait-s",
            "12",
            "--post-stop-wait-s",
            "20",
            "--insecure",
        ],
        cwd=str(PROJECT),
        env=env,
        capture_output=True,
        text=True,
        timeout=360,
    )
    if rep.returncode != 0:
        return {"ok": False, "scenario": "duplex_replay", "error": rep.stderr[-2000:]}

    out_dirs = sorted(session_dir.glob("replay_out_*"), key=lambda p: p.stat().st_mtime)
    summary_path = out_dirs[-1] / "summary.json" if out_dirs else None
    stats: Dict[str, Any] = {}
    if summary_path and summary_path.is_file():
        stats = json.loads(summary_path.read_text(encoding="utf-8")).get("duplex_stats") or {}
    nlisten = int(stats.get("result_listen", 0))
    nspeak = int(stats.get("result_speak", 0))
    ok = nlisten + nspeak >= 1
    return {
        "ok": ok,
        "scenario": "duplex_replay",
        "duplex_stats": stats,
        "session_dir": str(session_dir),
        "note": "尾音是否截断请听 replay_out_*/merged_output.wav 末尾 ~200ms",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="浏览器协议级 QA 场景（Gateway 已启动）")
    ap.add_argument("--gateway", default="http://127.0.0.1:8020")
    ap.add_argument(
        "--llamacpp-root",
        type=Path,
        default=Path("/cache/caitianchi/code/llama.cpp-omni"),
    )
    ap.add_argument("--timeout-turnbased", type=float, default=180.0)
    ap.add_argument(
        "--timeout-half-duplex",
        type=float,
        default=300.0,
        help="Half-duplex 等待 turn_done 上限（默认 300s，含 TTS）",
    )
    ap.add_argument("--skip-turnbased", action="store_true")
    ap.add_argument("--skip-half-duplex", action="store_true")
    ap.add_argument("--skip-duplex-replay", action="store_true")
    ap.add_argument(
        "--half-duplex-wav",
        type=Path,
        default=None,
        help="16kHz 用户语音 WAV；不传则用 llamacpp-root 下 tools/omni/assets/test_case 中 C++ 同款素材（见脚本文档）",
    )
    ap.add_argument(
        "--half-duplex-asset-index",
        type=int,
        default=0,
        help="兼容旧参数：在未改 --half-duplex-merge-start 时，用作合并起始编号（等同 merge-start）",
    )
    ap.add_argument(
        "--half-duplex-no-tts",
        action="store_true",
        help="关闭 TTS（网页默认为开；自动化想加速时用）",
    )
    ap.add_argument(
        "--half-duplex-merge-count",
        type=int,
        default=2,
        help="将 test_case 中连续多段 WAV 合并为一条用户语音（默认 2，与 C++ audio 测例常用段数一致）；设为 1 即单段",
    )
    ap.add_argument(
        "--half-duplex-merge-start",
        type=int,
        default=0,
        help="合并时的起始编号 NNNN（如 0000）",
    )
    ap.add_argument(
        "--half-duplex-merge-gap-ms",
        type=float,
        default=200.0,
        help="合并时段与段之间的静音毫秒数",
    )
    ap.add_argument(
        "--half-duplex-chunk-ms",
        type=float,
        default=500.0,
        help="每包时长(ms)@16kHz，默认 500 与 half-duplex-app.js CHUNK_DURATION_S=0.5 一致",
    )
    ap.add_argument(
        "--half-duplex-send-interval-s",
        type=float,
        default=0.5,
        help="相邻 audio_chunk 间隔(秒)，默认 0.5 对齐每 0.5s 一块实时麦克风节奏；压测可改小",
    )
    ap.add_argument(
        "--half-duplex-cpp-one-second-stream",
        action="store_true",
        help="16000 样本/包 + 1.0s 间隔，对齐 C++ test_case 按秒切片（非网页默认）",
    )
    ap.add_argument("--duplex-replay-count", type=int, default=2)
    args = ap.parse_args()

    results: List[Dict[str, Any]] = []

    if not args.skip_turnbased:
        results.append(asyncio.run(scenario_turnbased_stream_vs_done(args.gateway, args.timeout_turnbased)))

    if not args.skip_half_duplex:
        merged_sources: List[str] = []
        merged_saved: Optional[Path] = None
        if args.half_duplex_wav:
            probe_label = str(args.half_duplex_wav.resolve())
            audio = _load_wav_16k_mono(args.half_duplex_wav)
            merged_sources = [probe_label]
        else:
            start = args.half_duplex_merge_start
            if args.half_duplex_asset_index != 0 and args.half_duplex_merge_start == 0:
                # 兼容旧参数：仅指定了 --half-duplex-asset-index 时当作合并起点
                start = args.half_duplex_asset_index
            audio, probe_label, merged_sources = _merge_test_case_wavs(
                args.llamacpp_root,
                start=start,
                count=max(1, args.half_duplex_merge_count),
                gap_ms=args.half_duplex_merge_gap_ms,
            )
            if len(merged_sources) > 1 or probe_label.startswith("merged:"):
                cache_dir = PROJECT / "data" / "qa_cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                merged_saved = cache_dir / "half_duplex_merged_probe.wav"
                if sf is not None:
                    sf.write(str(merged_saved), audio, 16000, subtype="FLOAT")
        audio = np.concatenate([audio, _half_duplex_silence_tail()])
        if args.half_duplex_cpp_one_second_stream:
            chunk_ms = 1000.0
            chunk_interval_s = 1.0
        else:
            chunk_ms = args.half_duplex_chunk_ms
            chunk_interval_s = args.half_duplex_send_interval_s
        chunk_samples = max(1024, int(round(16000.0 * (chunk_ms / 1000.0))))
        sid = f"hdx_qa_{int(time.time())}"
        tts_on = not args.half_duplex_no_tts
        half_res = asyncio.run(
            scenario_half_duplex_like_browser(
                args.gateway,
                sid,
                audio,
                chunk_samples=chunk_samples,
                chunk_interval_s=chunk_interval_s,
                tts_enabled=tts_on,
                max_new_tokens=256,
                timeout_s=args.timeout_half_duplex,
            )
        )
        half_res["half_duplex_probe_label"] = probe_label
        half_res["half_duplex_source_wavs"] = merged_sources
        if merged_saved is not None:
            half_res["half_duplex_merged_wav"] = str(merged_saved.resolve())
        results.append(half_res)

    if not args.skip_duplex_replay:
        results.append(run_duplex_replay_omni(args.gateway, args.llamacpp_root, args.duplex_replay_count))

    print(json.dumps(results, ensure_ascii=False, indent=2))
    failed = [r for r in results if not r.get("ok")]
    if failed:
        print("\n[FAIL] 未通过的场景:", file=sys.stderr)
        for r in failed:
            print(json.dumps(r, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)
    print("\n[OK] 本脚本所含自动化场景均已通过（语义类问题仍需人工 + 真实录音）。")


if __name__ == "__main__":
    main()
