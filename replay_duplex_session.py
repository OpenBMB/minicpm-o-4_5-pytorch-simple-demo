#!/usr/bin/env python3
"""
回放 Duplex 会话数据到后端，并保存后端音频输出。

功能：
1) 读取 session 目录中的 meta.json / recording.json
2) 按 recording.json 的 receive_ts_ms 节奏发送 audio_chunk（可带 frame）
3) 接收后端 result/audio_only 音频
4) 保存：
   - 每个输出包的 raw wav
   - 按 1 秒切片的 sec_XXXX.wav
   - merged_output.wav
   - summary.json

示例：
python replay_duplex_session.py \
  --session-dir /cache/caitianchi/code/minicpm-o-4_5-pytorch-simple-demo/data/sessions/20260316_185629_omni_mmt2j2h9 \
  --gateway-ws-base ws://127.0.0.1:8006
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import random
import ssl
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import websockets

INPUT_SR = 16000
OUTPUT_SR = 24000


@dataclass
class AudioPacket:
    index: int
    msg_type: str
    recv_ts_ms: float
    samples: int
    audio: np.ndarray


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_base64_f32(audio: np.ndarray) -> str:
    audio = np.asarray(audio, dtype=np.float32)
    return base64.b64encode(audio.tobytes()).decode("utf-8")


def _from_base64_f32(audio_b64: str) -> np.ndarray:
    raw = base64.b64decode(audio_b64)
    return np.frombuffer(raw, dtype=np.float32)


def _read_audio_16k_mono(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32)
    if sr != INPUT_SR:
        # 简单线性重采样，避免额外依赖
        old_x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False, dtype=np.float64)
        new_len = int(round(len(audio) * INPUT_SR / sr))
        new_x = np.linspace(0.0, 1.0, num=max(new_len, 1), endpoint=False, dtype=np.float64)
        audio = np.interp(new_x, old_x, audio).astype(np.float32)
    return audio


def _load_ref_audio_b64(meta: Dict[str, Any], session_dir: Path, cli_ref_audio: Optional[str]) -> Optional[str]:
    if cli_ref_audio:
        p = Path(cli_ref_audio)
        if p.exists():
            return _to_base64_f32(_read_audio_16k_mono(p))
        return None

    ref_audio = meta.get("config", {}).get("ref_audio")
    if not ref_audio:
        return None
    p = Path(ref_audio)
    if not p.is_absolute():
        p = session_dir / ref_audio
    if p.exists():
        return _to_base64_f32(_read_audio_16k_mono(p))
    return None


def _resolve_session_id(app_type: str, cli_session_id: Optional[str]) -> str:
    if cli_session_id:
        return cli_session_id
    # gateway 根据 adx_ 前缀判断 audio_duplex；其余视为 omni_duplex
    prefix = "adx" if "audio" in app_type else "omni"
    return f"{prefix}_replay_{int(time.time())}_{random.randint(1000, 9999)}"


def _collect_send_items(
    session_dir: Path,
    recording: Dict[str, Any],
    app_type: str,
    chunk_duration_s: float,
) -> List[Dict[str, Any]]:
    chunks = recording.get("chunks", [])
    if not chunks:
        raise RuntimeError("recording.json 里没有 chunks")

    merged_audio_path = session_dir / recording.get("merged_replay", "merged_replay.wav")
    merged_audio: Optional[np.ndarray] = None
    if merged_audio_path.exists():
        merged_audio = _read_audio_16k_mono(merged_audio_path)

    chunk_samples = int(round(chunk_duration_s * INPUT_SR))
    items: List[Dict[str, Any]] = []

    for i, ch in enumerate(chunks):
        ts_ms = float(ch.get("receive_ts_ms", i * chunk_duration_s * 1000.0))

        audio_rel = ch.get("user_audio")
        audio_path = session_dir / audio_rel if audio_rel else None
        if audio_path and audio_path.exists():
            audio_np = _read_audio_16k_mono(audio_path)
        elif merged_audio is not None:
            start = i * chunk_samples
            end = start + chunk_samples
            if start < len(merged_audio):
                audio_np = merged_audio[start:end]
            else:
                audio_np = np.zeros((chunk_samples,), dtype=np.float32)
            if len(audio_np) < chunk_samples:
                audio_np = np.pad(audio_np, (0, chunk_samples - len(audio_np)), mode="constant")
        else:
            audio_np = np.zeros((chunk_samples,), dtype=np.float32)

        msg: Dict[str, Any] = {
            "type": "audio_chunk",
            "audio_base64": _to_base64_f32(audio_np),
        }

        # omni 模式尽量回放帧
        if "omni" in app_type:
            frame_rel = ch.get("user_frame")
            frame_path = session_dir / frame_rel if frame_rel else None
            if frame_path and frame_path.exists():
                frame_bytes = frame_path.read_bytes()
                msg["frame_base64_list"] = [base64.b64encode(frame_bytes).decode("utf-8")]

        items.append({"ts_ms": ts_ms, "msg": msg, "samples": int(len(audio_np))})

    return items


def _save_outputs(
    out_dir: Path,
    packets: List[AudioPacket],
    split_seconds: float,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw_packets"
    sec_dir = out_dir / "per_second"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sec_dir.mkdir(parents=True, exist_ok=True)

    for p in packets:
        sf.write(str(raw_dir / f"raw_{p.index:04d}_{p.msg_type}.wav"), p.audio, OUTPUT_SR, subtype="PCM_16")

    if packets:
        merged = np.concatenate([p.audio for p in packets]).astype(np.float32)
    else:
        merged = np.zeros((0,), dtype=np.float32)

    sf.write(str(out_dir / "merged_output.wav"), merged, OUTPUT_SR, subtype="PCM_16")

    sec_samples = int(round(split_seconds * OUTPUT_SR))
    per_second_files: List[str] = []
    if sec_samples > 0 and len(merged) > 0:
        idx = 0
        sec_i = 0
        while idx < len(merged):
            seg = merged[idx: idx + sec_samples]
            if len(seg) < sec_samples:
                seg = np.pad(seg, (0, sec_samples - len(seg)), mode="constant")
            fn = f"sec_{sec_i:04d}.wav"
            sf.write(str(sec_dir / fn), seg, OUTPUT_SR, subtype="PCM_16")
            per_second_files.append(f"per_second/{fn}")
            idx += sec_samples
            sec_i += 1

    return {
        "raw_packet_count": len(packets),
        "merged_samples": int(len(merged)),
        "merged_duration_s": round(float(len(merged) / OUTPUT_SR), 3),
        "per_second_count": len(per_second_files),
        "per_second_files": per_second_files,
    }


async def _run(args: argparse.Namespace) -> None:
    session_dir = Path(args.session_dir).resolve()
    meta_path = session_dir / "meta.json"
    rec_path = session_dir / "recording.json"
    if not meta_path.exists() or not rec_path.exists():
        raise RuntimeError(f"session 目录缺少 meta.json/recording.json: {session_dir}")

    meta = _read_json(meta_path)
    recording = _read_json(rec_path)
    app_type = str(meta.get("app_type") or recording.get("config", {}).get("app_type") or "omni_duplex")

    session_id = _resolve_session_id(app_type=app_type, cli_session_id=args.session_id)
    ws_url = f"{args.gateway_ws_base.rstrip('/')}/ws/duplex/{session_id}"
    print(f"[INFO] app_type={app_type}, ws={ws_url}")

    send_items = _collect_send_items(
        session_dir=session_dir,
        recording=recording,
        app_type=app_type,
        chunk_duration_s=args.chunk_duration_s,
    )
    print(f"[INFO] prepared {len(send_items)} input chunks")

    ref_audio_b64 = _load_ref_audio_b64(meta, session_dir, args.ref_audio)
    system_prompt = args.system_prompt or meta.get("config", {}).get("system_prompt") or "Streaming Omni Conversation."
    max_slice_nums = int(meta.get("config", {}).get("max_slice_nums", 1))

    output_packets: List[AudioPacket] = []
    recv_msg_count = 0
    sender_done = asyncio.Event()

    def _make_ssl_ctx(target_url: str, insecure: bool) -> Optional[ssl.SSLContext]:
        if not target_url.startswith("wss://"):
            return None
        ctx = ssl.create_default_context()
        if insecure:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        return ctx

    async def _open_ws(target_url: str, insecure: bool):
        _cm = websockets.connect(
            target_url,
            max_size=128 * 1024 * 1024,
            ssl=_make_ssl_ctx(target_url, insecure=insecure),
        )
        _ws = await _cm.__aenter__()
        return _cm, _ws

    actual_ws_url = ws_url
    ws_cm = None
    ws = None
    try:
        ws_cm, ws = await _open_ws(actual_ws_url, insecure=args.insecure)
    except websockets.exceptions.InvalidMessage as e:
        # 常见场景：用户填了 ws://，但端口实际是 wss://
        if actual_ws_url.startswith("ws://"):
            fallback_url = "wss://" + actual_ws_url[len("ws://"):]
            print(f"[WARN] ws 握手失败，尝试回退到: {fallback_url}")
            try:
                ws_cm, ws = await _open_ws(fallback_url, insecure=args.insecure)
                actual_ws_url = fallback_url
                print(f"[INFO] protocol fallback success: {actual_ws_url}")
            except ssl.SSLCertVerificationError as cert_err:
                # 自动再试一次 insecure，减少手工重试
                try:
                    print("[WARN] wss 证书校验失败，自动改为 insecure 重试一次...")
                    ws_cm, ws = await _open_ws(fallback_url, insecure=True)
                    actual_ws_url = fallback_url
                    print(f"[INFO] insecure fallback success: {actual_ws_url}")
                except Exception as insecure_err:
                    fallback_base = fallback_url.split("/ws/duplex/")[0]
                    raise RuntimeError(
                        "ws 握手失败，且 wss 证书校验失败（可能是自签名证书）。\n"
                        f"请重试：--gateway-ws-base {fallback_base} --insecure"
                    ) from insecure_err
            except Exception as fallback_err:
                fallback_base = fallback_url.split("/ws/duplex/")[0]
                raise RuntimeError(
                    "WebSocket 握手失败。\n"
                    f"尝试过：{actual_ws_url} 和 {fallback_url}\n"
                    f"请确认网关端口/协议是否正确，或改用 --gateway-ws-base {fallback_base}"
                ) from fallback_err
        else:
            raise
    except ssl.SSLCertVerificationError as e:
        raise RuntimeError(
            "WSS 证书校验失败（可能是自签名证书）。可加 --insecure 关闭校验。"
        ) from e

    try:
        prepare_msg: Dict[str, Any] = {
            "type": "prepare",
            "system_prompt": system_prompt,
            "max_slice_nums": max_slice_nums,
            "deferred_finalize": True,
        }
        if ref_audio_b64:
            prepare_msg["ref_audio_base64"] = ref_audio_b64
            prepare_msg["tts_ref_audio_base64"] = ref_audio_b64

        await ws.send(json.dumps(prepare_msg, ensure_ascii=False))

        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            tp = msg.get("type")
            if tp == "prepared":
                print("[INFO] prepared received")
                break
            if tp in ("queued", "queue_update"):
                print(f"[QUEUE] {tp}: {msg}")
            elif tp == "error":
                raise RuntimeError(f"prepare failed: {msg}")

        start_send_wall = time.perf_counter()
        first_ts = float(send_items[0]["ts_ms"]) if send_items else 0.0
        speed = max(args.speed, 1e-6)

        async def sender() -> None:
            for i, item in enumerate(send_items):
                # 使用“绝对时间点”发送，避免“发完再等 N 秒”的累计漂移
                if args.timing_mode == "recorded":
                    target_elapsed = (float(item["ts_ms"]) - first_ts) / 1000.0 / speed
                else:
                    # fixed: 第 i 个包在 t0 + i * interval 发送（可再叠加 speed）
                    target_elapsed = (i * args.send_interval_s) / speed
                now_elapsed = time.perf_counter() - start_send_wall
                sleep_s = target_elapsed - now_elapsed
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
                await ws.send(json.dumps(item["msg"], ensure_ascii=False))
                if i % 10 == 0:
                    print(f"[SEND] chunk={i}/{len(send_items)} samples={item['samples']}")
            await ws.send(json.dumps({"type": "stop"}))
            sender_done.set()
            print("[SEND] stop sent")

        async def receiver() -> None:
            nonlocal recv_msg_count
            packet_index = 0
            stop_deadline: Optional[float] = None
            while True:
                timeout_s = 2.0
                if sender_done.is_set():
                    if stop_deadline is None:
                        stop_deadline = time.perf_counter() + args.post_stop_wait_s
                    remaining = stop_deadline - time.perf_counter()
                    if remaining <= 0:
                        return
                    timeout_s = min(timeout_s, max(0.1, remaining))
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
                except asyncio.TimeoutError:
                    if sender_done.is_set():
                        continue
                    continue
                except websockets.exceptions.ConnectionClosed:
                    return

                recv_msg_count += 1
                msg = json.loads(raw)
                tp = msg.get("type")
                recv_ts_ms = (time.perf_counter() - start_send_wall) * 1000.0

                if tp == "result":
                    if not msg.get("is_listen", True) and msg.get("audio_data"):
                        audio = _from_base64_f32(msg["audio_data"]).astype(np.float32)
                        output_packets.append(
                            AudioPacket(
                                index=packet_index,
                                msg_type="result",
                                recv_ts_ms=recv_ts_ms,
                                samples=int(len(audio)),
                                audio=audio,
                            )
                        )
                        packet_index += 1
                elif tp == "audio_only":
                    if msg.get("audio_data"):
                        audio = _from_base64_f32(msg["audio_data"]).astype(np.float32)
                        output_packets.append(
                            AudioPacket(
                                index=packet_index,
                                msg_type="audio_only",
                                recv_ts_ms=recv_ts_ms,
                                samples=int(len(audio)),
                                audio=audio,
                            )
                        )
                        packet_index += 1
                elif tp == "stopped":
                    if sender_done.is_set():
                        return
                elif tp == "error":
                    print(f"[WARN] server error: {msg}")

        await asyncio.gather(sender(), receiver())
    finally:
        if ws_cm is not None:
            await ws_cm.__aexit__(None, None, None)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (session_dir / f"replay_out_{ts}")
    save_summary = _save_outputs(out_dir=out_dir, packets=output_packets, split_seconds=args.output_split_seconds)

    summary = {
        "session_dir": str(session_dir),
        "app_type": app_type,
        "ws_url": ws_url,
        "send_chunks": len(send_items),
        "recv_messages": recv_msg_count,
        "output_packets": len(output_packets),
        "speed": args.speed,
        "output": save_summary,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] output saved to: {out_dir}")
    print(
        "[DONE] packets=%d merged=%.2fs per_second=%d"
        % (
            summary["output_packets"],
            save_summary["merged_duration_s"],
            save_summary["per_second_count"],
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay duplex session to backend and save output audio.")
    parser.add_argument("--session-dir", required=True, help="会话目录，需包含 meta.json / recording.json")
    parser.add_argument("--gateway-ws-base", default="ws://127.0.0.1:8006", help="网关 WebSocket 基地址")
    parser.add_argument("--session-id", default=None, help="可选：指定 replay 的 session_id")
    parser.add_argument("--system-prompt", default=None, help="可选：覆盖 system_prompt")
    parser.add_argument("--ref-audio", default=None, help="可选：覆盖 ref 音频路径（16k wav）")
    parser.add_argument("--speed", type=float, default=1.0, help="回放倍速，1.0=实时，2.0=2倍速")
    parser.add_argument(
        "--timing-mode",
        choices=["fixed", "recorded"],
        default="fixed",
        help="发送节奏：fixed=固定时间点发送；recorded=按 recording.json 的 receive_ts_ms 回放",
    )
    parser.add_argument(
        "--send-interval-s",
        type=float,
        default=1.0,
        help="fixed 模式下两个音频发送时间点的间隔（秒）",
    )
    parser.add_argument("--chunk-duration-s", type=float, default=1.0, help="fallback 切 chunk 时长（秒）")
    parser.add_argument("--post-stop-wait-s", type=float, default=3.0, help="发送 stop 后额外接收等待秒数")
    parser.add_argument("--output-split-seconds", type=float, default=1.0, help="输出切片时长（秒）")
    parser.add_argument("--output-dir", default=None, help="输出目录；默认 session_dir/replay_out_<ts>")
    parser.add_argument("--insecure", action="store_true", help="当使用 wss:// 且证书为自签名时关闭 SSL 校验")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()

