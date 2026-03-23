#!/usr/bin/env python3
"""
从 llama.cpp-omni `tools/omni/assets/test_case/duplex_omni_test_case/`（与同目录下的
`audio_test_case/`、`omni_test_case/` 一样，供 C++ omni-cli / test-duplex 使用）生成可回放的会话目录
（meta.json + recording.json + user_audio/*.wav + user_frames/*.jpg），
供 replay_duplex_session.py 走与 Omni 网页相同的网关 WebSocket 路径。

示例：
  PYTHONPATH=. .venv/base/bin/python scripts/materialize_duplex_session_from_cpp_assets.py \\
    --llamacpp-root /path/to/llama.cpp-omni --count 3
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--llamacpp-root",
        type=Path,
        default=Path("/cache/caitianchi/code/llama.cpp-omni"),
        help="llama.cpp-omni 仓库根目录",
    )
    p.add_argument("--count", type=int, default=3, help="使用前几组 wav/jpg（0000 起）")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="输出会话目录；默认 data/sessions/duplex_cpp_assets_<ts>",
    )
    p.add_argument(
        "--ref-audio",
        type=Path,
        default=Path("assets/ref_audio/ref_minicpm_signature.wav"),
        help="写入 meta 的参考 wav（相对 demo 根或绝对路径）",
    )
    p.add_argument(
        "--system-prompt",
        default="Streaming Omni Conversation.",
        help="recording / meta 中的 system_prompt",
    )
    args = p.parse_args()

    demo_root = Path(__file__).resolve().parents[1]
    asset_dir = (
        args.llamacpp_root
        / "tools/omni/assets/test_case/duplex_omni_test_case"
    )
    if not asset_dir.is_dir():
        raise SystemExit(f"资产目录不存在: {asset_dir}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    session_id = f"duplex_cpp_assets_{ts}"
    out = args.out_dir or (demo_root / "data" / "sessions" / session_id)
    out = out.resolve()
    ua = out / "user_audio"
    uf = out / "user_frames"
    ua.mkdir(parents=True, exist_ok=True)
    uf.mkdir(parents=True, exist_ok=True)

    ref_audio = args.ref_audio
    if not ref_audio.is_absolute():
        ref_audio = (demo_root / ref_audio).resolve()

    chunks = []
    for i in range(args.count):
        src_wav = asset_dir / f"duplex_omni_test_case_{i:04d}.wav"
        src_jpg = asset_dir / f"duplex_omni_test_case_{i:04d}.jpg"
        if not src_wav.is_file():
            raise SystemExit(f"缺少测试音频: {src_wav}")
        dst_wav = ua / f"{i:03d}.wav"
        dst_jpg = uf / f"{i:03d}.jpg"
        shutil.copy2(src_wav, dst_wav)
        if src_jpg.is_file():
            shutil.copy2(src_jpg, dst_jpg)
        ch: dict = {
            "index": i,
            "receive_ts_ms": 1000.0 * i,
            "user_audio": f"user_audio/{i:03d}.wav",
        }
        if src_jpg.is_file():
            ch["user_frame"] = f"user_frames/{i:03d}.jpg"
        chunks.append(ch)

    meta = {
        "session_id": session_id,
        "app_type": "omni_duplex",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "system_prompt": args.system_prompt,
            "ref_audio": str(ref_audio),
            "deferred_finalize": True,
            "max_slice_nums": 1,
        },
    }
    recording = {
        "session_id": session_id,
        "mode": "duplex",
        "worker_id": 0,
        "start_ts": meta["created_at"],
        "config": meta["config"],
        "chunks": chunks,
    }

    (out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "recording.json").write_text(
        json.dumps(recording, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(out)


if __name__ == "__main__":
    main()
