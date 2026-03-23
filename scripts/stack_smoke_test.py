#!/usr/bin/env python3
"""
完整栈冒烟：Gateway / Worker / 静态页 / Chat HTTP / Chat WebSocket / Duplex 回放（C++ 测试素材）。

用法（服务已启动时）：
  PYTHONPATH=. .venv/base/bin/python scripts/stack_smoke_test.py
  PYTHONPATH=. .venv/base/bin/python scripts/stack_smoke_test.py --gateway http://127.0.0.1:8020
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx

PROJECT = Path(__file__).resolve().parents[1]
VENV_PY = PROJECT / ".venv" / "base" / "bin" / "python"


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)
    raise SystemExit(1)


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def check_gateway(gw: str) -> None:
    r = httpx.get(f"{gw.rstrip('/')}/health", timeout=10.0)
    if r.status_code != 200:
        _fail(f"gateway /health -> {r.status_code}")
    _ok("gateway /health")


def check_status(gw: str, min_workers: int) -> None:
    r = httpx.get(f"{gw.rstrip('/')}/status", timeout=10.0)
    if r.status_code != 200:
        _fail(f"gateway /status -> {r.status_code}")
    d = r.json()
    if not d.get("gateway_healthy"):
        _fail("gateway_healthy is false")
    if int(d.get("total_workers", 0)) < min_workers:
        _fail(f"total_workers={d.get('total_workers')} < {min_workers}")
    if int(d.get("offline_workers", 0)) > 0:
        _fail(f"offline_workers={d.get('offline_workers')}")
    _ok(f"gateway /status workers={d['total_workers']} idle={d.get('idle_workers')}")


def check_worker(wu: str) -> None:
    r = httpx.get(f"{wu.rstrip('/')}/health", timeout=10.0)
    if r.status_code != 200:
        _fail(f"worker /health -> {r.status_code}")
    d = r.json()
    if not d.get("model_loaded"):
        _fail("worker model_loaded is false")
    _ok(f"worker /health gpu_id={d.get('gpu_id')}")


def check_pages(gw: str, paths: List[str]) -> None:
    base = gw.rstrip("/")
    for p in paths:
        r = httpx.get(f"{base}{p}", timeout=15.0, follow_redirects=True)
        if r.status_code != 200:
            _fail(f"GET {p} -> {r.status_code}")
    _ok(f"static pages x{len(paths)}")


def check_chat_http(gw: str) -> None:
    last_err: str = ""
    for attempt in range(1, 4):
        try:
            r = httpx.post(
                f"{gw.rstrip('/')}/api/chat",
                json={
                    "messages": [{"role": "user", "content": "用不超过五个字回答：2+3=?"}],
                    "generation": {"max_new_tokens": 32},
                },
                timeout=240.0,
            )
        except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            last_err = str(e)
            if attempt < 3:
                time.sleep(5.0)
                continue
            _fail(f"/api/chat timeout after retries: {last_err}")
            return
        if r.status_code != 200:
            _fail(f"/api/chat -> {r.status_code} {r.text[:200]}")
        data = r.json()
        if not data.get("success"):
            _fail(f"/api/chat success=false err={data.get('error')}")
        text = (data.get("text") or "").strip()
        if len(text) < 1:
            _fail("/api/chat empty text")
        _ok(f"/api/chat len(text)={len(text)}")
        return


async def check_chat_ws(gw: str) -> None:
    import websockets

    base = gw.replace("http://", "ws://").replace("https://", "wss://").rstrip("/")
    uri = f"{base}/ws/chat"
    # 非流式 done 可能带大块 audio（易触发 WS 单帧上限）；流式 + 关 TTS 则多为小 chunk
    # 与 turnbased.html::_buildChatWSBody 一致（纯文本、关 TTS 时不带 tts 字段）
    payload = json.dumps(
        {
            "messages": [{"role": "user", "content": "只回答一个数字：1+1"}],
            "streaming": True,
            "generation": {
                "max_new_tokens": 16,
                "do_sample": True,
                "length_penalty": 1.1,
            },
            "image": {"max_slice_nums": 1},
            "omni_mode": False,
        },
        ensure_ascii=False,
    )
    got_done = False
    async with websockets.connect(uri, max_size=50_000_000) as ws:
        await ws.send(payload)
        for _ in range(200):
            raw = await asyncio.wait_for(ws.recv(), timeout=120.0)
            msg = json.loads(raw)
            t = msg.get("type")
            if t == "error":
                _fail(f"ws/chat error: {msg}")
            if t == "done":
                got_done = True
                break
    if not got_done:
        _fail("ws/chat no done")
    _ok("ws/chat done")


def run_duplex_replay(gw: str, llamacpp: Path, count: int) -> Dict[str, Any]:
    env = {**os.environ, "PYTHONPATH": str(PROJECT)}
    mat = subprocess.run(
        [
            str(VENV_PY),
            str(PROJECT / "scripts" / "materialize_duplex_session_from_cpp_assets.py"),
            "--llamacpp-root",
            str(llamacpp),
            "--count",
            str(count),
            "--system-prompt",
            "请用中文简短交流。",
        ],
        cwd=str(PROJECT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if mat.returncode != 0:
        _fail(f"materialize stderr={mat.stderr}")
    session_dir = Path(mat.stdout.strip().splitlines()[-1].strip())
    if not session_dir.is_dir():
        _fail(f"bad session dir: {session_dir}")

    ws_base = gw.replace("http://", "ws://").replace("https://", "wss://").rstrip("/")
    rep = subprocess.run(
        [
            str(VENV_PY),
            str(PROJECT / "replay_duplex_session.py"),
            "--session-dir",
            str(session_dir),
            "--gateway-ws-base",
            ws_base,
            "--timing-mode",
            "fixed",
            "--send-interval-s",
            "1.2",
            "--pre-stop-wait-s",
            "15",
            "--post-stop-wait-s",
            "25",
            "--insecure",
        ],
        cwd=str(PROJECT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if rep.returncode != 0:
        _fail(f"replay stderr={rep.stderr[-2000:]}")

    out_dirs = sorted(session_dir.glob("replay_out_*"), key=lambda p: p.stat().st_mtime)
    if not out_dirs:
        _fail("no replay_out_*")
    summary_path = out_dirs[-1] / "summary.json"
    if not summary_path.is_file():
        _fail(f"missing {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stats = summary.get("duplex_stats") or {}
    nlisten = int(stats.get("result_listen", 0))
    nspeak = int(stats.get("result_speak", 0))
    if nlisten + nspeak < 1:
        _fail(f"no duplex result messages stats={stats} stdout={rep.stdout[-1500:]}")
    _ok(f"duplex replay stats listen={nlisten} speak={nspeak} packets={summary.get('output_packets')}")
    return summary


def check_llama_server(llamacpp: Path) -> None:
    for rel in ("build/bin/llama-server",):
        p = llamacpp / rel
        if p.is_file():
            _ok(f"llama-server exists mtime={time.ctime(p.stat().st_mtime)}")
            return
    _fail(f"llama-server not found under {llamacpp}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default="http://127.0.0.1:8020")
    ap.add_argument("--worker", default="http://127.0.0.1:22400")
    ap.add_argument(
        "--llamacpp-root",
        type=Path,
        default=Path("/cache/caitianchi/code/llama.cpp-omni"),
    )
    ap.add_argument("--duplex-chunks", type=int, default=3)
    ap.add_argument("--min-total-workers", type=int, default=1)
    args = ap.parse_args()

    if not VENV_PY.is_file():
        _fail(f"venv python missing: {VENV_PY}")

    print("=== stack_smoke_test ===")
    check_llama_server(args.llamacpp_root)
    check_gateway(args.gateway)
    check_status(args.gateway, args.min_total_workers)
    check_worker(args.worker)
    check_pages(
        args.gateway,
        ["/", "/omni", "/turnbased", "/audio_duplex", "/half_duplex", "/admin", "/docs"],
    )
    check_chat_http(args.gateway)
    asyncio.run(check_chat_ws(args.gateway))
    run_duplex_replay(args.gateway, args.llamacpp_root, args.duplex_chunks)
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
