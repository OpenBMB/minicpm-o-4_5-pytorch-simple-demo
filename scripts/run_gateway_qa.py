#!/usr/bin/env python3
"""
在本机栈已启动时顺序执行网关级 QA（不只是准备数据）：

1. GET /health
2. pytest tests/test_api.py（先跑，避免 stack/QA 长跑后 Worker loading/429 误伤）
3. scripts/stack_smoke_test.py（含 duplex 回放）
4. scripts/qa_browser_like_scenarios.py（turn-based + half-duplex + duplex）

用法：
  PYTHONPATH=. .venv/base/bin/python scripts/run_gateway_qa.py
  PYTHONPATH=. .venv/base/bin/python scripts/run_gateway_qa.py --gateway http://127.0.0.1:8020 --skip-stack-smoke
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import httpx

PROJECT = Path(__file__).resolve().parents[1]
VENV_PY = PROJECT / ".venv" / "base" / "bin" / "python"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default="http://127.0.0.1:8020")
    ap.add_argument("--worker", default="http://127.0.0.1:22400")
    ap.add_argument(
        "--llamacpp-root",
        type=Path,
        default=Path("/cache/caitianchi/code/llama.cpp-omni"),
    )
    ap.add_argument("--skip-stack-smoke", action="store_true")
    ap.add_argument("--skip-qa-scenarios", action="store_true")
    ap.add_argument("--skip-pytest", action="store_true")
    ap.add_argument("--duplex-chunks", type=int, default=3)
    args = ap.parse_args()

    if not VENV_PY.is_file():
        print(f"[FAIL] 需要 venv: {VENV_PY}", file=sys.stderr)
        sys.exit(1)

    env = {**__import__("os").environ, "PYTHONPATH": str(PROJECT)}
    gw = args.gateway.rstrip("/")

    def _log(msg: str) -> None:
        print(msg, flush=True)

    _log("=== 1) gateway /health ===")
    try:
        r = httpx.get(f"{gw}/health", timeout=15.0)
        if r.status_code != 200:
            print(f"[FAIL] /health -> {r.status_code}", file=sys.stderr)
            sys.exit(1)
        _log("[OK] /health")
    except Exception as e:
        print(f"[FAIL] 无法连接 Gateway {gw}: {e}", file=sys.stderr)
        sys.exit(1)

    if not args.skip_pytest:
        _log("=== 2) pytest tests/test_api.py ===")
        p = subprocess.run(
            [str(VENV_PY), "-m", "pytest", "tests/test_api.py", "-v", "--tb=short"],
            cwd=str(PROJECT),
            env=env,
        )
        if p.returncode != 0:
            sys.exit(p.returncode)

    if not args.skip_stack_smoke:
        _log("=== 3) stack_smoke_test.py ===")
        cmd = [
            str(VENV_PY),
            str(PROJECT / "scripts" / "stack_smoke_test.py"),
            "--gateway",
            gw,
            "--worker",
            args.worker,
            "--llamacpp-root",
            str(args.llamacpp_root),
            "--duplex-chunks",
            str(args.duplex_chunks),
            "--min-total-workers",
            "1",
        ]
        p = subprocess.run(cmd, cwd=str(PROJECT), env=env)
        if p.returncode != 0:
            sys.exit(p.returncode)

    if not args.skip_qa_scenarios:
        _log("=== 4) qa_browser_like_scenarios.py ===")
        p = subprocess.run(
            [
                str(VENV_PY),
                str(PROJECT / "scripts" / "qa_browser_like_scenarios.py"),
                "--gateway",
                gw,
                "--llamacpp-root",
                str(args.llamacpp_root),
            ],
            cwd=str(PROJECT),
            env=env,
        )
        if p.returncode != 0:
            sys.exit(p.returncode)

    _log("=== run_gateway_qa: ALL PASSED ===")


if __name__ == "__main__":
    main()
