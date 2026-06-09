#!/usr/bin/env python3
"""Skill impl: nsys-capture.

Wraps `nsys profile` around a target command, waits for it to finish (or
times out), forces a flush, then immediately exports the .nsys-rep to
.sqlite. See ../SKILL.md for the contract.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 0
FALLBACK_NSYS = "/home/t-chendili/cuda/12.6/bin/nsys"
FLUSH_WAIT_S = 60
MIN_REP_BYTES = 1 * 1024 * 1024  # 1 MB


def find_nsys() -> str | None:
    p = shutil.which("nsys")
    if p:
        return p
    if Path(FALLBACK_NSYS).exists():
        return FALLBACK_NSYS
    return None


def get_nsys_version(nsys: str) -> str:
    try:
        out = subprocess.run([nsys, "--version"], capture_output=True, text=True, timeout=10)
        for line in out.stdout.splitlines():
            if "Version" in line or "version" in line:
                return line.strip()
        return out.stdout.strip().splitlines()[0] if out.stdout else "unknown"
    except Exception:
        return "unknown"


def fail(out_dir: Path, msg: str, stderr_tail: str = "") -> None:
    payload = {"schema_version": SCHEMA_VERSION, "ok": False, "error": msg}
    if stderr_tail:
        payload["stderr_tail"] = stderr_tail[-1024:]
    (out_dir / "nsys_capture.json").write_text(json.dumps(payload, indent=2))
    print(f"[nsys-capture] FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-cmd", required=True,
                    help="Shell command to run while profile is active")
    ap.add_argument("--duration-s", type=float, default=60.0,
                    help="Max time to let target run before SIGINT to nsys")
    ap.add_argument("--gpu-id", type=int, default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--extra-nsys", default="",
                    help="Extra args appended to `nsys profile`")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nsys = find_nsys()
    if not nsys:
        fail(out_dir, "nsys not found on PATH or fallback")
    version = get_nsys_version(nsys)

    rep_path = out_dir / "profile.nsys-rep"
    sqlite_path = out_dir / "profile.sqlite"

    nsys_argv = [
        nsys, "profile",
        "-t", "cuda,nvtx,osrt",
        "-s", "none",
        "--cuda-memory-usage=true",
        "--force-overwrite=true",
        "-o", str(out_dir / "profile"),
    ]
    if args.extra_nsys:
        nsys_argv.extend(args.extra_nsys.split())
    nsys_argv.extend(["sh", "-c", args.target_cmd])

    env = os.environ.copy()
    warnings: list[str] = []
    if args.gpu_id is not None:
        if env.get("CUDA_VISIBLE_DEVICES"):
            warnings.append(
                f"CUDA_VISIBLE_DEVICES was set to {env['CUDA_VISIBLE_DEVICES']}; overriding to {args.gpu_id}")
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    t_start = time.perf_counter()
    proc = subprocess.Popen(nsys_argv, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)

    target_exit = None
    try:
        try:
            proc.wait(timeout=args.duration_s + 30)
            target_exit = proc.returncode
        except subprocess.TimeoutExpired:
            warnings.append(f"target did not exit within duration_s+30; sending SIGINT")
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=FLUSH_WAIT_S)
                target_exit = proc.returncode
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
                target_exit = -9
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=FLUSH_WAIT_S)
        target_exit = proc.returncode

    elapsed = time.perf_counter() - t_start
    stderr_tail = (proc.stderr.read() if proc.stderr else "")[-2048:]

    # Wait for .nsys-rep flush
    for _ in range(FLUSH_WAIT_S):
        if rep_path.exists() and rep_path.stat().st_size >= MIN_REP_BYTES:
            break
        time.sleep(1)

    if not rep_path.exists():
        fail(out_dir, f"profile.nsys-rep not produced (target_exit={target_exit})", stderr_tail)
    rep_size = rep_path.stat().st_size
    if rep_size < MIN_REP_BYTES:
        fail(out_dir, f"profile.nsys-rep is only {rep_size} bytes; likely no GPU activity captured", stderr_tail)

    # Export to sqlite
    export = subprocess.run(
        [nsys, "export", "--type", "sqlite", "--force-overwrite=true",
         "--output", str(sqlite_path), str(rep_path)],
        capture_output=True, text=True, timeout=300,
    )
    if export.returncode != 0 or not sqlite_path.exists():
        fail(out_dir, "sqlite export failed", export.stderr)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "nsys_binary": nsys,
        "nsys_version": version,
        "gpu_id": args.gpu_id,
        "target_cmd": args.target_cmd,
        "target_exit_code": target_exit,
        "target_duration_s": elapsed,
        "files": {
            "nsys_rep": str(rep_path),
            "nsys_rep_size_mb": round(rep_size / 1e6, 2),
            "sqlite": str(sqlite_path),
            "sqlite_size_mb": round(sqlite_path.stat().st_size / 1e6, 2),
        },
        "warnings": warnings,
    }
    (out_dir / "nsys_capture.json").write_text(json.dumps(payload, indent=2))
    print(f"[nsys-capture] OK: {sqlite_path}")


if __name__ == "__main__":
    main()
