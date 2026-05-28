#!/usr/bin/env python3
"""Poll a SGLang server until it answers /health, /v1/models, or accepts TCP."""
from __future__ import annotations

import argparse
import socket
import sys
import time
import urllib.error
import urllib.request


def http_ok(url: str, timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 500
    except (urllib.error.URLError, ConnectionResetError, TimeoutError, OSError):
        return False


def tcp_ok(host: str, port: int, timeout: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--timeout", type=int, default=300, help="Total wait seconds.")
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    deadline = time.monotonic() + args.timeout
    last_status = ""

    while time.monotonic() < deadline:
        if http_ok(f"{base}/health"):
            print(f"[wait_ready] /health OK after {int(time.monotonic() - (deadline - args.timeout))}s")
            return 0
        if http_ok(f"{base}/v1/models"):
            print(f"[wait_ready] /v1/models OK")
            return 0
        if http_ok(f"{base}/get_server_info"):
            print(f"[wait_ready] /get_server_info OK")
            return 0
        if not tcp_ok(args.host, args.port):
            status = "tcp closed"
        else:
            status = "tcp open, http not ready"
        if status != last_status:
            print(f"[wait_ready] {status}")
            last_status = status
        time.sleep(args.interval)

    print(f"[wait_ready] TIMEOUT after {args.timeout}s", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
