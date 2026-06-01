#!/usr/bin/env python3
"""nvidia-smi periodic sampler. Append-only CSV.

Usage:
  python gpu_sampler.py --gpu 0 --interval 0.5 --out sampler.csv
  # Sends SIGTERM to stop cleanly; CSV is closed.
"""
from __future__ import annotations

import argparse
import csv
import signal
import subprocess
import sys
import time
from pathlib import Path


def query(gpu_id: int) -> dict | None:
    fields = [
        "timestamp",
        "memory.used", "memory.total", "memory.free",
        "utilization.gpu", "utilization.memory",
        "power.draw", "temperature.gpu",
        "clocks.current.sm", "clocks.current.memory",
    ]
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--id", str(gpu_id),
             f"--query-gpu={','.join(fields)}",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT, text=True, timeout=3,
        ).strip()
    except Exception:  # noqa: BLE001
        return None
    parts = [p.strip() for p in out.split(",")]
    if len(parts) != len(fields):
        return None
    return dict(zip(fields, parts))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    stop = {"flag": False}
    def _stop(_sig, _frm):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    with open(out, "w", newline="") as f:
        w = None
        while not stop["flag"]:
            row = query(args.gpu)
            if row is not None:
                row["t_unix"] = time.time()
                if w is None:
                    w = csv.DictWriter(f, fieldnames=list(row.keys()))
                    w.writeheader()
                w.writerow(row)
                f.flush()
            time.sleep(args.interval)
    print(f"[gpu_sampler] stopped, wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
