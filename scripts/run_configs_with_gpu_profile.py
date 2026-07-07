#!/usr/bin/env python3
"""Run selected configs with real GPU sampling + TTFT/TPOT capture from sglang.

For each (config, regime) pair:
  1. Start sglang server with --enable-metrics --show-time-cost
  2. Start nvidia-smi sampler in background (0.5s interval)
  3. Run 3 bench iterations
  4. Stop sampler + server
  5. Parse server.log for per-request TTFT / decode time (sglang logs these)
  6. Aggregate gpu_samples.csv over the bench window

Output per (config, regime):
  results/2026-07-07_gpu_profiled/{config}/{regime}/
    gpu_samples.csv
    server.log (with TTFT / decode time entries)
    summary.json (usual harness output)
    hw_stats.json (peak/mean of GPU sampler over bench window)

Usage:
    python scripts/run_configs_with_gpu_profile.py
"""
from __future__ import annotations

import csv
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# ────────────────────────────────────────────────────────────────
# CONFIGS TO PROFILE (Top-3 from v3 spreadsheet)
# ────────────────────────────────────────────────────────────────
CONFIGS = [
    {
        "name": "cookbook_baseline",
        "flags": {
            "moe-runner-backend": "auto",
            "max-running-requests": 32,
            "chunked-prefill-size": -1,
            "schedule-policy": "lpm",
            "mem-fraction-static": 0.85,
            # "disable-cuda-graph" omitted → cg on
        },
    },
    {
        "name": "v3_top1_trial29",
        "flags": {
            "moe-runner-backend": "auto",
            "max-running-requests": 64,
            "chunked-prefill-size": 2048,
            "schedule-policy": "fcfs",
            "mem-fraction-static": 0.85,
        },
    },
    {
        "name": "v3_top3_trial16",
        "flags": {
            "moe-runner-backend": "auto",
            "max-running-requests": 32,
            "chunked-prefill-size": 8192,
            "schedule-policy": "fcfs",
            "mem-fraction-static": 0.75,
        },
    },
]

# ────────────────────────────────────────────────────────────────
# REGIMES TO PROFILE (representative subset)
# ────────────────────────────────────────────────────────────────
REGIMES = [
    "R_short_decode",         # decode dominant, small batch
    "R_concurrent_decode",    # decode dominant, big batch (32)
    "R_prompt_16k_c2_out128", # mixed
    "R_prompt_50k_c1_out64",  # prefill dominant
]

GPU_ID = 6
PORT_BASE = 31800
OUT_ROOT = REPO / "results" / "2026-07-07_gpu_profiled"


def build_spec(config_name: str, config_flags: dict, port: int) -> dict:
    """Assemble a full bench-spec dict for one config × regime run."""
    return {
        "submission_id": f"gpu-profile-{config_name}",
        "description": f"GPU-profiled run for config {config_name}",
        "tags": ["gpu-profile", "hbm", config_name],
        "server": {
            "config": "configs/lfm2.5_8b_a1b_v3_longctx.yaml",
            "overrides": {
                "_gpu_id": GPU_ID,
                "port": port,
                # Enable extra sglang telemetry
                "show-time-cost": True,
                "log-requests": True,
                "log-requests-level": 2,
                **config_flags,
            },
            "conda_env": "sglang-dev",
            "health_url": f"http://127.0.0.1:{port}/health",
            "base_url":   f"http://127.0.0.1:{port}",
            "startup_timeout_s": 900,
        },
        "regimes": {"file": "regimes/gpu_profile_subset.yaml"},
        "bench": {
            "num_runs": 3,
            "reliable_stddev_pct": 8,
            "per_request_timeout_s": 900,
            "backend": "sglang",
        },
        "quality_gate": {"type": "sanity"},
    }


def start_sampler(out_csv: Path) -> subprocess.Popen:
    """Start nvidia-smi sampler in background."""
    proc = subprocess.Popen(
        [sys.executable, str(REPO / "scripts/regime_study/gpu_sampler.py"),
         "--gpu", str(GPU_ID),
         "--interval", "0.5",
         "--out", str(out_csv)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)  # let it settle
    return proc


def stop_sampler(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def reduce_gpu_samples(csv_path: Path) -> dict:
    """Compute peak + mean stats from nvidia-smi CSV."""
    if not csv_path.exists():
        return {"error": "no samples file"}
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return {"error": "empty samples"}

    def _fnum(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    fields = {
        "utilization.gpu": [],
        "utilization.memory": [],
        "memory.used": [],
        "power.draw": [],
        "clocks.current.sm": [],
        "clocks.current.memory": [],
    }
    for r in rows:
        for f, lst in fields.items():
            v = _fnum(r.get(f))
            if v is not None:
                lst.append(v)

    def summarize(name, vals):
        if not vals:
            return {"n": 0}
        return {
            "n": len(vals),
            "mean": round(statistics.mean(vals), 2),
            "max": round(max(vals), 2),
            "min": round(min(vals), 2),
        }

    return {
        "n_samples": len(rows),
        "gpu_util_pct": summarize("gpu", fields["utilization.gpu"]),
        "memory_util_pct": summarize("dram", fields["utilization.memory"]),
        "memory_used_mib": summarize("mem", fields["memory.used"]),
        "power_draw_w": summarize("power", fields["power.draw"]),
        "sm_clock_mhz": summarize("sm_clk", fields["clocks.current.sm"]),
        "memory_clock_mhz": summarize("mem_clk", fields["clocks.current.memory"]),
    }


def parse_ttft_from_server_log(log_path: Path) -> dict:
    """Extract TTFT / decode timing from sglang server.log if it logged them.

    sglang with --show-time-cost logs lines like:
      [ ... ] prefill: 234.5ms, decode: 456.7ms
    or per-request:
      [ ... ] finished request, prefill: X ms, decode: Y ms, total: Z ms
    """
    if not log_path.exists():
        return {"error": "no server.log"}
    prefill_times = []
    decode_times = []
    ttft_times = []
    import re
    # Pattern variants sglang uses
    pf_pat = re.compile(r"prefill(?:_time)?[:\s]+([\d.]+)\s*ms", re.IGNORECASE)
    dc_pat = re.compile(r"decode(?:_time)?[:\s]+([\d.]+)\s*ms", re.IGNORECASE)
    ttft_pat = re.compile(r"ttft[:\s]+([\d.]+)\s*ms", re.IGNORECASE)

    with open(log_path) as f:
        for line in f:
            m = pf_pat.search(line)
            if m:
                prefill_times.append(float(m.group(1)))
            m = dc_pat.search(line)
            if m:
                decode_times.append(float(m.group(1)))
            m = ttft_pat.search(line)
            if m:
                ttft_times.append(float(m.group(1)))

    def stats(vals):
        if not vals:
            return {"n": 0}
        return {
            "n": len(vals),
            "mean_ms": round(statistics.mean(vals), 2),
            "p50_ms": round(statistics.median(vals), 2),
            "p99_ms": round(sorted(vals)[int(len(vals)*0.99)], 2) if len(vals)>1 else vals[0],
        }
    return {
        "prefill_ms": stats(prefill_times),
        "decode_ms": stats(decode_times),
        "ttft_ms": stats(ttft_times),
    }


def run_one(config_name: str, config_flags: dict, port: int) -> None:
    out_dir = OUT_ROOT / config_name
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = build_spec(config_name, config_flags, port)
    spec_path = out_dir / "spec.yaml"
    import yaml
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    print(f"\n{'='*60}\n[{config_name}] port={port}\n{'='*60}")

    # Start GPU sampler
    sampler_csv = out_dir / "gpu_samples.csv"
    print(f"  Starting GPU sampler → {sampler_csv}")
    sampler = start_sampler(sampler_csv)

    try:
        # Run harness (starts server + benches + tears down)
        print(f"  Running harness (this takes ~5-10 min)...")
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, str(REPO / "harness/run_bench.py"),
             "--spec", str(spec_path),
             "--out-dir", str(out_dir),
             "--mfu-hardware", "configs/hardware/h200.yaml",
             "--mfu-model", "configs/models/lfm2.5-8b-a1b.yaml"],
            capture_output=True, text=True,
            timeout=3600,
        )
        wall = time.time() - t0
        (out_dir / "harness.stdout.log").write_text(proc.stdout)
        (out_dir / "harness.stderr.log").write_text(proc.stderr)
        print(f"  Harness returned in {wall:.0f}s, exit={proc.returncode}")
    finally:
        stop_sampler(sampler)
        print(f"  Sampler stopped")

    # Reduce GPU samples
    gpu_stats = reduce_gpu_samples(sampler_csv)

    # Parse TTFT from server log
    ttft_stats = parse_ttft_from_server_log(out_dir / "server.log")

    (out_dir / "hw_stats.json").write_text(json.dumps({
        "config_name": config_name,
        "config_flags": config_flags,
        "gpu_id": GPU_ID,
        "gpu_stats": gpu_stats,
        "server_log_timing": ttft_stats,
    }, indent=2))
    print(f"  Wrote {out_dir / 'hw_stats.json'}")


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for i, cfg in enumerate(CONFIGS):
        port = PORT_BASE + i
        try:
            run_one(cfg["name"], cfg["flags"], port)
        except Exception as e:
            print(f"[ERROR] config {cfg['name']}: {e}")
            import traceback
            traceback.print_exc()
    print("\n=== ALL DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
