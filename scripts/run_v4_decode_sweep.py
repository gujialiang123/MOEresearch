#!/usr/bin/env python3
"""v4 decode-stress sweep: 3 models × 3 configs × 5 regimes = 45 combos.

Each combo runs on GPU 6 with:
  - nvidia-smi sampler (real HBM util + VRAM)
  - sglang --show-time-cost + --log-requests (real decode step time)
  - 3 bench runs per regime

Output tree:
  results/2026-07-07_v4_decode_sweep/
    <model_name>/
      <config_name>/
        gpu_samples.csv
        hw_stats.json
        server.log
        summary.json
        spec.yaml

Estimated total time: 5-8 hours (model download-warm: 0, server startup:
~30s each × 9 unique (model,config), bench per combo ~3-5 min including
KV OOM retry attempts).

Usage:
    nohup python scripts/run_v4_decode_sweep.py > logs/v4_decode.log 2>&1 &
"""
from __future__ import annotations

import csv
import json
import re
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ────────────────────────────────────────────────────────────────
# Models to test
# ────────────────────────────────────────────────────────────────
MODELS = [
    {
        "name": "qwen3-30b-a3b-bf16",
        "server_config": "configs/qwen3_30b_a3b_bf16.yaml",
        "mfu_model": "configs/models/qwen3-30b-a3b-instruct-2507.yaml",
    },
    {
        "name": "qwen3-30b-a3b-fp8",
        "server_config": "configs/qwen3_30b_a3b_fp8.yaml",
        "mfu_model": "configs/models/qwen3-30b-a3b-fp8.yaml",
    },
    {
        "name": "qwen3-0.6b",
        "server_config": "configs/qwen3_0.6b.yaml",
        "mfu_model": "configs/models/qwen3-0.6b.yaml",
    },
]

# ────────────────────────────────────────────────────────────────
# Configs to test per model
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
        },
    },
    {
        "name": "v3_best_chunk8k",
        "flags": {
            "moe-runner-backend": "auto",
            "max-running-requests": 32,
            "chunked-prefill-size": 8192,
            "schedule-policy": "fcfs",
            "mem-fraction-static": 0.75,
        },
    },
    {
        "name": "big_batch_cap128",
        "flags": {
            "moe-runner-backend": "auto",
            "max-running-requests": 128,
            "chunked-prefill-size": 2048,
            "schedule-policy": "fcfs",
            "mem-fraction-static": 0.90,
        },
    },
]

GPU_ID = 6
PORT_BASE = 32000
OUT_ROOT = REPO / "results" / "2026-07-07_v4_decode_sweep"
REGIME_FILE = "regimes/decode_stress_sweep.yaml"


def load_server_config(path: str) -> dict:
    import yaml
    return yaml.safe_load((REPO / path).read_text())


def build_spec(model: dict, config: dict, port: int) -> dict:
    """Build one bench-spec that includes all overrides."""
    return {
        "submission_id": f"v4-{model['name']}-{config['name']}",
        "description": f"v4 decode sweep: {model['name']} × {config['name']}",
        "tags": ["v4", "decode-sweep", model["name"], config["name"]],
        "server": {
            "config": model["server_config"],
            "overrides": {
                "_gpu_id": GPU_ID,
                "port": port,
                "show-time-cost": True,
                "log-requests": True,
                "log-requests-level": 2,
                **config["flags"],
            },
            "conda_env": "sglang-dev",
            "health_url": f"http://127.0.0.1:{port}/health",
            "base_url":   f"http://127.0.0.1:{port}",
            "startup_timeout_s": 1500,
        },
        "regimes": {"file": REGIME_FILE},
        "bench": {
            "num_runs": 3,
            "reliable_stddev_pct": 10,
            "per_request_timeout_s": 900,
            "backend": "sglang",
        },
        "quality_gate": {"type": "sanity"},
    }


def start_sampler(out_csv: Path) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, str(REPO / "scripts/regime_study/gpu_sampler.py"),
         "--gpu", str(GPU_ID),
         "--interval", "0.5",
         "--out", str(out_csv)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    return proc


def stop_sampler(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def reduce_gpu_samples(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {"error": "no samples"}
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return {"error": "empty"}
    def _f(x):
        try: return float(x)
        except: return None
    fields = {
        "utilization.gpu": [], "utilization.memory": [],
        "memory.used": [], "power.draw": [],
        "clocks.current.sm": [], "clocks.current.memory": [],
    }
    for r in rows:
        for f, lst in fields.items():
            v = _f(r.get(f))
            if v is not None: lst.append(v)
    def sm(v):
        if not v: return {"n": 0}
        return {"n": len(v), "mean": round(statistics.mean(v), 2),
                "max": round(max(v), 2), "min": round(min(v), 2)}
    return {
        "n_samples": len(rows),
        "gpu_util_pct": sm(fields["utilization.gpu"]),
        "memory_util_pct": sm(fields["utilization.memory"]),
        "memory_used_mib": sm(fields["memory.used"]),
        "power_draw_w": sm(fields["power.draw"]),
        "sm_clock_mhz": sm(fields["clocks.current.sm"]),
        "memory_clock_mhz": sm(fields["clocks.current.memory"]),
    }


def parse_sglang_timings(log_path: Path) -> dict:
    """Extract per-batch decode step time from sglang gen throughput lines."""
    if not log_path.exists():
        return {"error": "no log"}
    from collections import defaultdict
    dc_pat = re.compile(
        r"Decode batch, #running-req: (\d+),.*gen throughput \(token/s\): ([\d.]+)"
    )
    decode_by_batch = defaultdict(list)
    for line in open(log_path):
        m = dc_pat.search(line)
        if m:
            n_run = int(m.group(1))
            gen_thru = float(m.group(2))
            if gen_thru > 0:
                step_ms = 1000.0 * n_run / gen_thru
                decode_by_batch[n_run].append((gen_thru, step_ms))

    out = {}
    for bs, samples in decode_by_batch.items():
        thrus = [s[0] for s in samples]
        if not thrus: continue
        # filter startup transients (throughput <50% of peak)
        peak = max(thrus)
        good = [(t, st) for t, st in samples if t > peak * 0.5]
        if not good: continue
        steps = [g[1] for g in good]
        thrs = [g[0] for g in good]
        out[f"batch_{bs}"] = {
            "n_samples": len(good),
            "median_step_ms": round(statistics.median(steps), 3),
            "p10_step_ms": round(sorted(steps)[max(0, len(steps)//10)], 3),
            "median_gen_throughput_tps": round(statistics.median(thrs), 1),
            "max_gen_throughput_tps": round(max(thrs), 1),
        }
    return out


def run_one(model: dict, config: dict, port: int) -> None:
    out_dir = OUT_ROOT / model["name"] / config["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = build_spec(model, config, port)
    spec_path = out_dir / "spec.yaml"
    import yaml
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    print(f"\n{'='*70}")
    print(f"[{model['name']} × {config['name']}] port={port}")
    print(f"{'='*70}", flush=True)

    sampler_csv = out_dir / "gpu_samples.csv"
    sampler = start_sampler(sampler_csv)

    try:
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, str(REPO / "harness/run_bench.py"),
             "--spec", str(spec_path),
             "--out-dir", str(out_dir),
             "--mfu-hardware", "configs/hardware/h200.yaml",
             "--mfu-model", model["mfu_model"]],
            capture_output=True, text=True, timeout=7200,
        )
        wall = time.time() - t0
        (out_dir / "harness.stdout.log").write_text(proc.stdout)
        (out_dir / "harness.stderr.log").write_text(proc.stderr)
        print(f"  Harness returned in {wall:.0f}s, exit={proc.returncode}", flush=True)
    finally:
        stop_sampler(sampler)

    gpu_stats = reduce_gpu_samples(sampler_csv)
    timings = parse_sglang_timings(out_dir / "server.log")

    (out_dir / "hw_stats.json").write_text(json.dumps({
        "model": model["name"],
        "config": config["name"],
        "gpu_id": GPU_ID,
        "config_flags": config["flags"],
        "gpu_stats": gpu_stats,
        "sglang_timings": timings,
    }, indent=2))
    print(f"  Wrote hw_stats.json ({len(timings)} batch sizes captured)", flush=True)


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    port_counter = 0
    total = len(MODELS) * len(CONFIGS)
    idx = 0
    t_start = time.time()

    for model in MODELS:
        for config in CONFIGS:
            idx += 1
            port = PORT_BASE + port_counter
            port_counter += 1
            print(f"\n>>> [{idx}/{total}] Starting {model['name']} × {config['name']} at "
                  f"{time.time() - t_start:.0f}s elapsed <<<", flush=True)
            try:
                run_one(model, config, port)
            except Exception as e:
                print(f"[ERROR] {model['name']} × {config['name']}: {e}",
                      flush=True)
                import traceback; traceback.print_exc()

    total_wall = time.time() - t_start
    print(f"\n=== ALL DONE in {total_wall/60:.1f} min ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
