#!/usr/bin/env python3
"""v5 NCU + kineto profiling pipeline.

For each (model, config, regime) combo:
  Phase 1 (kineto, ~5 min per combo):
    - Launch sglang server
    - Send warmup (32 requests)
    - Call /start_profile
    - Send regime bench (5 requests)
    - Call /stop_profile → get kineto trace
    - Parse trace to identify top-3 hot kernels by total GPU time
    - Store per-kernel: name, count, total_ms, mean_ms

  Phase 2 (NCU on top-3 hot kernels, ~10 min per combo):
    - Launch a MINIMAL script (using transformers, NOT sglang) that mimics
      the same workload shape (batch, seq, hidden)
    - Wrap in NCU with -k <hot_kernel_name> -c 20 --set full
    - Extract: tensor_pipe_active%, dram_throughput%, sm_active%,
      achieved_occupancy%, L2_hit_rate%, warp_stall reasons

Output tree:
  results/2026-07-08_v5_ncu/<model>/<config>/<regime>/
    kineto_trace.json.gz
    kineto_top_kernels.json
    ncu_<kernel_short_name>.csv  (per hot kernel)
    combo_summary.json

Total combos: 4 models × 3 configs × 5 regimes = 60 (or 3×3×5=45 if we drop Qwen3-0.6B)

Time budget: ~15 min per combo × 45 combos = ~11 hours. Given NCU's real
overhead we'll start with a subset (2 models × 2 configs × 3 regimes = 12
combos, ~3 hours) and expand if time permits.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NCU = "/opt/nvidia/nsight-compute/2026.2.1/ncu"
SGLANG_PY = "/home/t-jialianggu/.conda/envs/sglang-dev/bin/python"

# ────────────────────────────────────────────────────────────────
# Combos to profile
# ────────────────────────────────────────────────────────────────
MODELS = [
    {
        "name": "lfm2.5-8b-a1b",
        "server_config": "configs/lfm2.5_8b_a1b_v4.yaml",
        "mfu_model": "configs/models/lfm2.5-8b-a1b.yaml",
    },
    {
        "name": "qwen3-30b-a3b-bf16",
        "server_config": "configs/qwen3_30b_a3b_bf16.yaml",
        "mfu_model": "configs/models/qwen3-30b-a3b-instruct-2507.yaml",
    },
]

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

# Focus on 3 regimes that stress decode differently
REGIMES = [
    "R_decode_c1_out2k",       # pure batch=1 decode, memory-bound baseline
    "R_conc_ref",              # batch=32, our reference workload
    "R_decode_c128_out256",    # batch=128, biggest amortization
]

GPU_ID = 6
PORT_BASE = 33500

OUT_ROOT = REPO / "results" / "2026-07-08_v5_ncu"


def wait_for_health(host: str, port: int, timeout: int = 900) -> bool:
    deadline = time.time() + timeout
    url = f"http://{host}:{port}/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def http_post(url: str, body: dict = None, timeout: int = 30) -> dict:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def launch_sglang_server(server_config_yaml: Path, config_flags: dict,
                          port: int, gpu_id: int, out_dir: Path,
                          extra_env: dict = None) -> subprocess.Popen:
    """Launch sglang server with our config + flag overrides.
    Returns the subprocess handle."""
    import yaml
    base_config = yaml.safe_load(server_config_yaml.read_text())
    # Override with our config-specific flags
    for k, v in config_flags.items():
        base_config[k] = v
    base_config["port"] = port
    base_config["_gpu_id"] = gpu_id

    # Write resolved config to out_dir
    resolved = out_dir / "server_config_used.yaml"
    resolved.write_text(yaml.safe_dump(base_config, sort_keys=False))

    # Launch via launch_server.py
    launch_script = REPO / "scripts" / "launch_server.py"
    log_path = out_dir / "server.log"
    pid_path = out_dir / "server.pid"
    cmd = [
        SGLANG_PY, str(launch_script),
        "--config", str(resolved),
        "--log", str(log_path),
        "--pidfile", str(pid_path),
        "--conda-env", "sglang-dev",
    ]
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    print(f"  Launching sglang → {log_path}")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             env=env, start_new_session=True)
    return proc, pid_path


def stop_sglang_server(pid_path: Path) -> None:
    """Kill the sglang process group cleanly."""
    if not pid_path.exists():
        return
    try:
        pid = int(pid_path.read_text().strip())
        # Get process group
        try:
            pgid = os.getpgid(pid)
            print(f"  Killing pgid={pgid} (pid={pid})")
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            print(f"  pid={pid} already gone")
            return
        # Wait up to 15s for clean exit
        for _ in range(30):
            time.sleep(0.5)
            try:
                os.killpg(pgid, 0)  # signal 0 = check existence
            except ProcessLookupError:
                break
        else:
            print(f"  pgid={pgid} still alive after 15s → SIGKILL")
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    except Exception as e:
        print(f"  Warning: could not kill server: {e}")
    # Extra safety: kill any lingering sglang worker for our gpu
    time.sleep(3)


def kill_any_sglang_on_gpu(gpu_id: int) -> None:
    """Failsafe: kill any lingering sglang worker on this GPU."""
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid,cmd"], text=True, timeout=10)
        for line in out.splitlines():
            if "sglang.launch_server" in line and "run_v5" not in line and "t-vinka" not in line:
                # Also require it uses our GPU (CUDA_VISIBLE_DEVICES check is
                # tricky since env is set on parent). Just check port range.
                parts = line.strip().split(None, 1)
                if len(parts) == 2 and "port 33" in parts[1]:
                    pid = int(parts[0])
                    try:
                        pgid = os.getpgid(pid)
                        print(f"    Failsafe: killing lingering sglang pgid={pgid} pid={pid}")
                        os.killpg(pgid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
    except Exception as e:
        print(f"    kill_any_sglang failsafe error: {e}")
    time.sleep(3)


def send_warmup_bench(port: int, num_prompts: int = 32, model_path: str = "") -> None:
    """Send a warmup burst to steady-state the server."""
    # Use e2e-bench-runner's simple pattern
    bench = REPO / ".github/skills/e2e-bench-runner/impl/run_bench.py"
    # Write a temp regime with 32 short prompts
    regime_tmp = Path(f"/tmp/warmup_regime_{port}.yaml")
    regime_tmp.write_text("""regimes:
  warmup:
    num_prompts: 32
    prompt_words: 200
    max_new: 64
    concurrency: 8
""")
    cmd = [
        SGLANG_PY, str(bench),
        "--url", f"http://127.0.0.1:{port}",
        "--backend", "sglang",
        "--tag", "warmup",
        "--num-runs", "1",
        "--regimes-file", str(regime_tmp),
        "--out-dir", "/tmp/warmup_out",
    ]
    print(f"  Sending warmup burst (num_prompts={num_prompts})")
    subprocess.run(cmd, capture_output=True, timeout=600)


def start_profile(port: int, out_dir: Path) -> None:
    """Start sglang kineto profiling."""
    url = f"http://127.0.0.1:{port}/start_profile"
    body = {
        "output_dir": str(out_dir),
        "num_steps": 100,
        "activities": ["CPU", "GPU"],
        "with_stack": False,
        "record_shapes": False,
        "profile_by_stage": False,
    }
    try:
        r = http_post(url, body)
        print(f"  start_profile: {r}")
    except Exception as e:
        print(f"  start_profile failed: {e}")


def stop_profile(port: int) -> None:
    """Stop sglang kineto profiling and wait for trace to be written."""
    url = f"http://127.0.0.1:{port}/stop_profile"
    try:
        r = http_post(url, {})
        print(f"  stop_profile: {r}")
    except Exception as e:
        print(f"  stop_profile failed: {e}")
    # Wait for kineto to finish flushing to disk
    time.sleep(10)


def run_regime_bench(port: int, regime_id: str, out_dir: Path) -> dict:
    """Run one regime through e2e-bench-runner (3 runs)."""
    bench = REPO / ".github/skills/e2e-bench-runner/impl/run_bench.py"
    # Build a tiny regime yaml with only this regime (to avoid running all 5)
    import yaml
    full_yaml = yaml.safe_load(
        (REPO / "regimes/decode_stress_sweep.yaml").read_text())
    if regime_id not in full_yaml.get("regimes", {}):
        return {"error": f"regime {regime_id} not found in yaml"}
    single_regime = {"regimes": {regime_id: full_yaml["regimes"][regime_id]}}
    tmp_regime = out_dir / "single_regime.yaml"
    tmp_regime.write_text(yaml.safe_dump(single_regime))
    cmd = [
        SGLANG_PY, str(bench),
        "--url", f"http://127.0.0.1:{port}",
        "--backend", "sglang",
        "--tag", regime_id,
        "--num-runs", "3",
        "--regimes-file", str(tmp_regime),
        "--out-dir", str(out_dir),
    ]
    print(f"  Running regime {regime_id}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    (out_dir / "bench.stdout.log").write_text(proc.stdout)
    (out_dir / "bench.stderr.log").write_text(proc.stderr)
    # Load bench_summary.json
    bs_path = out_dir / "bench_summary.json"
    if bs_path.exists():
        return json.loads(bs_path.read_text())
    return {"error": "no bench_summary.json"}


def parse_kineto_trace(trace_path: Path, top_n: int = 5) -> list[dict]:
    """Load a kineto trace, sum GPU kernel time per-kernel-name, return top N.

    Kineto trace is a large JSON (or json.gz). It has an 'events' list where
    each GPU kernel is a dict with:
        {"cat": "kernel", "name": "...", "ts": ..., "dur": ..., ...}
    """
    if trace_path.suffix == ".gz":
        with gzip.open(trace_path, "rt") as f:
            data = json.load(f)
    else:
        with open(trace_path) as f:
            data = json.load(f)

    events = data.get("traceEvents", []) if isinstance(data, dict) else data
    kernel_stats = defaultdict(lambda: {"count": 0, "total_us": 0.0})
    for ev in events:
        if isinstance(ev, dict) and ev.get("cat") == "kernel":
            name = ev.get("name", "?")
            dur = float(ev.get("dur", 0))
            kernel_stats[name]["count"] += 1
            kernel_stats[name]["total_us"] += dur

    sorted_kernels = sorted(kernel_stats.items(),
                             key=lambda kv: -kv[1]["total_us"])
    top = []
    for name, st in sorted_kernels[:top_n]:
        top.append({
            "name": name,
            "count": st["count"],
            "total_us": round(st["total_us"], 1),
            "mean_us": round(st["total_us"] / max(st["count"], 1), 3),
        })
    return top


def profile_combo_kineto(model: dict, config: dict, regime: str,
                         port: int, out_dir: Path) -> dict:
    """Profile one (model, config, regime) with kineto to get hot kernels."""
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = out_dir / "kineto_traces"
    trace_dir.mkdir(exist_ok=True)

    # Set SGLANG_TORCH_PROFILER_DIR env so /start_profile writes here
    env_extra = {"SGLANG_TORCH_PROFILER_DIR": str(trace_dir)}

    proc, pid_path = launch_sglang_server(
        REPO / model["server_config"], config["flags"],
        port, GPU_ID, out_dir, extra_env=env_extra,
    )
    try:
        print(f"  Waiting for /health at port {port}...")
        if not wait_for_health("127.0.0.1", port, timeout=900):
            return {"error": "server didn't come up"}

        send_warmup_bench(port)

        # Now: start profile → run regime bench → stop profile
        start_profile(port, trace_dir)
        bench_result = run_regime_bench(port, regime, out_dir)
        stop_profile(port)

        # Find trace file
        traces = sorted(trace_dir.rglob("*.trace.json.gz")) \
            + sorted(trace_dir.rglob("*.trace.json"))
        if not traces:
            return {"error": "no trace produced"}
        trace_path = traces[-1]  # latest

        top_kernels = parse_kineto_trace(trace_path, top_n=5)
        (out_dir / "kineto_top_kernels.json").write_text(
            json.dumps({"top_kernels": top_kernels,
                        "trace_file": str(trace_path.name),
                        "bench_result": bench_result}, indent=2))
        return {"top_kernels": top_kernels, "bench_result": bench_result}
    finally:
        stop_sglang_server(pid_path)
        kill_any_sglang_on_gpu(GPU_ID)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only-kineto", action="store_true",
                    help="Skip NCU phase; only kineto discovery")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    port_ctr = 0
    total = len(MODELS) * len(CONFIGS) * len(REGIMES)
    idx = 0
    t_start = time.time()

    for model in MODELS:
        for config in CONFIGS:
            for regime in REGIMES:
                idx += 1
                port = PORT_BASE + port_ctr
                port_ctr += 1
                combo_dir = OUT_ROOT / model["name"] / config["name"] / regime
                print(f"\n>>> [{idx}/{total}] {model['name']} × {config['name']} × {regime}  "
                      f"(port={port}, elapsed={time.time()-t_start:.0f}s) <<<", flush=True)
                try:
                    result = profile_combo_kineto(model, config, regime, port, combo_dir)
                    print(f"  Kineto done: top kernel = "
                          f"{result.get('top_kernels', [{}])[0].get('name', '?')[:60]}",
                          flush=True)
                except Exception as e:
                    print(f"  ERROR: {e}", flush=True)
                    import traceback; traceback.print_exc()

    print(f"\n=== v5 kineto sweep done in {(time.time()-t_start)/60:.1f} min ===",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
