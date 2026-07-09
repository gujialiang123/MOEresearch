#!/usr/bin/env python3
"""v7 config sweep: compare tuned configs on realistic agent workloads.

For each (model, config), launch a single-GPU sglang server then run
bench_serving on the two agentic datasets (mooncake toolagent + shared_prefix)
with a LARGE, saturating, IDENTICAL load so config differences are visible.

Configs come from our 2026-06-25 autotuning winners:
  - Qwen3-30B: triton/lpm family vs flashinfer_cutlass/fcfs family.
  - LFM2.5:    cookbook baseline (~tuned optimum) + chunked/fcfs variants.

Sampling: toolagent --num-prompts 5000 (21% of the 23,608-session trace) with a
fixed slowdown so every config sees the same arrival pattern; shared_prefix
32 groups x 32 prompts = 1024. Fixed across configs => fair comparison.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONDA_ENV = "/home/t-jialianggu/.conda/envs/sglang-dev"
PY = f"{CONDA_ENV}/bin/python"
GPU_ID = os.environ.get("V7_GPU", "1")
PORT = 31218
HOST = "127.0.0.1"
OUT_ROOT = REPO / "results" / "2026-07-09_v7_config_sweep"

# Per-model server configs (the "knobs" being compared).
MODELS = [
    {
        "name": "lfm2.5-8b-a1b", "path": "/data/hf/LFM2.5-8B-A1B",
        "configs": [
            {"name": "cookbook",     "args": ["--moe-runner-backend", "triton", "--max-running-requests", "32", "--chunked-prefill-size", "-1", "--schedule-policy", "lpm"]},
            {"name": "chunked8192",  "args": ["--moe-runner-backend", "triton", "--max-running-requests", "32", "--chunked-prefill-size", "8192", "--schedule-policy", "lpm"]},
            {"name": "fcfs",         "args": ["--moe-runner-backend", "triton", "--max-running-requests", "32", "--chunked-prefill-size", "-1", "--schedule-policy", "fcfs"]},
        ],
    },
    {
        "name": "qwen3-30b-a3b-bf16", "path": "/data/hf/models/Qwen3-30B-A3B-Instruct-2507",
        "configs": [
            {"name": "baseline_triton", "args": ["--moe-runner-backend", "triton",             "--max-running-requests", "32", "--chunked-prefill-size", "-1",   "--schedule-policy", "lpm"]},
            {"name": "tuned_prefill",   "args": ["--moe-runner-backend", "flashinfer_cutlass",  "--max-running-requests", "32", "--chunked-prefill-size", "-1",   "--schedule-policy", "fcfs"]},
            {"name": "tuned_chunked",   "args": ["--moe-runner-backend", "triton",             "--max-running-requests", "32", "--chunked-prefill-size", "8192", "--schedule-policy", "lpm"]},
        ],
    },
]

DATASETS = [
    {
        "name": "toolagent",
        "args": [
            "--dataset-name", "mooncake", "--mooncake-workload", "toolagent",
            "--num-prompts", "5000", "--mooncake-slowdown-factor", "0.1",
            "--max-concurrency", "64",
        ],
    },
    {
        "name": "shared_prefix",
        "args": [
            "--dataset-name", "generated-shared-prefix",
            "--gsp-num-groups", "32", "--gsp-prompts-per-group", "32",
            "--gsp-system-prompt-len", "2048", "--gsp-question-len", "128",
            "--gsp-output-len", "256",
            "--max-concurrency", "64",
        ],
    },
]

COMMON_SERVER = [
    "--mem-fraction-static", "0.85",
    "--context-length", "32768",
    "--trust-remote-code",
]


def env_for_run() -> dict:
    e = dict(os.environ)
    e["HOME"] = "/home/t-jialianggu"
    e["CUDA_VISIBLE_DEVICES"] = GPU_ID
    e["CUDA_HOME"] = CONDA_ENV
    e["PATH"] = f"{CONDA_ENV}/bin:" + e.get("PATH", "")
    e["LD_LIBRARY_PATH"] = f"{CONDA_ENV}/lib:" + e.get("LD_LIBRARY_PATH", "")
    e["TRITON_CACHE_DIR"] = "/tmp/sglang_triton_cache_v7cs"
    return e


def wait_ready(timeout=600) -> bool:
    url = f"http://{HOST}:{PORT}/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def launch_server(model: dict, cfg: dict, log_path: Path) -> subprocess.Popen:
    cmd = [
        PY, "-m", "sglang.launch_server",
        "--model-path", model["path"], "--tokenizer-path", model["path"],
        "--host", HOST, "--port", str(PORT), "--tensor-parallel-size", "1",
        *COMMON_SERVER, *cfg["args"],
    ]
    log = log_path.open("w")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            env=env_for_run(), start_new_session=True)


def kill_server(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(40):
        if proc.poll() is not None:
            return
        time.sleep(1)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_bench(model: dict, ds: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result_file = out_dir / "bench_serving_result.jsonl"
    log_file = out_dir / "bench.log"
    cmd = [
        PY, "-m", "sglang.bench_serving", "--backend", "sglang",
        "--host", HOST, "--port", str(PORT), "--model", model["path"],
        *ds["args"], "--output-file", str(result_file),
    ]
    print(f"    bench: {model['name']} × {ds['name']}", flush=True)
    with log_file.open("w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                       env=env_for_run(), timeout=3600)


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    for m in MODELS:
        for cfg in m["configs"]:
            tag = f"{m['name']} / {cfg['name']}"
            print(f"\n=== {tag} (GPU {GPU_ID}, elapsed {(time.time()-t_start)/60:.1f}m) ===", flush=True)
            srv_log = OUT_ROOT / m["name"] / cfg["name"] / "server.log"
            srv_log.parent.mkdir(parents=True, exist_ok=True)
            proc = launch_server(m, cfg, srv_log)
            try:
                if not wait_ready():
                    print(f"  SERVER FAILED: {tag} (see {srv_log})", flush=True)
                    continue
                print("  server ready", flush=True)
                for ds in DATASETS:
                    out_dir = OUT_ROOT / m["name"] / cfg["name"] / ds["name"]
                    try:
                        run_bench(m, ds, out_dir)
                    except Exception as e:
                        print(f"    ERROR {ds['name']}: {e}", flush=True)
            finally:
                kill_server(proc)
                time.sleep(12)
    print(f"\n=== v7 config sweep done in {(time.time()-t_start)/60:.1f} min ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
