#!/usr/bin/env python3
"""v8: knob tuning sweep on REAL agent workloads (one model per GPU).

Sweeps chunked-prefill-size x max-running-requests (the knobs that mattered in
the v7 config sweep) on the real agent datasets, to find the true optimum on
realistic load rather than on synthetic regimes.

Run one instance per (model, GPU) so two models tune in parallel:
  V8_MODEL=lfm  V8_GPU=1 V8_PORT=31220 python scripts/run_v8_tuning_sweep.py
  V8_MODEL=qwen V8_GPU=2 V8_PORT=31221 python scripts/run_v8_tuning_sweep.py

Client offers a fixed load (--max-concurrency 128) so the server's
max-running-requests is the actual limiter being swept.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONDA_ENV = "/home/t-jialianggu/.conda/envs/sglang-dev"
PY = f"{CONDA_ENV}/bin/python"

MODEL_KEY = os.environ.get("V8_MODEL", "lfm")
GPU_ID = os.environ.get("V8_GPU", "1")
PORT = int(os.environ.get("V8_PORT", "31220"))
HOST = "127.0.0.1"
OUT_ROOT = REPO / "results" / "2026-07-09_v8_tuning"

MODELS = {
    "lfm":  {"name": "lfm2.5-8b-a1b",      "path": "/data/hf/LFM2.5-8B-A1B"},
    "qwen": {"name": "qwen3-30b-a3b-bf16", "path": "/data/hf/models/Qwen3-30B-A3B-Instruct-2507"},
}
MODEL = MODELS[MODEL_KEY]

# Knob grid (MoE backend fixed to triton = v7 winner; schedule=lpm).
CHUNKED = [2048, 4096, 8192, 16384]
MAXRUN = [32, 64, 128]

DATASETS = [
    {
        "name": "toolagent",
        "args": [
            "--dataset-name", "mooncake", "--mooncake-workload", "toolagent",
            "--num-prompts", "2000", "--mooncake-slowdown-factor", "0.1",
            "--max-concurrency", "128",
        ],
    },
    {
        "name": "shared_prefix",
        "args": [
            "--dataset-name", "generated-shared-prefix",
            "--gsp-num-groups", "32", "--gsp-prompts-per-group", "32",
            "--gsp-system-prompt-len", "2048", "--gsp-question-len", "128",
            "--gsp-output-len", "256",
            "--max-concurrency", "128",
        ],
    },
]

COMMON_SERVER = [
    "--mem-fraction-static", "0.85",
    "--context-length", "32768",
    "--trust-remote-code",
    "--moe-runner-backend", "triton",
    "--schedule-policy", "lpm",
]


def env_for_run() -> dict:
    e = dict(os.environ)
    e["HOME"] = "/home/t-jialianggu"
    e["CUDA_VISIBLE_DEVICES"] = GPU_ID
    e["CUDA_HOME"] = CONDA_ENV
    e["PATH"] = f"{CONDA_ENV}/bin:" + e.get("PATH", "")
    e["LD_LIBRARY_PATH"] = f"{CONDA_ENV}/lib:" + e.get("LD_LIBRARY_PATH", "")
    e["TRITON_CACHE_DIR"] = f"/tmp/sglang_triton_cache_v8_{MODEL_KEY}"
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


def launch_server(chunked: int, maxrun: int, log_path: Path) -> subprocess.Popen:
    cmd = [
        PY, "-m", "sglang.launch_server",
        "--model-path", MODEL["path"], "--tokenizer-path", MODEL["path"],
        "--host", HOST, "--port", str(PORT), "--tensor-parallel-size", "1",
        *COMMON_SERVER,
        "--chunked-prefill-size", str(chunked),
        "--max-running-requests", str(maxrun),
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


def run_bench(ds: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result_file = out_dir / "bench_serving_result.jsonl"
    log_file = out_dir / "bench.log"
    cmd = [
        PY, "-m", "sglang.bench_serving", "--backend", "sglang",
        "--host", HOST, "--port", str(PORT), "--model", MODEL["path"],
        *ds["args"], "--output-file", str(result_file),
    ]
    print(f"    bench: {ds['name']}", flush=True)
    with log_file.open("w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                       env=env_for_run(), timeout=3600)


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    combos = [(c, m) for c in CHUNKED for m in MAXRUN]
    for i, (chunked, maxrun) in enumerate(combos, 1):
        cfg_name = f"chunk{chunked}_cap{maxrun}"
        tag = f"{MODEL['name']} / {cfg_name}"
        print(f"\n=== [{i}/{len(combos)}] {tag} (GPU {GPU_ID}, {(time.time()-t_start)/60:.1f}m) ===", flush=True)
        cfg_dir = OUT_ROOT / MODEL["name"] / cfg_name
        cfg_dir.mkdir(parents=True, exist_ok=True)
        proc = launch_server(chunked, maxrun, cfg_dir / "server.log")
        try:
            if not wait_ready():
                print(f"  SERVER FAILED: {tag}", flush=True)
                continue
            print("  server ready", flush=True)
            for ds in DATASETS:
                try:
                    run_bench(ds, cfg_dir / ds["name"])
                except Exception as e:
                    print(f"    ERROR {ds['name']}: {e}", flush=True)
        finally:
            kill_server(proc)
            time.sleep(12)
    print(f"\n=== v8 tuning [{MODEL['name']}] done in {(time.time()-t_start)/60:.1f} min ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
