#!/usr/bin/env python3
"""v7: characterize REAL / agentic workloads via sglang.bench_serving.

For each model, launch a single-GPU sglang server (cookbook-style config) then
run bench_serving on two datasets that mimic realistic agent workloads:
  1. mooncake --mooncake-workload toolagent  (real tool-agent trace, FAST'25)
  2. generated-shared-prefix                 (long shared system prompt + short Q)

We ONLY collect sglang's own serving metrics (throughput / TTFT / TPOT / ITL /
input-output length distribution). No NCU here (server path is multi-process).
The goal is to extract representative (input_len, output_len, concurrency)
points that we can later back-fill into bench_one_batch + NCU (v6 methodology).
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
PORT = 31217
HOST = "127.0.0.1"
OUT_ROOT = REPO / "results" / "2026-07-09_v7_agentic"

MODELS = [
    {"name": "lfm2.5-8b-a1b", "path": "/data/hf/LFM2.5-8B-A1B"},
    {"name": "qwen3-30b-a3b-bf16", "path": "/data/hf/models/Qwen3-30B-A3B-Instruct-2507"},
]

# Datasets → bench_serving args. Keep prompt counts modest so each run is short.
DATASETS = [
    {
        "name": "toolagent",
        "args": [
            "--dataset-name", "mooncake",
            "--mooncake-workload", "toolagent",
            "--num-prompts", "200",
        ],
    },
    {
        "name": "shared_prefix",
        "args": [
            "--dataset-name", "generated-shared-prefix",
            "--gsp-num-groups", "8",
            "--gsp-prompts-per-group", "16",
            "--gsp-system-prompt-len", "2048",
            "--gsp-question-len", "128",
            "--gsp-output-len", "256",
        ],
    },
]

SERVER_ARGS = [
    "--mem-fraction-static", "0.85",
    "--chunked-prefill-size", "-1",
    "--schedule-policy", "lpm",
    "--max-running-requests", "32",
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
    e["TRITON_CACHE_DIR"] = "/tmp/sglang_triton_cache_v7"
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


def launch_server(model: dict, log_path: Path) -> subprocess.Popen:
    cmd = [
        PY, "-m", "sglang.launch_server",
        "--model-path", model["path"],
        "--tokenizer-path", model["path"],
        "--host", HOST, "--port", str(PORT),
        "--tensor-parallel-size", "1",
        *SERVER_ARGS,
    ]
    log = log_path.open("w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            env=env_for_run(), start_new_session=True)
    return proc


def kill_server(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(30):
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
        PY, "-m", "sglang.bench_serving",
        "--backend", "sglang",
        "--host", HOST, "--port", str(PORT),
        "--model", model["path"],
        *ds["args"],
        "--output-file", str(result_file),
    ]
    print(f"    bench_serving: {model['name']} × {ds['name']}", flush=True)
    with log_file.open("w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                       env=env_for_run(), timeout=1800)


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for m in MODELS:
        print(f"\n=== MODEL {m['name']} (GPU {GPU_ID}) ===", flush=True)
        srv_log = OUT_ROOT / f"{m['name']}_server.log"
        proc = launch_server(m, srv_log)
        try:
            if not wait_ready():
                print(f"  SERVER FAILED for {m['name']}, see {srv_log}", flush=True)
                continue
            print(f"  server ready", flush=True)
            for ds in DATASETS:
                out_dir = OUT_ROOT / m["name"] / ds["name"]
                try:
                    run_bench(m, ds, out_dir)
                except Exception as e:
                    print(f"    ERROR {ds['name']}: {e}", flush=True)
        finally:
            kill_server(proc)
            time.sleep(10)
    print("\n=== v7 agentic bench done ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
