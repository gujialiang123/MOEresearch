#!/usr/bin/env python3
"""v10: offered-load sweep — how much does higher concurrency recover?

Keep the best config fixed, keep ONE server per model (max-running-requests 256),
and vary the CLIENT offered load (--max-concurrency) on the real toolagent
workload. Measures throughput / achieved-concurrency / TTFT / TPOT vs offered
load, to quantify how much of the serving idle is recoverable by feeding more
requests (and where it plateaus + the throughput-vs-TBT tradeoff).

One model per GPU:
  V10_MODEL=lfm  V10_GPU=1 V10_PORT=31240 python scripts/run_v10_load_sweep.py
  V10_MODEL=qwen V10_GPU=2 V10_PORT=31241 python scripts/run_v10_load_sweep.py
"""
from __future__ import annotations
import json, os, signal, subprocess, time, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONDA = "/home/t-jialianggu/.conda/envs/sglang-dev"
PY = f"{CONDA}/bin/python"
MK = os.environ.get("V10_MODEL", "lfm")
GPU = os.environ.get("V10_GPU", "1")
PORT = int(os.environ.get("V10_PORT", "31240"))
HOST = "127.0.0.1"
OUT = REPO / "results" / "2026-07-14_v10_load_sweep"

MODELS = {
    "lfm":  {"name": "lfm2.5-8b-a1b",      "path": "/data/hf/LFM2.5-8B-A1B",                    "chunk": 4096},
    "qwen": {"name": "qwen3-30b-a3b-bf16", "path": "/data/hf/models/Qwen3-30B-A3B-Instruct-2507","chunk": 16384},
}
M = MODELS[MK]
CONC = [8, 16, 32, 64, 128, 256]  # client offered load levels


def env():
    e = dict(os.environ)
    e.update(HOME="/home/t-jialianggu", CUDA_VISIBLE_DEVICES=GPU, CUDA_HOME=CONDA,
             PATH=f"{CONDA}/bin:" + e.get("PATH", ""),
             LD_LIBRARY_PATH=f"{CONDA}/lib:" + e.get("LD_LIBRARY_PATH", ""),
             TRITON_CACHE_DIR=f"/tmp/tc_v10_{MK}")
    return e


def ready(timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://{HOST}:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    srv_log = OUT / f"{M['name']}_server.log"
    cmd = [PY, "-m", "sglang.launch_server", "--model-path", M["path"],
           "--tokenizer-path", M["path"], "--trust-remote-code", "--host", HOST,
           "--port", str(PORT), "--tensor-parallel-size", "1",
           "--mem-fraction-static", "0.85", "--chunked-prefill-size", str(M["chunk"]),
           "--schedule-policy", "lpm", "--max-running-requests", "256",
           "--context-length", "32768", "--moe-runner-backend", "triton"]
    proc = subprocess.Popen(cmd, stdout=srv_log.open("w"), stderr=subprocess.STDOUT,
                            env=env(), start_new_session=True)
    try:
        if not ready():
            print("server failed", flush=True); return 1
        print("server ready", flush=True)
        for c in CONC:
            d = OUT / M["name"] / f"conc{c}"
            d.mkdir(parents=True, exist_ok=True)
            # slowdown 0.02 => arrivals fast enough that --max-concurrency is the real limiter
            bcmd = [PY, "-m", "sglang.bench_serving", "--backend", "sglang",
                    "--host", HOST, "--port", str(PORT), "--model", M["path"],
                    "--dataset-name", "mooncake", "--mooncake-workload", "toolagent",
                    "--num-prompts", "1500", "--mooncake-slowdown-factor", "0.02",
                    "--max-concurrency", str(c),
                    "--output-file", str(d / "result.jsonl")]
            print(f"  conc={c} ...", flush=True)
            with (d / "bench.log").open("w") as f:
                subprocess.run(bcmd, stdout=f, stderr=subprocess.STDOUT, env=env(), timeout=1800)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            time.sleep(10)
        except ProcessLookupError:
            pass
    print(f"v10 [{M['name']}] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
