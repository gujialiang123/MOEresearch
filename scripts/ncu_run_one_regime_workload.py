#!/usr/bin/env python3
"""Run one regime workload against the local sglang server. Used inside ncu_one_regime.sh."""
import sys, time, random, concurrent.futures, requests, yaml
from pathlib import Path

REGIME = sys.argv[1]
URL = "http://127.0.0.1:30000"

regimes = yaml.safe_load(
    open(Path(__file__).resolve().parent.parent / "regimes" / "qwen3_30b_moe_sglang_perf_sweep.yaml")
)["regimes"]
if REGIME not in regimes:
    print(f"unknown regime '{REGIME}'", file=sys.stderr)
    sys.exit(1)
r = regimes[REGIME]

random.seed(2026)
WORDS = ("machine learning artificial intelligence deep neural network training inference "
         "optimization performance benchmark profiling kernel implementation source code "
         "framework library backend transformer attention encoder decoder embedding tokenizer "
         "batch processing efficiency throughput latency memory bandwidth").split()

def mp(n, w):
    return [" ".join(random.choice(WORDS) for _ in range(w)) for _ in range(n)]

def send(p):
    return requests.post(f"{URL}/generate", json={
        "text": p,
        "sampling_params": {
            "max_new_tokens": r["max_new"], "temperature": 0.0, "ignore_eos": True
        }
    }, timeout=600).json()

# Warmup — give ncu several thousand kernel launches to skip past
for _ in range(5):
    send("hello world")

# Real workload
prompts = mp(r["num_prompts"], r["prompt_words"])
t0 = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=r["concurrency"]) as ex:
    list(ex.map(send, prompts))
print(f"{REGIME} done in {time.perf_counter()-t0:.2f}s", flush=True)
