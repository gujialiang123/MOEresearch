"""4-way MoE backend benchmark — one harness, callable with different URLs."""
import sys, json, time, random
import concurrent.futures
import requests

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30000"
TAG = sys.argv[2] if len(sys.argv) > 2 else "unnamed"
OUT_DIR = sys.argv[3] if len(sys.argv) > 3 else "/tmp"

# vLLM uses OpenAI-compatible endpoint
IS_VLLM = "vllm" in TAG.lower()

# Generate prompts (same seed = same prompts across runs)
random.seed(2026)
WORDS = ("machine learning artificial intelligence deep neural network "
         "training inference optimization performance benchmark profiling "
         "kernel implementation source code framework library backend "
         "transformer attention encoder decoder embedding tokenizer batch "
         "processing efficiency throughput latency memory bandwidth").split()


def make_prompts(n, words):
    return [" ".join(random.choice(WORDS) for _ in range(words)) for _ in range(n)]


def send_sglang(prompt, max_new_tokens=256):
    r = requests.post(f"{URL}/generate", json={
        "text": prompt,
        "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0.0, "ignore_eos": True}
    }, timeout=600)
    out = r.json()
    return out.get('text', ''), out.get('meta_info', {})


def send_vllm(prompt, max_new_tokens=256):
    # OpenAI completions API
    r = requests.post(f"{URL}/v1/completions", json={
        "model": "qwen3-30b-a3b-moe",
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
    }, timeout=600)
    out = r.json()
    text = out.get('choices', [{}])[0].get('text', '')
    return text, out.get('usage', {})


send_fn = send_vllm if IS_VLLM else send_sglang

# Warmup
print(f"[{TAG}] Warmup (4 reqs)...", flush=True)
for _ in range(4):
    _ = send_fn("hello world", max_new_tokens=32)

# Three regimes
results = {}
for regime_name, num_prompts, words_per_prompt, max_new, concurrency in [
    ("R_short", 8, 200, 64, 1),     # batch=1 latency
    ("R_medium", 16, 800, 256, 8),  # mid-batch (~R7-like)
    ("R_long", 8, 2000, 256, 16),   # large prompts
]:
    print(f"[{TAG}] Running {regime_name} (n={num_prompts}, words={words_per_prompt}, out={max_new}, conc={concurrency})...", flush=True)
    prompts = make_prompts(num_prompts, words_per_prompt)
    
    t0 = time.time()
    latencies = []
    out_tokens = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(send_fn, p, max_new) for p in prompts]
        for f in concurrent.futures.as_completed(futures):
            text, meta = f.result()
            ot = (meta.get('completion_tokens', None) 
                  if IS_VLLM 
                  else meta.get('completion_tokens', None))
            if ot is None:
                ot = max_new  # fallback estimate
            out_tokens.append(ot)
    wall = time.time() - t0
    
    total_out_tokens = sum(out_tokens)
    req_per_s = num_prompts / wall
    tokens_per_s = total_out_tokens / wall
    
    results[regime_name] = {
        "num_prompts": num_prompts, "prompt_words": words_per_prompt,
        "max_new": max_new, "concurrency": concurrency,
        "wall_s": wall, "req_per_s": req_per_s, "tokens_per_s": tokens_per_s,
        "total_out_tokens": total_out_tokens,
    }
    print(f"  → wall={wall:.1f}s  req/s={req_per_s:.3f}  tok/s={tokens_per_s:.1f}")

# Save
import os
os.makedirs(OUT_DIR, exist_ok=True)
out_path = f"{OUT_DIR}/bench_{TAG}.json"
with open(out_path, 'w') as f:
    json.dump({"tag": TAG, "url": URL, "results": results}, f, indent=2)
print(f"[{TAG}] Saved: {out_path}")
