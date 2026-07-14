#!/usr/bin/env python3
"""v10 roofline: decode end-to-end floor + per-kernel bandwidth roofline.

Two standard-roofline analyses to reframe the interim 'TBT headroom' metric:

(A) Decode end-to-end floor (first-principles, memory-bound):
    Each decode step must read (active weights + KV cache) from HBM.
    TBT_floor = bytes_per_step / HBM_peak_BW.
    Compare to measured TBT -> how close to the memory roofline we already are,
    and how much a perfect kernel could still gain.

(B) Per-kernel roofline position:
    achieved DRAM BW = DRAM% x peak; distance to roof = 1/(DRAM%/100).
    Uses v9 NCU DRAM% (already measured).

All inputs are real: model configs (safetensors/config.json), NCU DRAM%.
H200 peaks: HBM3e BW 4.8 TB/s, bf16 tensor ~989 TFLOP/s.
"""
import json, csv, glob
from collections import defaultdict

HBM_BW = 4.8e12          # H200 HBM3e peak bandwidth, bytes/s
BF16 = 2                 # bytes per bf16 element

def weight_bytes(path):
    idx = glob.glob(path + "/*.safetensors.index.json")
    if idx:
        return json.load(open(idx[0]))["metadata"]["total_size"]
    import os
    return sum(os.path.getsize(f) for f in glob.glob(path + "/*.safetensors"))

MODELS = {
    "LFM2.5-8B-A1B": {
        "path": "/data/hf/LFM2.5-8B-A1B",
        "layers": 24, "hidden": 2048, "kv_heads": 8, "head_dim": 2048//32,
        "dense_layers": 2, "attn_layers": 6,  # 'full_attention' count in layer_types
        "experts": 32, "experts_per_tok": 4, "moe_inter": 1792,
        # active (non-embedding) params per token: shared + routed-expert
        "total_params_b": 8.3, "active_params_b": 1.0,  # LFM2.5-8B-A1B = 8B total, 1B active
    },
    "Qwen3-30B-A3B": {
        "path": "/data/hf/models/Qwen3-30B-A3B-Instruct-2507",
        "layers": 48, "hidden": 2048, "kv_heads": 4, "head_dim": 128,
        "dense_layers": 0, "attn_layers": 48,
        "experts": 128, "experts_per_tok": 8, "moe_inter": 768,
        "total_params_b": 30.5, "active_params_b": 3.3,  # A3B = 3B active
    },
}

# Measured TBT (decode step, ms) from v9c clean bench (batch=32) and v9d.
MEASURED_TBT_MS = {
    "LFM2.5-8B-A1B": {"b1": 5.97, "b32": 5.95},   # v9c batch32 decode step
    "Qwen3-30B-A3B": {"b1": 8.71, "b32": 8.71},
}

def decode_floor(m, batch):
    """bytes/step read from HBM per decode step.

    - Dense/attention weights: read once per step (shared across batch).
    - MoE expert weights: a token activates experts_per_tok experts, but a
      BATCH of tokens collectively activates up to min(batch*ept, num_experts)
      distinct experts -> that many expert weights must be read per step.
    - KV cache: read per sequence in batch.
    """
    hidden = m["hidden"]; moe_inter = m["moe_inter"]
    layers = m["layers"]; dense = m.get("dense_layers", 0)
    moe_layers = layers - dense
    # dense/attention/shared active weights (approx): active minus MoE part
    exp_bytes = 3 * hidden * moe_inter * BF16          # one expert (gate/up/down)
    moe_1tok = m["experts_per_tok"] * exp_bytes * moe_layers
    dense_active = m["active_params_b"]*1e9*BF16 - moe_1tok   # non-MoE active
    if dense_active < 0: dense_active = m["active_params_b"]*1e9*BF16
    # MoE experts actually touched by the batch
    experts_touched = min(batch * m["experts_per_tok"], m["experts"])
    moe_batch = experts_touched * exp_bytes * moe_layers
    active_w = dense_active + moe_batch
    kv_per_tok_layer = 2 * m["kv_heads"] * m["head_dim"] * BF16
    ctx = 2700
    kv_bytes = kv_per_tok_layer * m["attn_layers"] * ctx * batch
    total = active_w + kv_bytes
    floor_s = total / HBM_BW
    return active_w, kv_bytes, total, floor_s

print("="*78)
print("(A) DECODE END-TO-END ROOFLINE FLOOR  (H200 HBM 4.8 TB/s, memory-bound)")
print("="*78)
for name, m in MODELS.items():
    wb = weight_bytes(m["path"])
    print(f"\n### {name}  (weights on disk {wb/1e9:.1f} GB, active {m['active_params_b']}B)")
    for batch in [1, 32]:
        aw, kv, tot, floor = decode_floor(m, batch)
        floor_ms = floor*1e3
        meas = MEASURED_TBT_MS[name][f"b{batch}"] if batch in (1,32) else None
        print(f"  batch={batch:>2}: active-W {aw/1e9:.1f}GB + KV {kv/1e9:.2f}GB = {tot/1e9:.1f}GB/step")
        print(f"           roofline floor = {floor_ms:.2f} ms/step (bytes/step / 4.8TB/s)")
        if meas:
            print(f"           measured TBT   = {meas:.2f} ms/step  ->  {meas/floor_ms:.2f}x above floor "
                  f"(=> up to {meas/floor_ms:.2f}x faster if it hit the memory roof)")

print("\n" + "="*78)
print("(B) PER-KERNEL BANDWIDTH ROOFLINE  (from v9 NCU DRAM%, decode b=32)")
print("="*78)
try:
    rows = list(csv.DictReader(open("results/consolidated_v9_ncu.csv")))
    for r in rows:
        for k in ["dram","sm","dur"]:
            try: r[k]=float(r[k])
            except: r[k]=None
    def short(k):
        if k.startswith("nvjet"): return "nvjet_gemm"
        if "fused_moe" in k or "FusedMoe" in k: return "fused_moe"
        if "flash" in k.lower() and "cutlass" in k.lower(): return "flash_attn"
        if "conv1d" in k: return "causal_conv1d"
        if "RMSNorm" in k: return "RMSNorm"
        return k[:16]
    for model in ["lfm2.5-8b-a1b","qwen3-30b-a3b-bf16"]:
        sub=[r for r in rows if r["model"]==model and r["reg"]=="agent_decode_b32" and r["dram"] is not None and r["dur"]]
        best={}
        for r in sub:
            s=short(r["kernel"])
            if s not in best or r["dur"]>best[s]["dur"]: best[s]=r
        print(f"\n### {model} decode b=32")
        print(f"  {'kernel':14}{'DRAM%':>7}{'achieved BW':>13}{'dist to roof':>13}")
        for s,r in sorted(best.items(),key=lambda x:-x[1]["dur"])[:5]:
            bw = r["dram"]/100*4.8
            dist = 100/r["dram"] if r["dram"]>0 else float('inf')
            print(f"  {s:14}{r['dram']:>7.1f}{bw:>10.2f}TB/s{dist:>11.2f}x")
except FileNotFoundError:
    print("  (v9 NCU CSV not found)")
