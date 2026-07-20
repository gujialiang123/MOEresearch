"""Gain calibration for v30 (train/dev only). Collects, per (layer, phase,
decode-position bin), the empirical distribution of the full-renorm gain
g = 1/r = M_full/M_K at a target low-K, so we can build:
  - fixed_gain / layer_mean_gain: frozen mean gain per (layer,k)
  - position_bin_mean_gain: mean per (layer, position-bin)
  - shuffled_gain: sample a gain from the matching pool (breaks token correspondence)
  - clipped_gain quantiles: q90/q95 thresholds

ALL statistics come from a calibration split (GSM8K train); test tokens never
provide their own gain. Saved to JSON (+ pools for shuffling).
"""
from __future__ import annotations
import json, random
import torch
import torch.nn.functional as F

# decode-position bins per plan: 1-32 / 33-96 / 97+
def position_bin(step: int) -> int:
    if step <= 32:
        return 0
    if step <= 96:
        return 1
    return 2


class GainCalibrator:
    """Hooks MoE blocks to record 1/r gains under a fixed low-K teacher-forced pass."""

    def __init__(self, blocks, low_k: int, norm_topk_prob=True):
        self.blocks = list(blocks)
        self.low_k = low_k
        self.norm_topk_prob = norm_topk_prob
        # pools[(layer, bin)] = list of gains ; also track means
        self.pools = {}

    @torch.inference_mode()
    def record(self, model, tok, prompts, device, max_prefill_tokens=1024):
        """Teacher-force each prompt (prefill only) and record per-layer gains.
        (Prefill approximates the layer gain distribution; decode-position bins use
        prefill token index as a proxy since we teacher-force the whole text.)"""
        handles = []
        state = {"layer_gains": {}}

        def mk(idx):
            def hook(module, inp, out):
                hs = inp[0].view(-1, inp[0].shape[-1])
                rw = F.softmax(module.gate(hs), dim=1, dtype=torch.float)
                rwk, _ = torch.topk(rw, module.top_k, dim=-1)
                base = rwk / rwk.sum(-1, keepdim=True) if self.norm_topk_prob else rwk
                keep = (torch.arange(base.shape[-1], device=base.device) < self.low_k)
                M_full = base.sum(-1, keepdim=True).clamp_min(1e-20)
                M_K = (base * keep).sum(-1, keepdim=True).clamp_min(1e-20)
                inv_r = (M_full / M_K).squeeze(-1)  # [T]
                for pos, g in enumerate(inv_r.tolist()):
                    key = (idx, position_bin(pos))
                    self.pools.setdefault(key, []).append(g)
            return hook

        for i, b in enumerate(self.blocks):
            handles.append(b.register_forward_pre_hook(lambda m, a, idx=i: None))  # noop keep order
        # use forward hooks instead
        for h in handles:
            h.remove()
        handles = [b.register_forward_hook(mk(i)) for i, b in enumerate(self.blocks)]
        SUFFIX = "\nPlease reason step by step, and put your final answer after '#### '."
        for ex in prompts:
            text = tok.apply_chat_template([{"role": "user", "content": ex + SUFFIX}],
                                           tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt").to(device)["input_ids"][:, :max_prefill_tokens]
            model(ids)
        for h in handles:
            h.remove()
        return self

    def scalars(self):
        """fixed_gain / layer_mean_gain: mean gain per 'layer,k' (averaged over bins)."""
        by_layer = {}
        for (l, b), gs in self.pools.items():
            by_layer.setdefault(l, []).extend(gs)
        return {f"{l},{self.low_k}": round(sum(gs)/len(gs), 5) for l, gs in by_layer.items() if gs}

    def position_bin_means(self):
        return {f"{l},{b},{self.low_k}": round(sum(gs)/len(gs), 5) for (l, b), gs in self.pools.items() if gs}

    def quantiles(self, qs=(0.90, 0.95, 0.99)):
        allg = sorted(g for gs in self.pools.values() for g in gs)
        out = {}
        for q in qs:
            out[str(q)] = round(allg[min(len(allg)-1, int(q*len(allg)))], 5) if allg else 1.0
        return out

    def save(self, path):
        json.dump({
            "low_k": self.low_k,
            "scalars_layer_mean": self.scalars(),
            "position_bin_means": self.position_bin_means(),
            "quantiles": self.quantiles(),
            "pool_sizes": {f"{l},{b}": len(gs) for (l, b), gs in self.pools.items()},
        }, open(path, "w"), indent=2)

    def make_shuffled_provider(self, seed=0):
        """Returns fn(layer, phase, step, n, device) -> [n,1] gains sampled from the
        matching (layer, position-bin) pool. Breaks correspondence to the current token."""
        rng = random.Random(seed)
        pools = {k: v for k, v in self.pools.items() if v}
        def provider(layer_idx, phase, decode_step, n, device):
            key = (layer_idx, position_bin(decode_step))
            pool = pools.get(key) or pools.get((layer_idx, 0)) or [1.0]
            vals = [pool[rng.randrange(len(pool))] for _ in range(n)]
            return torch.tensor(vals, device=device, dtype=torch.float).unsqueeze(-1)
        return provider


def load_calibration(path):
    return json.load(open(path))
