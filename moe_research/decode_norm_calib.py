"""v29 decode-specific norm calibration (same-hidden dual-branch).

The OLD norm-match (scripts/calibrate_norm_match.py) estimated the per-layer scalar
    s(l,K) = E[||y8||] / E[||yK_no_renorm||]
on PREFILL tokens only, then applied it during DECODE. That confounds the mechanism:
prefill-token branch-norm statistics need not equal decode-token statistics.

This module fixes it. For each calibration prompt we (1) generate the native K8
greedy baseline, then (2) teacher-force the FULL sequence through the model exactly
once with a calibration forward on every MoE block that, from the SAME hidden state
h, forms both branches with NO extra expert compute:
    y8   = sum_{j in native top-8} base_j E_j(h)          (native output)
    yK   = sum_{j in top-K}       base_j E_j(h)           (no-renorm, dropped tail)
and accumulates ||y8|| and ||yK|| separately for PREFILL positions (pos < prompt_len)
and DECODE positions (pos >= prompt_len, further split into decode-position bins).

Scalars:
    s_phase(l,K) = E_phase[||y8||] / E_phase[||yK||]
are saved for phase in {prefill, decode}. Only calibration prompts feed this; no test
token ever provides its own scale.
"""
from __future__ import annotations
import json, types
import torch
import torch.nn.functional as F

from .k_policy import _position_bin


class DecodeNormCalibrator:
    def __init__(self, blocks, top_k: int, k_targets, norm_topk_prob=True):
        self.blocks = list(blocks)
        self.top_k = top_k
        self.k_targets = sorted(set(int(k) for k in k_targets))
        self.norm_topk_prob = norm_topk_prob
        # acc[(layer, phase, K)] = [sum||y8||, sum||yK||, count]; phase in {'prefill','decode'}
        self.acc = {}
        # per decode-bin: accd[(layer, bin, K)] = [sum||y8||, sum||yK||, count]
        self.accd = {}
        self._prompt_len = 0  # set per teacher-forced pass

    def _cal_forward(self, cal):
        top_k, k_targets, norm_topk_prob = self.top_k, self.k_targets, self.norm_topk_prob

        def forward(self, hidden_states):
            bsz, seqlen, hdim = hidden_states.shape
            hs = hidden_states.view(-1, hdim)
            router_logits = self.gate(hs)
            rw_full = F.softmax(router_logits, dim=1, dtype=torch.float)
            rw, selected = torch.topk(rw_full, top_k, dim=-1)  # [T, top_k]
            base = rw / rw.sum(-1, keepdim=True) if norm_topk_prob else rw  # native weights
            T = hs.shape[0]
            layer_idx = getattr(self, "_kp_layer_idx", -1)

            final = torch.zeros((T, hdim), dtype=hs.dtype, device=hs.device)   # y8
            finalK = {k: torch.zeros((T, hdim), dtype=hs.dtype, device=hs.device)
                      for k in k_targets}                                       # yK (no_renorm)
            expert_mask = F.one_hot(selected, num_classes=self.num_experts).permute(2, 1, 0).bool()
            hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for eidx in hit:
                elayer = self.experts[eidx]
                idx, top_x = torch.where(expert_mask[eidx].squeeze(0))  # idx=rank, top_x=token
                cur = hs[None, top_x].reshape(-1, hdim)
                out = elayer(cur)  # [n, hdim] raw expert output (no weight yet)
                w = base[top_x, idx, None]
                final.index_add_(0, top_x, (out * w).to(hs.dtype))
                for k in k_targets:
                    m = (idx < k)
                    if m.any():
                        finalK[k].index_add_(0, top_x[m], (out[m] * w[m]).to(hs.dtype))

            n8 = final.float().norm(dim=-1)  # [T]
            for pos in range(T):
                phase = "prefill" if pos < cal["_prompt_len"] else "decode"
                nb = _position_bin(pos - cal["_prompt_len"]) if phase == "decode" else -1
                for k in k_targets:
                    nk = finalK[k][pos].float().norm().item()
                    a = self.acc_ref.setdefault((layer_idx, phase, k), [0.0, 0.0, 0])
                    a[0] += n8[pos].item(); a[1] += nk; a[2] += 1
                    if phase == "decode":
                        d = self.accd_ref.setdefault((layer_idx, nb, k), [0.0, 0.0, 0])
                        d[0] += n8[pos].item(); d[1] += nk; d[2] += 1
            return final.reshape(bsz, seqlen, hdim), router_logits
        return forward

    @torch.inference_mode()
    def record(self, model, tok, prompts, device, max_new=512, gen_batch=16):
        SUFFIX = "\nPlease reason step by step, and put your final answer after '#### '."
        texts = [tok.apply_chat_template([{"role": "user", "content": p + SUFFIX}],
                 tokenize=False, add_generation_prompt=True) for p in prompts]
        # 1) native K8 greedy baseline (batched) to get the true decode trajectory
        eos_ids = tok.eos_token_id
        eos_set = set(eos_ids) if isinstance(eos_ids, list) else {eos_ids}
        seqs = []  # (prompt_len, full_ids)
        for bs in range(0, len(texts), gen_batch):
            enc = tok(texts[bs:bs+gen_batch], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(device)
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            in_len = enc["input_ids"].shape[1]
            am = enc["attention_mask"]
            for j in range(out.shape[0]):
                pl = int(am[j].sum().item())
                prompt_ids = enc["input_ids"][j][am[j].bool()].tolist()
                gen = out[j, in_len:].tolist()
                cut = len(gen)
                for pos, t in enumerate(gen):
                    if t in eos_set:
                        cut = pos + 1; break
                seqs.append((len(prompt_ids), prompt_ids + gen[:cut]))
        # 2) teacher-force each full sequence once with the calibration forward
        cal = {"_prompt_len": 0}
        fwd = self._cal_forward(cal)
        for b in self.blocks:
            b.acc_ref = self.acc; b.accd_ref = self.accd
            b._orig_fwd = b.forward
            b.forward = types.MethodType(fwd, b)
        try:
            for prompt_len, ids in seqs:
                cal["_prompt_len"] = prompt_len
                t = torch.tensor([ids], device=device)
                model(t)
        finally:
            for b in self.blocks:
                b.forward = b._orig_fwd
                del b._orig_fwd, b.acc_ref, b.accd_ref
        return self

    def scalars(self, phase="decode"):
        out = {}
        for (l, ph, k), (s8, sk, c) in self.acc.items():
            if ph != phase or c == 0 or sk <= 0:
                continue
            out[f"{l},{k}"] = round(s8 / sk, 6)
        return out

    def bin_scalars(self):
        out = {}
        for (l, b, k), (s8, sk, c) in self.accd.items():
            if c == 0 or sk <= 0:
                continue
            out[f"{l},{b},{k}"] = round(s8 / sk, 6)
        return out

    def realized_ratio(self, scalars, phase="decode"):
        """E[ s(l,K)*||yK|| ] / E[ ||y8|| ] per K, on the calibration tokens.
        By construction ~1.0 on the calibration set; used as a sanity check (a HELD-OUT
        dev diagnostic re-uses this class on dev prompts with the FROZEN scalars)."""
        num = {}; den = {}
        for (l, ph, k), (s8, sk, c) in self.acc.items():
            if ph != phase or c == 0:
                continue
            s = scalars.get(f"{l},{k}", 1.0)
            num[k] = num.get(k, 0.0) + s * sk
            den[k] = den.get(k, 0.0) + s8
        return {k: round(num[k] / den[k], 4) for k in num if den.get(k)}

    def save(self, path, extra=None):
        d = {
            "k_targets": self.k_targets,
            "decode_scalars": self.scalars("decode"),
            "prefill_scalars": self.scalars("prefill"),
            "decode_bin_scalars": self.bin_scalars(),
            "counts": {f"{l},{ph},{k}": c for (l, ph, k), (_, _, c) in self.acc.items()},
        }
        if extra:
            d.update(extra)
        json.dump(d, open(path, "w"), indent=2)
        return d


def load_scalars(path, key="decode_scalars"):
    return json.load(open(path))[key]
