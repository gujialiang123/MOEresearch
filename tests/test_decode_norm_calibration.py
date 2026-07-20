"""test_decode_norm_calibration: the same-hidden dual-branch calibration forward
(a) reproduces the native y8 output exactly, (b) records ||yK|| equal to an
independent top-K no-renorm partial sum, and (c) yields s(l,K)=E||y8||/E||yK|| with
realized ratio == 1.0 on the calibration tokens."""
import os, sys, types
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from _toy import ToyMoE, native_forward, HDIM
from moe_research.decode_norm_calib import DecodeNormCalibrator


def _independent_yK_norm(block, hidden, k):
    hs = hidden.view(-1, HDIM)
    rw = F.softmax(block.gate(hs), dim=1, dtype=torch.float)
    rw, sel = torch.topk(rw, block.top_k, dim=-1)
    base = rw / rw.sum(-1, keepdim=True)
    final = torch.zeros_like(hs)
    mask = F.one_hot(sel, num_classes=block.num_experts).permute(2, 1, 0)
    for e in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
        idx, top_x = torch.where(mask[e].squeeze(0))
        keepm = idx < k
        if not keepm.any():
            continue
        cur = hs[None, top_x[keepm]].reshape(-1, HDIM)
        out = block.experts[e](cur) * base[top_x[keepm], idx[keepm], None].to(hs.dtype)
        final.index_add_(0, top_x[keepm], out.to(hs.dtype))
    return final.float().norm(dim=-1)


def test_calibration_forward_matches_native_and_partial():
    block = ToyMoE(); block._kp_layer_idx = 0
    cal = DecodeNormCalibrator([block], top_k=8, k_targets=[4, 6])
    hidden = torch.randn(1, 5, HDIM)
    ref8 = native_forward(block, hidden)
    yk4 = _independent_yK_norm(block, hidden, 4)
    yk6 = _independent_yK_norm(block, hidden, 6)

    calstate = {"_prompt_len": 2}   # 2 prefill positions, 3 decode positions
    fwd = cal._cal_forward(calstate)
    block.acc_ref = cal.acc; block.accd_ref = cal.accd
    out, _ = types.MethodType(fwd, block)(hidden)
    assert (out - ref8).abs().max().item() < 1e-5, "y8 not native-exact"

    # decode positions are 2,3,4 -> check recorded ||yK|| sums match independent calc
    dec_idx = [2, 3, 4]
    exp4 = yk4[dec_idx].sum().item(); exp6 = yk6[dec_idx].sum().item()
    got4 = cal.acc[(0, "decode", 4)][1]; got6 = cal.acc[(0, "decode", 6)][1]
    assert abs(exp4 - got4) < 1e-3, (exp4, got4)
    assert abs(exp6 - got6) < 1e-3, (exp6, got6)

    s = cal.scalars("decode")
    rr = cal.realized_ratio(s, "decode")
    for k in (4, 6):
        assert abs(rr[k] - 1.0) < 1e-3, (k, rr[k])
    print("PASS test_calibration_forward_matches_native_and_partial")


if __name__ == "__main__":
    test_calibration_forward_matches_native_and_partial()
