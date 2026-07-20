"""test_native_equivalence: KPolicy keep-all (K=native) reproduces the native MoE
output exactly, for every weight mode and both norm settings. Real-30B next-token
equivalence for policy (8,8) is additionally asserted by the run harness preflight;
here we prove the algebra on a toy so it runs in CI without big weights."""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
from _toy import ToyMoE, native_forward, attach, HDIM
from moe_research import k_policy as KP


def test_native_equivalence_all_modes():
    hidden = torch.randn(1, 6, HDIM)
    for norm in (True, False):
        block = ToyMoE(norm=norm)
        ref = native_forward(block, hidden)
        for wm in KP.WEIGHT_MODES:
            kwargs = {}
            if wm in ("calibrated_norm_match", "decode_norm_match", "fixed_gain", "position_bin_gain"):
                kwargs["calib_scalars"] = {}          # empty -> scalar defaults 1.0
            if wm == "clipped_gain":
                kwargs["gain_clip"] = 5.0
            if wm == "shuffled_gain":
                kwargs["gain_provider"] = None         # falls back to inv_r (==1 at keep-all)
            pol = KP.KPolicy(prefill_k=8, decode_k=8, weight_mode=wm, **kwargs)
            ctx = attach(block, pol)
            KP._PHASE.set("prefill")
            out, _ = block(hidden)
            err = (out - ref).abs().max().item()
            assert err < 1e-5, f"norm={norm} wm={wm} keep-all err={err}"
    print("PASS test_native_equivalence_all_modes")


if __name__ == "__main__":
    test_native_equivalence_all_modes()
