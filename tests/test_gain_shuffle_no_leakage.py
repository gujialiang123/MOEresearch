"""test_gain_shuffle_no_leakage: the shuffled-gain provider draws only from the
calibration pool and never from the current request's own tokens; and it matches by
(layer, decode-position-bin)."""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moe_research.gain_calibration import GainCalibrator, position_bin


def test_shuffled_gain_pool_only():
    cal = GainCalibrator(blocks=[], low_k=4)
    # inject a known calibration pool with sentinel values per (layer, bin)
    cal.pools = {(0, 0): [11.0, 12.0], (0, 1): [21.0], (0, 2): [31.0],
                 (3, 0): [41.0]}
    prov = cal.make_shuffled_provider(seed=0)
    allowed = {11.0, 12.0, 21.0, 31.0, 41.0}
    for step, exp_bin in ((1, 0), (40, 1), (200, 2)):
        g = prov(0, "decode", step, 5, torch.device("cpu"))
        assert g.shape == (5, 1)
        for v in g.flatten().tolist():
            assert v in allowed, f"gain {v} not from calibration pool"
            # must match the layer/bin pool
            assert v in cal.pools[(0, position_bin(step))], (v, step)
    print("PASS test_shuffled_gain_pool_only")


def test_position_bin_boundaries():
    assert position_bin(1) == 0 and position_bin(32) == 0
    assert position_bin(33) == 1 and position_bin(96) == 1
    assert position_bin(97) == 2 and position_bin(500) == 2
    print("PASS test_position_bin_boundaries")


if __name__ == "__main__":
    test_shuffled_gain_pool_only()
    test_position_bin_boundaries()
