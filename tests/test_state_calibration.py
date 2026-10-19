"""Numerical contract tests for decode calibration; run with unittest or pytest."""
import copy
import os
from pathlib import Path
import tempfile
import unittest

import torch

from moe_research.state_calibration import (
    CalibrationFitter, FrozenStateCalibration, NumericalProtection, StreamingMoments,
)


class StateCalibrationTests(unittest.TestCase):
    def test_population_moments_pool_tokens_and_merge_centered_statistics(self):
        x = torch.tensor([[1e9 + 1, 4], [1e9 + 2, 8], [1e9 + 6, 11],
                          [1e9 + 10, 19]], dtype=torch.float64)
        first = StreamingMoments().update(x[:1])
        second = StreamingMoments().update(x[1:])
        first.merge(second)
        self.assertEqual(first.count, 4)
        self.assertEqual(first.mean.dtype, torch.float64)
        torch.testing.assert_close(first.mean, x.mean(0), rtol=1e-15, atol=1e-7)
        torch.testing.assert_close(first.variance(), x.var(0, correction=0), rtol=1e-8, atol=1e-7)
        restored = StreamingMoments.from_dict(first.to_dict())
        torch.testing.assert_close(restored.m2, first.m2)
        # Combining prompts must include their between-mean variation.
        within = torch.stack([x[:1].var(0, correction=0), x[1:].var(0, correction=0)]).mean(0)
        self.assertFalse(torch.allclose(first.variance(), within))

    def test_scalar_is_ratio_of_norm_sums_and_postaggregation_fp32(self):
        y8 = torch.tensor([[3., 4.], [0., 10.]])
        u4 = torch.tensor([[0., 1.], [0., 4.]])
        fitter = CalibrationFitter("scalar")
        fitter.update(2, y8[:1], u4[:1], prompt_id="short")
        fitter.update(2, y8[1:], u4[1:], prompt_id="long")
        frozen = fitter.finalize({"split": "C_fit", "mask_hash": "frozen_positions"})
        self.assertEqual(frozen.parameters["2"]["scale"], 3.)
        x = torch.tensor([[.0053, 1.3359]], dtype=torch.bfloat16)
        expected = (x.float() * 3.).to(torch.bfloat16)
        torch.testing.assert_close(frozen.apply(2, x), expected, rtol=0, atol=0)
        self.assertIs(frozen.apply(987, x, "prefill"), x)
        with self.assertRaises(KeyError):
            frozen.apply(987, x)
        self.assertEqual(frozen.metadata["counts_per_prompt_layer"], {"short": {"2": 1}, "long": {"2": 1}})

    def test_lda_uses_full_renorm_branch_and_population_per_dimension_moments(self):
        y8 = torch.tensor([[1., 7., 2.], [5., 11., 2.], [9., 15., 2.]])
        full = torch.tensor([[-2., 1., 4.], [0., 2., 4.], [2., 3., 4.]])
        wrong_u4 = full * 13 + 7
        fitter = CalibrationFitter("lda")
        fitter.update(0, y8[:1], wrong_u4[:1], full[:1], prompt_id="a")
        fitter.update(0, y8[1:], wrong_u4[1:], full[1:], prompt_id="b")
        frozen = fitter.finalize()
        params = frozen.parameters["0"]
        torch.testing.assert_close(torch.tensor(params["mean_reduced"]), full.mean(0))
        torch.testing.assert_close(torch.tensor(params["std_reduced"]), full.std(0, correction=0))
        gain = y8.double().std(0, correction=0).float() / (
            full.double().std(0, correction=0).float() + frozen.protection.eps_lda_std)
        expected = ((full.float() - full.double().mean(0).float()) * gain + y8.double().mean(0).float())
        torch.testing.assert_close(frozen.apply(0, full), expected, rtol=0, atol=0)
        # A nonzero affine offset must still be absent during prefill.
        self.assertIs(frozen.output_transform(0, "prefill", 0, full), full)
        self.assertEqual(frozen.metadata["fit_diagnostics"]["0"]["reduced_zero_variance_dimensions"], 1)
        self.assertTrue(torch.isfinite(frozen.apply(0, full)).all())

    def test_hash_integrity_contract_and_no_raw_overwrite(self):
        fitter = CalibrationFitter("scalar")
        fitter.update(0, torch.ones(2, 3), torch.ones(2, 3) * .7)
        frozen = fitter.finalize({"source_driver": "native", "position_mask_hash": "m"})
        payload = frozen.to_dict()
        restored = FrozenStateCalibration.from_dict(payload)
        self.assertEqual(restored.hash, frozen.hash)
        changed = copy.deepcopy(payload)
        changed["parameters"]["0"]["scale"] += .001
        with self.assertRaises(ValueError):
            FrozenStateCalibration.from_dict(changed)
        changed = copy.deepcopy(payload)
        changed["ddof"] = 1
        with self.assertRaises(ValueError):
            FrozenStateCalibration.from_dict(changed)
        scratch = Path(os.environ.get("TMPDIR", "/scratch/jgu7/tmp"))
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path = Path(directory) / "calibration.json"
            frozen.save(path)
            self.assertEqual(FrozenStateCalibration.load(path).hash, frozen.hash)
            with self.assertRaises(FileExistsError):
                frozen.save(path)

    def test_invalid_fits_and_independent_epsilon_roles_fail_explicitly(self):
        with self.assertRaises(ValueError):
            CalibrationFitter("scalar").finalize()
        fitter = CalibrationFitter("scalar")
        fitter.update(0, torch.ones(1, 2), torch.zeros(1, 2))
        with self.assertRaises(ValueError):
            fitter.finalize()
        with self.assertRaises(ValueError):
            fitter.update(0, torch.ones(1, 2), torch.ones(1, 2), phase="prefill")
        with self.assertRaises(ValueError):
            NumericalProtection(eps_lda_std=0)
        protection = NumericalProtection(eps_lda_std=1e-5, eps_variance=0,
                                         eps_diagnostic_norm=1e-9, eps_diagnostic_moment=1e-18)
        self.assertNotEqual(protection.eps_lda_std, protection.eps_diagnostic_norm)


if __name__ == "__main__":
    unittest.main()
