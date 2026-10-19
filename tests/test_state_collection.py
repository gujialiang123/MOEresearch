"""Same-state diagnostics, durable sufficient stats and prompt bootstrap checks."""
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from moe_research.state_collection import (
    StateCollector, assert_matched_masks, paired_state_bootstrap,
    state_bootstrap, common_mask_interaction_bootstrap, _prepare, _evaluate,
)


def observe(collector, pid, path, index, native, calibrated, *, layer=0, token=None,
            common=None, calibration_hash="frozen_theta"):
    native = torch.as_tensor(native, dtype=torch.float64)
    calibrated = torch.as_tensor(calibrated, dtype=torch.float64)
    return collector.observe(prompt_id=pid, path=path, layer=layer,
                             decode_forward_index=index, token_id=index if token is None else token,
                             y8=native, u4=native * .5, z4=calibrated,
                             residual=torch.ones_like(native) * 10,
                             driver_hash=f"driver_{path}", calibration_hash=calibration_hash,
                             common_decode_limit=common)


class StateCollectionTests(unittest.TestCase):
    def test_scalar_rows_use_actual_residual_and_preserve_all_inputs(self):
        collector = StateCollector()
        y = torch.tensor([3., 4.])
        u = torch.tensor([1., 2.])
        z = torch.tensor([6., 8.])
        residual = torch.tensor([12., 16.])
        before = [v.clone() for v in (y, u, z, residual)]
        row = collector.observe(prompt_id=1, path="C", layer=0, decode_forward_index=1,
                                token_id=77, y8=y, u4=u, z4=z, residual=residual,
                                driver_hash="student", calibration_hash="theta")
        self.assertAlmostEqual(row["residual_relative_error"], .25, places=10)
        self.assertAlmostEqual(row["projection_error_signed"], 5., places=10)
        self.assertAlmostEqual(row["orthogonal_error_norm"], 0., places=10)
        for original, current in zip(before, (y, u, z, residual)):
            torch.testing.assert_close(original, current, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            collector.observe(prompt_id=1, path="C", layer=0, decode_forward_index=1,
                              token_id=77, y8=y, u4=u, z4=z, residual=residual,
                              driver_hash="student", calibration_hash="theta")

    def test_pooled_population_moments_and_prompt_equal_sensitivity(self):
        collector = StateCollector()
        arrays = [("short", [[1., 2.]], [[2., 2.]]),
                  ("long", [[3., 1.], [5., 7.], [9., 3.]], [[3., 2.], [8., 9.], [7., 4.]])]
        for pid, native, calibrated in arrays:
            for i, (y, z) in enumerate(zip(native, calibrated), 1):
                observe(collector, pid, "C", i, y, z)
            collector.finish_prompt(pid, "C")
        native = torch.tensor(arrays[0][1] + arrays[1][1], dtype=torch.float64)
        calibrated = torch.tensor(arrays[0][2] + arrays[1][2], dtype=torch.float64)
        expected_ratio = float(calibrated.norm(dim=-1).sum() / native.norm(dim=-1).sum())
        expected_moment = torch.sqrt(((calibrated.mean(0) - native.mean(0)).square().sum() +
            (calibrated.std(0, correction=0) - native.std(0, correction=0)).square().sum()) /
            (native.mean(0).square().sum() + native.std(0, correction=0).square().sum() + 1e-24))
        result = collector.summarize("C")
        self.assertAlmostEqual(result["E_norm"], abs(math.log(expected_ratio)), places=12)
        self.assertAlmostEqual(result["E_moment"], float(expected_moment), places=12)
        # Equal-prompt weighting represents each prompt as one distribution,
        # retaining within-prompt variance rather than averaging prompt stds.
        weighted_y = torch.cat((native[:1].repeat(3, 1), native[1:]))
        weighted_z = torch.cat((calibrated[:1].repeat(3, 1), calibrated[1:]))
        prompt_result = collector.summarize("C", weighting="prompt")
        ratio = float(weighted_z.norm(dim=-1).sum() / weighted_y.norm(dim=-1).sum())
        self.assertAlmostEqual(prompt_result["E_norm"], abs(math.log(ratio)), places=12)
        self.assertEqual(result["prompt_count"], 2)
        self.assertEqual(result["effective_position_count"], 4)

    def test_bootstrap_recomputes_ratios_instead_of_averaging_prompt_errors(self):
        collector = StateCollector()
        for pid, y, z in (("a", [1., 0.], [2., 0.]), ("b", [100., 0.], [100., 0.])):
            observe(collector, pid, "C", 1, y, z)
            observe(collector, pid, "D", 1, y, y)
            collector.finish_prompt(pid, "C")
            collector.finish_prompt(pid, "D")
        result = paired_state_bootstrap(collector, "C", "D", iters=200, seed=17, chunk_size=37)
        self.assertAlmostEqual(result["metrics"]["E_norm"]["difference"], math.log(102 / 101), places=12)
        self.assertGreater(abs(result["metrics"]["E_norm"]["difference"] - math.log(2) / 2), .3)
        self.assertAlmostEqual(result["metrics"]["E_norm"]["ci_low"], 0.)
        self.assertAlmostEqual(result["metrics"]["E_norm"]["ci_high"], math.log(2))
        repeat = paired_state_bootstrap(collector, "C", "D", iters=200, seed=17, chunk_size=200)
        self.assertEqual(result, repeat)
        self.assertFalse(result["calibration_parameters_refitted"])

    def test_matched_mask_validation_and_shared_prefix_keeps_full_suffix_separate(self):
        collector = StateCollector()
        for path, length in (("C", 3), ("D", 2)):
            for index in range(1, length + 1):
                observe(collector, "p", path, index, [1., index], [1., index], common=2)
            collector.finish_prompt("p", path)
        with self.assertRaises(ValueError):
            assert_matched_masks(collector, "C", "D")
        common = assert_matched_masks(collector, "C", "D", position_bin="common")
        self.assertTrue(common["mask_hash"])
        self.assertEqual(collector.summarize("C")["effective_position_count"], 3)
        self.assertEqual(collector.summarize("C", position_bin="common")["effective_position_count"], 2)
        mismatch = StateCollector()
        observe(mismatch, "p", "A", 1, [1., 2.], [1., 2.], token=3)
        observe(mismatch, "p", "B", 1, [1., 2.], [1., 2.], token=4)
        with self.assertRaises(ValueError):
            assert_matched_masks(mismatch, "A", "B")

    def test_update_comparison_requires_explicit_calibration_and_mask_flags(self):
        collector = StateCollector()
        observe(collector, "p", "NT", 1, [1., 2.], [1., 2.], token=3, calibration_hash="theta_nt")
        observe(collector, "p", "ST", 1, [1., 2.], [2., 3.], token=4, calibration_hash="theta_st")
        with self.assertRaises(ValueError):
            paired_state_bootstrap(collector, "ST", "NT", iters=10)
        result = paired_state_bootstrap(collector, "ST", "NT", iters=10,
                                        require_matched_masks=False, require_same_calibration=False)
        self.assertFalse(result["matched_masks"])
        self.assertGreater(result["metrics"]["E_moment"]["difference"], 0)

    def test_per_prompt_archives_roundtrip_moments_and_scalar_rows_without_hidden_states(self):
        scratch = Path(os.environ.get("TMPDIR", "/scratch/jgu7/tmp"))
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            collector = StateCollector(directory)
            for pid in ("a", "b"):
                for index in (1, 2):
                    observe(collector, pid, "A", index, [index, 1., 3.], [2., 3., index])
                saved = collector.finish_prompt(pid, "A")
                with np.load(saved, allow_pickle=False) as archive:
                    # Every vector is a sufficient statistic of hidden size;
                    # position arrays are integer indices, never activations.
                    for key in archive.files:
                        if key.endswith("_mean") or key.endswith("_m2"):
                            self.assertEqual(archive[key].shape, (3,))
                            self.assertEqual(archive[key].dtype, np.float64)
            restored = StateCollector.load(directory)
            self.assertEqual(restored.summarize("A"), collector.summarize("A"))
            self.assertEqual(len(list(Path(directory).glob("*.tokens.jsonl"))), 2)
            with self.assertRaises(ValueError):
                collector.finish_prompt("a", "A")

    def test_undefined_zero_vector_cosine_and_zero_ratio_are_reported(self):
        collector = StateCollector()
        row = observe(collector, "zero", "A", 1, [0., 0.], [0., 0.])
        self.assertIsNone(row["cos_u4_y8"])
        self.assertFalse(row["cos_defined"])
        result = collector.summarize("A")
        self.assertIsNone(result["E_norm"])
        self.assertEqual(result["status"], "undefined_zero_norm_ratio")
        self.assertEqual(result["E_moment"], 0.)

    def test_signed_layer_intervals_and_four_way_interaction_use_common_positions(self):
        collector = StateCollector()
        for pid in ("a", "b"):
            for path, scale in (("A", 1.), ("B", 2.), ("C", 4.), ("D", 1.)):
                # Text differs between native and student sources; paired text
                # stays identical, and the predeclared common position is 1.
                token = 10 if path in ("A", "B") else 20
                observe(collector, pid, path, 1, [1., 2.], [scale, 2 * scale], common=1, token=token)
                if path in ("C", "D"):
                    observe(collector, pid, path, 2, [2., 3.], [8., 12.], common=1, token=21)
                collector.finish_prompt(pid, path)
        single = state_bootstrap(collector, "B", position_bin="common", iters=50)
        signed = single["metrics"]["layer:0:signed_log_ratio"]
        self.assertAlmostEqual(signed["estimate"], math.log(2), places=12)
        self.assertEqual(signed["ci_low"], signed["ci_high"])
        interaction = common_mask_interaction_bootstrap(collector, a="A", b="B", c="C", d="D", iters=50)
        self.assertAlmostEqual(interaction["metrics"]["E_norm"]["estimate"], math.log(2), places=12)
        self.assertEqual(interaction["common_position_count"], 2)
        with self.assertRaises(ValueError):
            common_mask_interaction_bootstrap(collector, a="A", b="B", c="C", d="D",
                                               position_bin="all", iters=10)

    def test_explicit_zero_decode_marker_is_durable_and_never_imputes_moments(self):
        scratch = Path(os.environ.get("TMPDIR", "/scratch/jgu7/tmp"))
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            collector = StateCollector(directory)
            with self.assertRaises(ValueError):
                collector.finish_prompt("immediate_eos", "C")
            metadata = {"protocol_hash": "p", "driver_hash": "d", "calibration_hash": "c", "trace_hash": "t"}
            marker = collector.finish_prompt("immediate_eos", "C", allow_empty=True, empty_metadata=metadata)
            self.assertTrue(marker.name.endswith(".empty.json"))
            self.assertFalse(list(Path(directory).glob("*.npz")))
            loaded = StateCollector.load(directory)
            self.assertEqual(loaded.records, {})
            self.assertEqual(loaded.empty_records, {("immediate_eos", "C"): "trace_has_no_valid_decode_positions"})
            self.assertEqual(loaded.empty_metadata[("immediate_eos", "C")], metadata)
            self.assertIn(("immediate_eos", "C"), loaded.finished)
            with self.assertRaises(ValueError):
                loaded.finish_prompt("immediate_eos", "C", allow_empty=True)
            with self.assertRaises(ValueError):
                loaded.summarize("C")

    def test_interrupted_pair_is_not_finished_and_retry_preserves_uncommitted_files(self):
        scratch = Path(os.environ.get("TMPDIR", "/scratch/jgu7/tmp"))
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            collector = StateCollector(directory)
            observe(collector, "p", "C", 1, [1., 2.], [2., 3.])
            with patch("moe_research.state_collection._atomic_json_exclusive", side_effect=OSError("interrupted before commit")):
                with self.assertRaises(OSError):
                    collector.finish_prompt("p", "C")
            recovered = StateCollector.load(directory)
            self.assertFalse(recovered.finished)
            self.assertFalse(recovered.records)
            self.assertEqual(len(recovered.incomplete_archives), 1)
            recovered.output_dir = Path(directory)
            observe(recovered, "p", "C", 1, [1., 2.], [2., 3.])
            recovered.finish_prompt("p", "C")
            complete = StateCollector.load(directory)
            self.assertIn(("p", "C"), complete.finished)
            self.assertEqual(complete.incomplete_archives, [])
            self.assertTrue(list((Path(directory) / "uncommitted").glob("*/*.npz")))
            # A committed archive cannot be resumed while its token rows have
            # been truncated or otherwise modified.
            token_file = next(Path(directory).glob("*.tokens.jsonl"))
            token_file.write_text("")
            with self.assertRaises(ValueError):
                StateCollector.load(directory)

    def test_zero_count_prompts_stay_in_bootstrap_and_no_state_draw_is_undefined(self):
        collector = StateCollector()
        observe(collector, "active", "C", 1, [1., 2.], [2., 4.])
        collector.finish_prompt("active", "C")
        collector.finish_prompt("zero", "C", allow_empty=True)
        pack = _prepare(collector, "C", "all")
        self.assertEqual(pack["prompts"], ["active", "zero"])
        self.assertEqual(pack["active_prompts"], ["active"])
        for weighting in ("token", "prompt"):
            values, _ = _evaluate(pack, np.array([[2., 0.], [0., 2.]]), weighting, collector.protection)
            self.assertAlmostEqual(values["E_norm"][0], math.log(2))
            self.assertTrue(np.isnan(values["E_norm"][1]))
            self.assertTrue(np.isnan(values["E_moment"][1]))
        boot = state_bootstrap(collector, "C", iters=100)
        self.assertEqual(boot["prompt_count"], 2)
        self.assertEqual(boot["observed_prompt_count"], 1)
        self.assertEqual(boot["metrics"]["E_moment"]["status"], "undefined_bootstrap_draw")
        self.assertLess(boot["metrics"]["E_moment"]["finite_draws"], 100)


if __name__ == "__main__":
    unittest.main()
