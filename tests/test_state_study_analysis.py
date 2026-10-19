"""Decision gates use frozen practical bounds and archived prompt resamples."""
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

import torch

from moe_research.state_collection import StateCollector
from moe_research.state_study_analysis import analyze_diagnostics


def make_archive(directory, scales, *, bad_mask=False):
    collector = StateCollector(directory)
    for pid in ("p0", "p1"):
        for path, scale in scales.items():
            theta = "theta_updated_" + path if path.startswith("M_") else "theta0"
            for index in (1, 2):
                native = torch.tensor([1. + index, 3. + int(pid[-1])], dtype=torch.float64)
                token = index + (100 if path in ("C", "D") else 0)
                if bad_mask and path == "D":
                    token += 1
                collector.observe(prompt_id=pid, path=path, layer=0,
                    decode_forward_index=index, token_id=token,
                    y8=native, u4=native * .5, z4=native * scale,
                    residual=native * 4, driver_hash="driver_" + path,
                    calibration_hash=theta, common_decode_limit=1)
            collector.finish_prompt(pid, path)


class StateStudyAnalysisTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(os.environ.get("TMPDIR", "/scratch/jgu7/tmp"))
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def analyze(self, scales, **kwargs):
        source = self.root / "raw"
        make_archive(source, scales, bad_mask=kwargs.pop("bad_mask", False))
        return analyze_diagnostics(source, self.root / "summary", iters=31,
                                   chunk_size=11, **kwargs)

    def test_lda_state_mismatch_gate_and_recomputable_bin_outputs(self):
        summary = self.analyze({"A": 1.01, "B": 1.03, "C": 1.20, "D": 1.01,
                                "M_NT": 1.08, "M_ST": 1.02})
        self.assertEqual(summary["status"], "STATE_MISMATCH_SUPPORTED")
        self.assertTrue(summary["gate_recalibration"])
        self.assertEqual(summary["chosen_mismatch_metric"], "E_moment")
        primary = summary["analyses"]["all"]["token"]["comparisons"]["C_minus_D"]
        self.assertAlmostEqual(primary["metrics"]["E_moment"]["estimate"], .19, places=12)
        self.assertIn("layer:0:signed_log_ratio", summary["analyses"]["1-32"]["token"]["paths"]["C"]["intervals"])
        self.assertEqual(summary["analyses"]["33-128"]["status"], "NO_OBSERVATIONS")
        self.assertEqual(summary["analyses"]["common"]["token"]["interaction"]["status"], "ok")
        update = summary["analyses"]["all"]["token"]["comparisons"]["M_ST_minus_M_NT_own_trajectories"]
        self.assertFalse(update["same_calibration_required"])
        self.assertFalse(update["same_mask_required"])
        self.assertLess(update["metrics"]["E_moment"]["estimate"], 0)
        with (self.root / "summary" / "diagnostic_summary.json").open() as handle:
            persisted = json.load(handle)
        self.assertEqual(persisted, summary)

    def test_negative_requires_intervals_inside_both_matching_and_effect_bounds(self):
        summary = self.analyze({"A": 1.01, "B": 1.01, "C": 1.02, "D": 1.02})
        self.assertEqual(summary["status"], "NEGATIVE")
        self.assertFalse(summary["gate_recalibration"])

    def test_native_failure_and_broken_pair_cannot_trigger_recalibration(self):
        summary = self.analyze({"A": 1.12, "B": 1.15, "C": 1.30, "D": 1.01})
        self.assertEqual(summary["status"], "INCONCLUSIVE")
        self.assertEqual(summary["decision"]["reason"], "calibration_heldout_native_not_valid")
        self.assertFalse(summary["gate_recalibration"])
        other = self.root / "different_raw"
        make_archive(other, {"A": 1.01, "B": 1.03, "C": 1.2, "D": 1.01}, bad_mask=True)
        broken = analyze_diagnostics(other, self.root / "broken_summary", iters=11)
        self.assertFalse(broken["gate_recalibration"])
        self.assertEqual(broken["decision"]["reason"], "primary_fixed_text_pair_not_estimable")

    def test_scalar_primary_and_derived_summary_history(self):
        summary = self.analyze({"A": 1.01, "B": 1.03, "C": 1.2, "D": 1.01}, family="scalar")
        self.assertEqual(summary["chosen_mismatch_metric"], "E_norm")
        self.assertAlmostEqual(summary["analyses"]["all"]["token"]["paths"]["C"]["intervals"]["E_norm"]["estimate"], math.log(1.2))
        analyze_diagnostics(self.root / "raw", self.root / "summary", family="scalar", iters=11)
        self.assertEqual(len(list((self.root / "summary").glob("diagnostic_summary.previous.*.json"))), 1)

    def test_all_immediate_eos_paths_report_no_state_evidence(self):
        source = self.root / "empty_raw"
        collector = StateCollector(source)
        for path in ("A", "B", "C", "D"):
            collector.finish_prompt("first_token_eos", path, allow_empty=True)
        summary = analyze_diagnostics(source, self.root / "empty_summary", iters=11)
        self.assertEqual(summary["status"], "INCONCLUSIVE")
        self.assertFalse(summary["gate_recalibration"])
        self.assertEqual(summary["decision"]["reason"], "no_full_trajectory_observations")
        self.assertEqual(summary["coverage"]["all"]["C"]["prompt_count"], 0)
        self.assertEqual(summary["coverage"]["all"]["C"]["zero_observation_prompt_count"], 1)
        self.assertEqual(summary["coverage"]["all"]["C"]["total_recorded_prompt_count"], 1)
        self.assertEqual(len(summary["zero_observation_markers"]), 4)

    def test_mixed_empty_prompts_are_resampled_not_dropped_and_gate_stays_inconclusive(self):
        source = self.root / "mixed_raw"
        make_archive(source, {"A": 1.01, "B": 1.03, "C": 1.2, "D": 1.01})
        collector = StateCollector(source)
        for path in ("A", "B", "C", "D"):
            collector.finish_prompt("zero", path, allow_empty=True)
        summary = analyze_diagnostics(source, self.root / "mixed_summary", iters=200,
                                      expected_prompt_ids=["p0", "p1", "zero"])
        self.assertEqual(summary["status"], "INCONCLUSIVE")
        self.assertFalse(summary["gate_recalibration"])
        result = summary["analyses"]["all"]["token"]["paths"]["C"]
        self.assertEqual(result["point"]["prompt_count"], 3)
        self.assertEqual(result["point"]["observed_prompt_count"], 2)
        self.assertEqual(result["intervals"]["E_moment"]["status"], "undefined_bootstrap_draw")
        self.assertTrue(summary["prompt_completeness"]["C"]["complete_expected_ids"])

    def test_gate_requires_B_and_all_expected_prompt_ids(self):
        source = self.root / "missing_b"
        make_archive(source, {"A": 1.01, "C": 1.2, "D": 1.01})
        summary = analyze_diagnostics(source, self.root / "missing_b_summary", iters=11,
                                      expected_prompt_ids=["p0", "p1"])
        self.assertFalse(summary["gate_recalibration"])
        self.assertEqual(summary["decision"]["reason"], "required_A_B_C_D_observations_missing")
        complete = self.root / "complete_raw"
        make_archive(complete, {"A": 1.01, "B": 1.03, "C": 1.2, "D": 1.01})
        wrong_ids = analyze_diagnostics(complete, self.root / "wrong_ids_summary", iters=11,
                                        expected_prompt_ids=["p0", "p1", "missing"])
        self.assertFalse(wrong_ids["gate_recalibration"])
        self.assertEqual(wrong_ids["decision"]["reason"], "four_path_prompt_universe_differs_from_expected_ids")
        self.assertEqual(wrong_ids["prompt_completeness"]["C"]["missing_ids"], ["missing"])

    def test_updated_only_paths_keep_comparison_but_never_trigger_core_gate(self):
        source = self.root / "updated_only"
        make_archive(source, {"M_NT": 1.08, "M_ST": 1.02})
        summary = analyze_diagnostics(source, self.root / "updated_only_summary", iters=31,
                                      expected_prompt_ids=["p0", "p1"])
        self.assertFalse(summary["gate_recalibration"])
        self.assertEqual(summary["status"], "INCONCLUSIVE")
        comparison = summary["analyses"]["all"]["token"]["comparisons"]["M_ST_minus_M_NT_own_trajectories"]
        self.assertEqual(comparison["status"], "ok")
        self.assertEqual(comparison["metrics"]["E_moment"]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
