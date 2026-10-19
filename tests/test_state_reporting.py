"""Scientific reporting checks: pairing, censoring, failures and frozen selection."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from moe_research.state_reporting import (
    build_generation_summary, choose_baseline, paired_generation_bootstrap,
    render_generation_report, summarize_generations,
)


def row(index, arm="N", tokens=10, strict=True, tolerant=True, reason="eos", seconds=1.):
    return {"sample_id":index, "arm":arm, "question_hash":f"q{index}",
        "protocol_hash":"fixed", "parser_version":"p1", "arm_hash":arm,
        "generated_tokens":tokens, "max_new_tokens":128,
        "strict_correct":strict, "tolerant_correct":tolerant,
        "stop_reason":reason, "observer_enabled":False, "timing_debug_logits":False,
        "generation_seconds":seconds, "prefill_seconds":.1, "decode_seconds":seconds-.1,
        "peak_allocated_bytes":1000, "peak_reserved_bytes":2000,
        "repeated_4gram_fraction":.125, "has_answer_marker":True,
        "first_marker_position":tokens-2, "post_marker_tokens":1,
        "parser_disagreement":strict!=tolerant, "raw_output":"#### 42"}


class StateReportingTests(unittest.TestCase):
    def test_cap_last_eos_is_natural_and_failures_remain_in_accuracy(self):
        rows = [row(0,tokens=128), row(1,tokens=128,reason="max_new_tokens"),
                {"sample_id":2,"stop_reason":"oom"}]
        summary = summarize_generations(rows, expected_ids=[0,1,2,3])
        self.assertEqual(summary["strict_accuracy"], 2/3)
        self.assertEqual(summary["eos_rate"], 1/3)
        self.assertEqual(summary["administrative_cap_rate"], 1/3)
        self.assertEqual(summary["n_failed"], 1)
        self.assertEqual(summary["missing_ids"], [3])
        self.assertFalse(summary["complete_expected_ids"])
        self.assertEqual(summary["length"]["n"], 2)
        self.assertEqual(summary["length"]["mean"], 128)

    def test_missing_pairs_are_rejected_instead_of_intersected(self):
        result = paired_generation_bootstrap([row(0),row(1)], [row(0)], iters=100)
        self.assertEqual(result["status"], "BLOCKED_INCOMPLETE_PAIRS")
        self.assertFalse(result["formal_valid"])
        self.assertEqual(result["pairing"]["right_missing"], [1])
        self.assertEqual(result["metrics"], {})
        # Identical rows still cannot stand in for an incomplete expected sample plan.
        result = paired_generation_bootstrap([row(0)], [row(0)], expected_ids=[0,1], iters=100)
        self.assertFalse(result["formal_valid"])

    def test_prompt_pair_bootstrap_is_exact_for_constant_deltas_and_reproducible(self):
        left = [row(i,"L_N",tokens=20+i) for i in range(4)]
        right = [row(i,"M_N",tokens=10+i) for i in range(4)]
        one = paired_generation_bootstrap(left,right)
        two = paired_generation_bootstrap(left,right)
        self.assertEqual(one,two)
        self.assertTrue(one["formal_valid"])
        self.assertEqual(one["bootstrap_unit"], "prompt")
        self.assertEqual(one["metrics"]["mean_generated_tokens"]["mean"],10)
        self.assertEqual(one["metrics"]["mean_generated_tokens"]["ci95"],[10.,10.])
        self.assertEqual(one["metrics"]["strict_accuracy"]["ci95"],[0.,0.])

    def test_identity_mismatch_or_duplicate_rows_cannot_produce_formal_comparison(self):
        left, right = [row(0)], [row(0)]
        right[0]["question_hash"] = "another_problem"
        comparison = paired_generation_bootstrap(left,right,iters=100)
        self.assertEqual(comparison["status"], "BLOCKED_IDENTITY_MISMATCH")
        with self.assertRaises(ValueError):
            summarize_generations([row(0),row(0)])
        mixed = [row(0),row(1)]
        mixed[1]["protocol_hash"] = "another_protocol"
        self.assertIn("protocol_hash",summarize_generations(mixed)["invariant_conflicts"])

    def test_probe_timing_and_missing_measurements_are_not_silently_used(self):
        left, right = [row(0),row(1)], [row(0),row(1)]
        left[1]["observer_enabled"] = True
        left[1]["generation_seconds"] = 9999.
        result = paired_generation_bootstrap(left,right,iters=100)
        metric = result["metrics"]["mean_generation_seconds"]
        self.assertEqual(metric["status"], "BLOCKED_MISSING_MEASUREMENTS")
        self.assertEqual(metric["missing_ids"], [1])
        summary = summarize_generations(left)
        self.assertEqual(summary["generation_seconds"]["n"],1)
        self.assertEqual(summary["generation_seconds"]["mean"],1.)
        self.assertEqual(summary["peak_allocated_bytes"]["n"],1)
        self.assertEqual(result["metrics"]["mean_generated_tokens"]["status"],"OK")

    def test_reference_selection_uses_both_scores_length_and_fixed_tie_order(self):
        arms = {arm:[row(i,arm,tokens=20) for i in range(4)] for arm in ("F","R8","R16")}
        arms["F"][0]["tolerant_correct"] = False
        arms["R8"][0]["strict_correct"] = False
        # R16 wins min(strict,tolerant), although every strict-only F row is correct.
        summary = build_generation_summary(arms,expected_ids=range(4),bootstrap_iters=100)
        selection = choose_baseline(summary)
        self.assertEqual(selection["selected_reference"],"R16")
        self.assertEqual(selection["S0"],"L_N")
        with self.assertRaises(ValueError):
            choose_baseline(summary,dataset_role="D_validate")
        for rows in arms.values():
            for item in rows:item.update(strict_correct=True,tolerant_correct=True)
        arms["R8"] = [dict(item,generated_tokens=10) for item in arms["R8"]]
        self.assertEqual(choose_baseline(build_generation_summary(arms,bootstrap_iters=100))["selected_reference"],"R8")
        arms["R16"] = [dict(item,generated_tokens=10) for item in arms["R16"]]
        self.assertEqual(choose_baseline(build_generation_summary(arms,bootstrap_iters=100))["selected_reference"],"R8")
        mismatched = copy.deepcopy(arms)
        mismatched["R8"][0]["question_hash"] = "changed"
        with self.assertRaises(ValueError):
            choose_baseline(build_generation_summary(mismatched,bootstrap_iters=100))

    def test_report_reads_jsonl_limits_disagreements_and_contains_actual_numbers(self):
        rows = [row(i,"N",tokens=20,strict=False,tolerant=True) for i in range(25)]
        scratch = Path("/scratch/jgu7/tmp");scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path = Path(directory)/"N.jsonl"
            path.write_text("".join(json.dumps(r)+"\n" for r in rows))
            summary = build_generation_summary({"N":path},expected_ids=range(25),bootstrap_iters=100)
        self.assertEqual(len(summary["arms"]["N"]["parser_disagreement_examples"]),20)
        self.assertEqual(summary["arms"]["N"]["scoring_disagreement_count"],25)
        rendered = render_generation_report(summary,context={"模型":"fixture"})
        self.assertIn("25/25",rendered)
        self.assertIn("100.00%",rendered)
        self.assertIn("非劣",rendered)
        self.assertTrue(summary["all_expected_rows_present"])


if __name__ == "__main__":
    unittest.main()
