"""Finite-binomial quality bounds and paired cost ratios need distinct gates."""
import json
import math
import unittest

from scripts.state_study_positive_gate import (
    QUALITY_METHOD,
    TOKEN_METHOD,
    _clopper_pearson_lower,
    _clopper_pearson_upper,
    paired_relative_token_reduction,
    quality_superiority,
)


class ExactQualityTests(unittest.TestCase):
    def test_nist_binomial_inversion_example_and_pair_counts(self):
        # NIST's N=20, k=4, 90% exact interval has endpoints .071354/.401029.
        st = [True] * 4 + [False] * 4 + [True] * 6 + [False] * 6
        ctrl = [False] * 4 + [True] * 4 + [True] * 6 + [False] * 6
        result = quality_superiority(st, ctrl, family_alpha=.10,
                                     family_comparisons=1)
        self.assertEqual(result["counts"], {
            "wins": 4, "losses": 4, "both_correct": 6, "both_wrong": 6})
        self.assertAlmostEqual(result["win_probability_lower"], .0713538843, places=9)
        self.assertAlmostEqual(result["loss_probability_upper"], .4010281173, places=9)
        self.assertEqual(result["estimate"], 0)
        self.assertLess(result["exact_lower_bound"], 0)
        self.assertEqual(result["status"], "INCONCLUSIVE")

    def test_no_discordance_is_never_quality_preservation(self):
        for scores in ([True] * 512, [False] * 512, [True, False] * 256):
            result = quality_superiority(scores, scores)
            self.assertEqual(result["wins"], 0)
            self.assertEqual(result["losses"], 0)
            self.assertEqual(result["estimate"], 0)
            self.assertLess(result["exact_lower_bound"], 0)
            self.assertEqual(result["status"], "INCONCLUSIVE")
            self.assertIsNone(result["quality_noninferiority_margin"])
            self.assertAlmostEqual(result["per_tail_alpha"], .00625)

    def test_all_wins_analytic_boundary_and_sufficient_sample(self):
        tiny = quality_superiority([True], [False])
        self.assertAlmostEqual(tiny["win_probability_lower"], .00625)
        self.assertAlmostEqual(tiny["loss_probability_upper"], .99375)
        self.assertEqual(tiny["status"], "INCONCLUSIVE")
        large = quality_superiority([True] * 2000, [False] * 2000)
        self.assertEqual(large["status"], "SUPERIORITY_ESTABLISHED")
        self.assertTrue(math.isfinite(large["exact_lower_bound"]))
        expected = 2 * .00625 ** (1 / 2000) - 1
        self.assertAlmostEqual(large["exact_lower_bound"], expected, places=13)

    def test_more_comparisons_produces_a_more_conservative_bound(self):
        st = [True] * 30 + [False] * 10
        ctrl = [False] * 30 + [True] * 10
        one = quality_superiority(st, ctrl, family_comparisons=1)
        four = quality_superiority(st, ctrl, family_comparisons=4)
        self.assertLess(four["exact_lower_bound"], one["exact_lower_bound"])
        self.assertEqual(four["method"], QUALITY_METHOD)
        self.assertEqual(four["ci_unit"], "accuracy_fraction")

    def test_cp_tail_coverage_on_small_binomial_sample_space(self):
        # Enumerate all observations: an exact one-sided bound must fail with
        # probability no greater than its allocated tail, including boundaries.
        n, tail = 6, .025
        lower = [_clopper_pearson_lower(k, n, tail) for k in range(n + 1)]
        upper = [_clopper_pearson_upper(k, n, tail) for k in range(n + 1)]
        for p in (0, .001, .1, .3, .5, .8, .999, 1):
            probabilities = [math.comb(n, k) * p ** k * (1-p) ** (n-k)
                             for k in range(n + 1)]
            lower_failure = sum(prob for k, prob in enumerate(probabilities)
                                if lower[k] > p)
            upper_failure = sum(prob for k, prob in enumerate(probabilities)
                                if upper[k] < p)
            self.assertLessEqual(lower_failure, tail + 1e-14)
            self.assertLessEqual(upper_failure, tail + 1e-14)

    def test_validation_empty_and_json_metadata(self):
        empty = quality_superiority([], [])
        self.assertEqual(empty["status"], "INCONCLUSIVE")
        self.assertIsNone(empty["exact_lower_bound"])
        self.assertEqual(empty["reason"], "no_paired_observations")
        json.dumps(empty, allow_nan=False)
        for left, right in (([True], []), (["False"], [False]), ([.5], [False])):
            with self.assertRaises(ValueError):
                quality_superiority(left, right)
        for kwargs in ({"family_alpha": 0}, {"family_alpha": float("nan")},
                       {"family_comparisons": 0}, {"family_comparisons": True}):
            with self.assertRaises(ValueError):
                quality_superiority([True], [False], **kwargs)


class PairedTokenRatioTests(unittest.TestCase):
    def test_ratio_of_sums_does_not_average_prompt_ratios(self):
        result = paired_relative_token_reduction([0, 10], [1, 100], iters=200,
                                                 seed=17)
        self.assertAlmostEqual(result["estimate"], 1 - 10 / 101)
        self.assertNotAlmostEqual(result["estimate"], (1 + .9) / 2)
        self.assertAlmostEqual(result["lower_bound"], .9)
        self.assertAlmostEqual(result["upper_bound"], 1)
        self.assertEqual(result["per_tail_alpha"], .0125)
        self.assertEqual(result["bootstrap_unit"], "paired_prompt")
        self.assertEqual(result["method"], TOKEN_METHOD)

    def test_pairing_is_preserved_with_heterogeneous_prompt_lengths(self):
        result = paired_relative_token_reduction([1, 100], [2, 200],
                                                 iters=97, seed=10)
        self.assertEqual(result["estimate"], .5)
        self.assertEqual(result["ci_low"], .5)
        self.assertEqual(result["ci_high"], .5)
        self.assertEqual(result["undefined_draws"], 0)
        self.assertEqual(result["valid_draws"], 97)

    def test_boundary_and_fixed_seed_reproducibility(self):
        reduction = paired_relative_token_reduction([95], [100], iters=3)
        self.assertGreaterEqual(reduction["lower_bound"], .05)
        self.assertAlmostEqual(reduction["lower_bound"], .05)
        neutral = paired_relative_token_reduction([2, 10], [2, 10], iters=31)
        self.assertEqual(neutral["ci_low"], 0)
        self.assertEqual(neutral["ci_high"], 0)
        first = paired_relative_token_reduction([4, 20, 7], [5, 23, 30], iters=41)
        second = paired_relative_token_reduction([4, 20, 7], [5, 23, 30], iters=41)
        self.assertEqual(first, second)
        json.dumps(first, allow_nan=False)

    def test_zero_denominator_draws_are_not_silently_omitted(self):
        result = paired_relative_token_reduction([0, 2], [0, 4], iters=100,
                                                 seed=20261019)
        self.assertEqual(result["estimate"], .5)
        self.assertEqual(result["status"], "INCONCLUSIVE")
        self.assertEqual(result["reason"], "undefined_bootstrap_draw")
        self.assertGreater(result["undefined_draws"], 0)
        self.assertEqual(result["valid_draws"] + result["undefined_draws"], 100)
        self.assertIsNone(result["lower_bound"])
        self.assertIsNone(result["upper_bound"])

    def test_empty_and_all_zero_control_are_explicitly_inconclusive(self):
        for st, ctrl, reason in (([], [], "no_paired_observations"),
                                 ([1, 0], [0, 0], "zero_control_total")):
            result = paired_relative_token_reduction(st, ctrl, iters=11)
            self.assertEqual(result["status"], "INCONCLUSIVE")
            self.assertEqual(result["reason"], reason)
            self.assertIsNone(result["estimate"])
            self.assertIsNone(result["lower_bound"])
            self.assertFalse(result["bootstrap_performed"])
            json.dumps(result, allow_nan=False)

    def test_invalid_lengths_and_parameters(self):
        for left, right in (([1], []), ([-1], [1]), ([float("nan")], [1]),
                            ([True], [1]), ([1], [float("inf")])):
            with self.assertRaises(ValueError):
                paired_relative_token_reduction(left, right, iters=3)
        for kwargs in ({"iters": 0}, {"iters": True}, {"seed": "20261019"},
                       {"family_alpha": 1}, {"family_comparisons": -1}):
            with self.assertRaises(ValueError):
                paired_relative_token_reduction([1], [2], **kwargs)

    def test_finite_input_cannot_leak_nonfinite_ratio_or_sum(self):
        point = paired_relative_token_reduction([1e308], [1e-308], iters=2)
        self.assertEqual(point["status"], "INCONCLUSIVE")
        self.assertEqual(point["reason"], "nonfinite_point_ratio")
        self.assertIsNone(point["estimate"])
        json.dumps(point, allow_nan=False)
        with self.assertRaises(ValueError):
            paired_relative_token_reduction([1e308, 1e308], [1, 1], iters=2)


if __name__ == "__main__":
    unittest.main()
