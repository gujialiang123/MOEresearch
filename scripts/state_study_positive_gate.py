"""Independent CPU statistics for a prospectively frozen positive-study gate.

This module reads no experiment files and imports no tensor packages.  The
caller must first establish the frozen prompt universe, paired identities,
controls, and the other links in the scientific chain.  These functions do not
establish noninferiority or validate that complete chain.

For each accuracy comparison, let W and L be the counts of ST-only and
control-only correct answers among N *all paired prompts*.  The population
accuracy difference is p_W - p_L.  A one-sided Clopper--Pearson (CP) lower bound
for p_W minus a one-sided CP upper bound for p_L is a conservative difference
lower bound.  For m predeclared comparisons, each of these 2m bounds uses tail
alpha/(2m).  The union bound gives simultaneous coverage at least 1-alpha;
neither independence of W and L nor independence of comparisons is required.

CP uses the finite binomial distribution, not a normal approximation.  Its
root inversion is evaluated with 64-digit Decimal arithmetic, 80 bisections,
and an outward float rounding step.  "Exact" describes the binomial inference;
the returned decimal numbers necessarily have finite numerical precision.

Token intervals instead use an approximate paired-prompt percentile bootstrap
of 1 - sum(ST)/sum(control).  Bonferroni tail allocation is predeclared across
the cost comparisons; this does not turn bootstrap coverage into a finite-
sample exact guarantee.  A draw with zero control denominator is never dropped.

Sources for binomial inversion and the CP method:
https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html
"""
from __future__ import annotations

from decimal import Decimal, localcontext
import hashlib
import json
import math
from numbers import Real
import random


QUALITY_METHOD = "simultaneous_bonferroni_clopper_pearson_difference.v1"
TOKEN_METHOD = "paired_prompt_ratio_of_sums_percentile_bootstrap.v1"
BINOMIAL_SOURCES = (
    "https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm",
    "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html",
)


def _family_parameters(family_alpha, family_comparisons):
    if (isinstance(family_alpha, bool) or not isinstance(family_alpha, Real)
            or not math.isfinite(family_alpha) or not 0 < family_alpha < 1):
        raise ValueError("family_alpha must be finite and strictly between 0 and 1")
    if (isinstance(family_comparisons, bool)
            or not isinstance(family_comparisons, int) or family_comparisons < 1):
        raise ValueError("family_comparisons must be a positive integer")
    alpha = float(family_alpha)
    return alpha, alpha / (2 * family_comparisons)


def _paired_hash(left, right):
    value = {"left": left, "right": right}
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _binary_values(values, name):
    output = []
    for value in values:
        if not (isinstance(value, (bool, int)) and value in (0, 1)):
            raise ValueError(f"{name} must contain only booleans or integer 0/1")
        output.append(bool(value))
    return output


def _clopper_pearson_lower(successes, n, tail_alpha):
    """Invert P[Binomial(n,p) >= successes] = tail_alpha from below."""
    if successes == 0:
        return 0.0
    with localcontext() as context:
        context.prec = 64
        target = Decimal.from_float(tail_alpha)
        count = Decimal(successes)
        total = Decimal(n)
        coefficient = Decimal(math.comb(n, successes))
        lo, hi = Decimal(0), count / total
        for _ in range(80):
            p = (lo + hi) / 2
            q = 1 - p
            term = coefficient * p ** successes * q ** (n - successes)
            probability = term
            # p <= successes/n, so these tail terms decrease, avoiding large
            # cancellation and an unstable subtraction of CDF from one.
            for k in range(successes, n):
                term *= Decimal(n - k) * p / (Decimal(k + 1) * q)
                probability += term
            if probability <= target:
                lo = p
            else:
                hi = p
        return max(0.0, math.nextafter(float(lo), 0.0))


def _clopper_pearson_upper(successes, n, tail_alpha):
    if successes == n:
        return 1.0
    lower_failures = _clopper_pearson_lower(n - successes, n, tail_alpha)
    return min(1.0, math.nextafter(1.0 - lower_failures, 1.0))


def quality_superiority(st_correct, control_correct, *, family_alpha=.05,
                        family_comparisons=4):
    """Return a simultaneous, exact-binomial lower bound on paired accuracy gain.

    The four-comparison default corresponds to strict/tolerant parser x
    ST-vs-NT/strongest.  The caller must freeze that family before validation.
    Units are accuracy fractions, so .01 is one percentage point.  A strictly
    positive lower bound is sufficient superiority; a failed bound is only
    INCONCLUSIVE and supplies no noninferiority claim.
    """
    alpha, per_tail = _family_parameters(family_alpha, family_comparisons)
    st = _binary_values(st_correct, "st_correct")
    control = _binary_values(control_correct, "control_correct")
    if len(st) != len(control):
        raise ValueError("paired correctness vectors must have equal lengths")
    n = len(st)
    wins = sum(left and not right for left, right in zip(st, control))
    losses = sum(right and not left for left, right in zip(st, control))
    both_correct = sum(left and right for left, right in zip(st, control))
    both_wrong = n - wins - losses - both_correct
    result = {
        "method": QUALITY_METHOD,
        "n": n, "wins": wins, "losses": losses,
        "both_correct": both_correct, "both_wrong": both_wrong,
        "counts": {"wins": wins, "losses": losses,
                   "both_correct": both_correct, "both_wrong": both_wrong},
        "estimate": (wins - losses) / n if n else None,
        "family_alpha": alpha, "family_comparisons": family_comparisons,
        "per_tail_alpha": per_tail,
        "family_confidence_level_at_least": 1 - alpha,
        "ci_unit": "accuracy_fraction",
        "bootstrap_unit": None,
        "quality_noninferiority_margin": None,
        "criterion": "exact_lower_bound > 0",
        "input_pair_sha256": _paired_hash(st, control),
        "sources": list(BINOMIAL_SOURCES),
    }
    if not n:
        result.update(status="INCONCLUSIVE", reason="no_paired_observations",
                      win_probability_lower=None, loss_probability_upper=None,
                      exact_lower_bound=None, lower_bound=None)
        return result
    win_lower = _clopper_pearson_lower(wins, n, per_tail)
    loss_upper = _clopper_pearson_upper(losses, n, per_tail)
    lower = max(-1.0, math.nextafter(win_lower - loss_upper, -math.inf))
    superior = lower > 0
    result.update(
        status="SUPERIORITY_ESTABLISHED" if superior else "INCONCLUSIVE",
        reason="simultaneous_accuracy_lower_bound_positive" if superior
               else "simultaneous_accuracy_lower_bound_not_positive",
        win_probability_lower=win_lower, loss_probability_upper=loss_upper,
        exact_lower_bound=lower, lower_bound=lower,
    )
    return result


def _length_values(values, name):
    output = []
    for value in values:
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must contain finite nonnegative numbers")
        output.append(float(value))
    return output


def _percentile(sorted_values, probability):
    """Linearly interpolated percentile at index (B-1)*probability."""
    index = (len(sorted_values) - 1) * probability
    below = int(index)
    above = min(below + 1, len(sorted_values) - 1)
    return (sorted_values[below]
            + (sorted_values[above] - sorted_values[below]) * (index - below))


def paired_relative_token_reduction(st_lengths, control_lengths, *, iters=10000,
                                    seed=20261019, family_alpha=.05,
                                    family_comparisons=2):
    """Recompute a ratio of sums on every paired-prompt bootstrap draw.

    Positive values favor ST.  The default interval uses percentile endpoints
    .0125 and .9875, a Bonferroni allocation across two cost comparisons.  Its
    units are relative fractions; .05 means five percent fewer complete
    generated tokens.  A prospectively specified lower_bound >= .05 can be
    tested by the caller.  This function does not choose a practical threshold.

    Call lazily after other chain requirements pass, and cache the returned
    object using the input hash and all method/parameter fields.  No actual
    experiment data are read here.  Seeds deterministically preserve prompt
    pairing; an undefined denominator invalidates the entire interval.
    """
    alpha, per_tail = _family_parameters(family_alpha, family_comparisons)
    if isinstance(iters, bool) or not isinstance(iters, int) or iters < 1:
        raise ValueError("iters must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    st = _length_values(st_lengths, "st_lengths")
    control = _length_values(control_lengths, "control_lengths")
    if len(st) != len(control):
        raise ValueError("paired length vectors must have equal lengths")
    n = len(st)
    try:
        st_total, control_total = math.fsum(st), math.fsum(control)
    except OverflowError as error:
        raise ValueError("paired token sums must be finite") from error
    if not math.isfinite(st_total) or not math.isfinite(control_total):
        raise ValueError("paired token sums must be finite")
    point = 1 - st_total / control_total if control_total else None
    nonfinite_point = point is not None and not math.isfinite(point)
    result = {
        "method": TOKEN_METHOD, "n": n,
        "st_token_sum": st_total, "control_token_sum": control_total,
        "estimate": None if nonfinite_point else point,
        "iters": iters, "seed": seed,
        "family_alpha": alpha, "family_comparisons": family_comparisons,
        "per_tail_alpha": per_tail,
        "nominal_family_confidence_level": 1 - alpha,
        "coverage": "approximate_percentile_bootstrap_with_bonferroni_tail_allocation",
        "ci_unit": "relative_fraction", "bootstrap_unit": "paired_prompt",
        "estimand": "1 - sum(st_lengths) / sum(control_lengths)",
        "quantile_method": "linear_at_(B-1)*q",
        "sampling_method": "python.random.Random.randrange_paired_indices_with_replacement.v1",
        "input_pair_sha256": _paired_hash(st, control),
        "lower_bound": None, "upper_bound": None,
        "ci_low": None, "ci_high": None,
    }
    if not n or not control_total or nonfinite_point:
        result.update(status="INCONCLUSIVE",
                      reason=("no_paired_observations" if not n else
                              "zero_control_total" if not control_total else
                              "nonfinite_point_ratio"),
                      bootstrap_performed=False, executed_draws=0,
                      undefined_draws=iters, valid_draws=0)
        return result
    rng = random.Random(seed)
    values = []
    undefined = 0
    for _ in range(iters):
        indices = [rng.randrange(n) for _ in range(n)]
        try:
            denominator = math.fsum(control[i] for i in indices)
            numerator = math.fsum(st[i] for i in indices)
        except OverflowError:
            undefined += 1
            continue
        if denominator == 0:
            undefined += 1
            continue
        value = 1 - numerator / denominator
        if not math.isfinite(value):
            undefined += 1
        else:
            values.append(value)
    result.update(bootstrap_performed=True, executed_draws=iters,
                  undefined_draws=undefined, valid_draws=len(values))
    if undefined:
        result.update(status="INCONCLUSIVE", reason="undefined_bootstrap_draw")
        return result
    values.sort()
    lower = _percentile(values, per_tail)
    upper = _percentile(values, 1 - per_tail)
    result.update(status="OK", reason="paired_ratio_interval_estimable",
                  lower_bound=lower, upper_bound=upper, ci_low=lower, ci_high=upper)
    return result
