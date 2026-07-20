"""Paired stats: bootstrap CI, McNemar, Holm correction (v23+ plan §十一)."""
import random
import math


def paired_bootstrap_ci(deltas, iters=10000, alpha=0.05, seed=0):
    """95% CI of the mean of paired deltas via bootstrap resampling."""
    if not deltas:
        return (None, None, None)
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(iters):
        s = 0.0
        for _ in range(n):
            s += deltas[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(alpha / 2 * iters)]
    hi = means[int((1 - alpha / 2) * iters)]
    mean = sum(deltas) / n
    return (round(mean, 3), round(lo, 3), round(hi, 3))


def mcnemar_exact(base_correct, intervention_correct):
    """Exact McNemar test on paired boolean correctness.
    b = base correct & intervention wrong; c = base wrong & intervention correct.
    Returns dict with b, c, and exact two-sided p-value (binomial)."""
    b = sum(1 for x, y in zip(base_correct, intervention_correct) if x and not y)
    c = sum(1 for x, y in zip(base_correct, intervention_correct) if (not x) and y)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0}
    k = min(b, c)
    # two-sided exact binomial with p=0.5
    p = 0.0
    for i in range(0, k + 1):
        p += math.comb(n, i) * (0.5 ** n)
    p = min(1.0, 2 * p)
    return {"b": b, "c": c, "p_value": round(p, 6)}


def holm_correction(pvals_dict, alpha=0.05):
    """Holm-Bonferroni. pvals_dict: name->p. Returns name->(p, reject)."""
    items = sorted(pvals_dict.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    prev_reject = True
    for rank, (name, p) in enumerate(items):
        thresh = alpha / (m - rank)
        reject = prev_reject and (p <= thresh)
        prev_reject = reject
        out[name] = {"p": p, "holm_thresh": round(thresh, 6), "reject": bool(reject)}
    return out


def ecdf(values):
    """Return sorted (x, cumulative_fraction) for a survival/ECDF plot."""
    if not values:
        return []
    xs = sorted(values)
    n = len(xs)
    return [(x, (i + 1) / n) for i, x in enumerate(xs)]
