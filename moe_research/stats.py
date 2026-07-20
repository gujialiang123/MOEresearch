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


def km_survival(lengths, events, tmax=None):
    """Kaplan-Meier survival curve for generation length treating hit-max as
    right-censored. `lengths[i]` = generated length; `events[i]` = True if the
    sequence terminated (EOS observed), False if right-censored (hit-max).
    Returns (times, survival) step arrays."""
    data = sorted(zip(lengths, events), key=lambda x: x[0])
    n = len(data)
    at_risk = n
    S = 1.0
    times, surv = [0], [1.0]
    i = 0
    while i < len(data):
        t = data[i][0]
        d = 0; c = 0
        while i < len(data) and data[i][0] == t:
            if data[i][1]:
                d += 1
            else:
                c += 1
            i += 1
        if at_risk > 0 and d > 0:
            S *= (1 - d / at_risk)
        times.append(t); surv.append(S)
        at_risk -= (d + c)
    if tmax is not None:
        times.append(tmax); surv.append(S)
    return times, surv


def restricted_mean_survival(lengths, events, tau):
    """Restricted mean survival time (area under KM up to horizon tau).
    A censoring-aware analogue of mean length that does not treat hit-max samples
    as if they terminated at exactly tau."""
    times, surv = km_survival(lengths, events, tmax=tau)
    area = 0.0
    for j in range(1, len(times)):
        t0 = times[j - 1]; t1 = min(times[j], tau)
        if t1 > t0:
            area += surv[j - 1] * (t1 - t0)
        if times[j] >= tau:
            break
    return round(area, 3)


def prompt_cluster_bootstrap_ci(deltas_by_cluster, iters=10000, alpha=0.05, seed=0):
    """Cluster (prompt) bootstrap: resample CLUSTERS with replacement, not individual
    token/observation-level deltas. `deltas_by_cluster` is a list of lists; each inner
    list holds the observation deltas for one prompt. Returns (mean, lo, hi) for the
    grand mean, correctly widening the CI for within-prompt correlation."""
    clusters = [c for c in deltas_by_cluster if c]
    if not clusters:
        return (None, None, None)
    rng = random.Random(seed)
    nC = len(clusters)
    all_vals = [v for c in clusters for v in c]
    grand = sum(all_vals) / len(all_vals)
    means = []
    for _ in range(iters):
        num = 0.0; den = 0
        for _ in range(nC):
            c = clusters[rng.randrange(nC)]
            num += sum(c); den += len(c)
        means.append(num / den if den else 0.0)
    means.sort()
    lo = means[int(alpha / 2 * iters)]
    hi = means[int((1 - alpha / 2) * iters)]
    return (round(grand, 4), round(lo, 4), round(hi, 4))

