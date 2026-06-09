#!/usr/bin/env python3
"""Skill impl: cross-regime-anomaly.

Rule-based finder over a regime_sweep_summary.json matrix. Emits a ranked
anomaly_report.json. See ../SKILL.md for kinds + severity heuristics.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

SCHEMA_VERSION = 0
RELIABLE_STDDEV_PCT = 8.0


def _gap_pct(a: float, b: float) -> float:
    """% by which a is faster than b (positive = a wins)."""
    if not b or b <= 0:
        return 0.0
    return (a - b) / b * 100.0


def _severity(gap_abs: float, kind: str) -> str:
    if kind == "reliability_flag" or kind == "failed_cell":
        return "medium"
    if gap_abs >= 50: return "high"
    if gap_abs >= 20: return "high"
    if gap_abs >= 10: return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Detectors — each returns list[finding-dict]
# ---------------------------------------------------------------------------

def detect_failed_cells(matrix: dict, _ctx) -> list[dict]:
    out = []
    for tag, cell in matrix.items():
        if not cell.get("ok", False):
            out.append({
                "kind": "failed_cell",
                "severity": _severity(0, "failed_cell"),
                "summary": f"config '{tag}' cell did not produce data: {cell.get('error','unknown')}",
                "evidence": {"config": tag, "error": cell.get("error")},
                "hypothesis_seed": "Server didn't respond / crashed / wrong URL. Without this cell, comparison is incomplete.",
                "next_skill": f"manually verify server at {tag}'s URL; then re-run regime-sweep-runner",
            })
    return out


def detect_reliability_flags(matrix: dict, _ctx) -> list[dict]:
    out = []
    for tag, cell in matrix.items():
        if not cell.get("ok"): continue
        for r_id, r in (cell.get("regimes") or {}).items():
            sd = r.get("stddev_pct")
            if sd is not None and sd > RELIABLE_STDDEV_PCT:
                out.append({
                    "kind": "reliability_flag",
                    "severity": _severity(sd, "reliability_flag"),
                    "summary": f"{tag}/{r_id} stddev_pct={sd:.1f} — exceeds {RELIABLE_STDDEV_PCT}% threshold",
                    "evidence": {"config": tag, "regime": r_id, "stddev_pct": sd,
                                 "req_per_s_mean": r.get("req_per_s_mean")},
                    "hypothesis_seed": "Run-to-run noise dominates — increase num_runs, "
                                       "or investigate co-tenant GPU contention / thermal throttling.",
                    "next_skill": f"re-run regime-sweep-runner with --num-runs 5 on cell '{tag}/{r_id}'",
                })
    return out


def detect_winner_inversion(matrix: dict, ctx) -> list[dict]:
    """For each config pair, check if winner changes across regimes."""
    out = []
    min_gap = ctx["min_gap_pct"]
    ok_configs = {t: c for t, c in matrix.items() if c.get("ok")}
    if len(ok_configs) < 2:
        return out
    for tag_a, tag_b in combinations(ok_configs.keys(), 2):
        cell_a, cell_b = ok_configs[tag_a], ok_configs[tag_b]
        regime_wins = {}
        gaps = {}
        for r_id in cell_a.get("regimes", {}):
            if r_id not in cell_b.get("regimes", {}): continue
            a = cell_a["regimes"][r_id].get("req_per_s_mean")
            b = cell_b["regimes"][r_id].get("req_per_s_mean")
            if a is None or b is None: continue
            gap = _gap_pct(a, b)
            gaps[r_id] = gap
            if abs(gap) < min_gap / 2:
                regime_wins[r_id] = "tie"
            elif gap > 0:
                regime_wins[r_id] = tag_a
            else:
                regime_wins[r_id] = tag_b
        winners = set(v for v in regime_wins.values() if v != "tie")
        if len(winners) >= 2:  # inversion
            max_gap = max(gaps.values())
            min_gap_v = min(gaps.values())
            out.append({
                "kind": "winner_inversion",
                "severity": _severity(max(abs(max_gap), abs(min_gap_v)), "winner_inversion"),
                "summary": f"{tag_a} vs {tag_b}: winner changes across regimes",
                "evidence": {
                    "configs": [tag_a, tag_b],
                    "regime_wins": regime_wins,
                    "gap_pcts_per_regime": {k: round(v, 1) for k, v in gaps.items()},
                    "max_gap_pct": round(max_gap, 1),
                    "min_gap_pct": round(min_gap_v, 1),
                },
                "hypothesis_seed": ("Choice between these configs has regime-dependent value. "
                                    "Likely a bottleneck shift (launch overhead vs kernel compute vs memory) "
                                    "across workload shape. Pick the regime where the gap is largest and profile there."),
                "next_skill": "nsys-capture + nsys-timeline-sql on the regime with the largest |gap|",
            })
    return out


def detect_large_uniform_gap(matrix: dict, ctx) -> list[dict]:
    """For each config pair, check if one wins on all regimes by a similar large %."""
    out = []
    min_gap = ctx["min_gap_pct"]
    ok_configs = {t: c for t, c in matrix.items() if c.get("ok")}
    for tag_a, tag_b in combinations(ok_configs.keys(), 2):
        cell_a, cell_b = ok_configs[tag_a], ok_configs[tag_b]
        gaps = {}
        for r_id in cell_a.get("regimes", {}):
            if r_id not in cell_b.get("regimes", {}): continue
            a = cell_a["regimes"][r_id].get("req_per_s_mean")
            b = cell_b["regimes"][r_id].get("req_per_s_mean")
            if a is None or b is None: continue
            gaps[r_id] = _gap_pct(a, b)
        if len(gaps) < 2: continue
        signs = set(1 if g > 0 else -1 for g in gaps.values() if abs(g) >= min_gap / 2)
        if len(signs) != 1: continue   # not uniform direction
        abs_gaps = [abs(g) for g in gaps.values()]
        if max(abs_gaps) < min_gap: continue   # not large enough
        # Check uniformity: all within 30% of each other
        if max(abs_gaps) > 0 and (min(abs_gaps) / max(abs_gaps)) > 0.7:
            winner, loser = (tag_a, tag_b) if list(gaps.values())[0] > 0 else (tag_b, tag_a)
            out.append({
                "kind": "large_uniform_gap",
                "severity": _severity(max(abs_gaps), "large_uniform_gap"),
                "summary": f"{winner} beats {loser} on all {len(gaps)} regimes by similar magnitudes (~{max(abs_gaps):.0f}%)",
                "evidence": {
                    "configs": [tag_a, tag_b],
                    "winner": winner,
                    "gap_pcts_per_regime": {k: round(v, 1) for k, v in gaps.items()},
                },
                "hypothesis_seed": "A configuration-level difference (autotune, cudagraph, default tactic) "
                                   "that affects every workload uniformly. NOT a workload-shape issue.",
                "next_skill": f"server-log-mining on {loser} to find what's different in config",
            })
    return out


def detect_regime_dependent_gap(matrix: dict, ctx) -> list[dict]:
    """Same two configs but gap varies > 2× across regimes (without flipping sign)."""
    out = []
    ok_configs = {t: c for t, c in matrix.items() if c.get("ok")}
    for tag_a, tag_b in combinations(ok_configs.keys(), 2):
        cell_a, cell_b = ok_configs[tag_a], ok_configs[tag_b]
        gaps = {}
        for r_id in cell_a.get("regimes", {}):
            if r_id not in cell_b.get("regimes", {}): continue
            a = cell_a["regimes"][r_id].get("req_per_s_mean")
            b = cell_b["regimes"][r_id].get("req_per_s_mean")
            if a is None or b is None: continue
            gaps[r_id] = _gap_pct(a, b)
        if len(gaps) < 2: continue
        signs = set(1 if g > 0 else -1 for g in gaps.values())
        if len(signs) != 1: continue   # already handled by inversion
        abs_gaps = [abs(g) for g in gaps.values() if abs(g) > 1]
        if len(abs_gaps) < 2: continue
        ratio = max(abs_gaps) / max(0.01, min(abs_gaps))
        if ratio >= 2.0 and max(abs_gaps) >= ctx["min_gap_pct"]:
            out.append({
                "kind": "regime_dependent_gap",
                "severity": _severity(max(abs_gaps), "regime_dependent_gap"),
                "summary": f"{tag_a} vs {tag_b}: gap varies {ratio:.1f}× across regimes (consistent sign)",
                "evidence": {
                    "configs": [tag_a, tag_b],
                    "gap_pcts_per_regime": {k: round(v, 1) for k, v in gaps.items()},
                    "max_min_ratio": round(ratio, 2),
                },
                "hypothesis_seed": "Same winner but gap size depends on workload — bottleneck shifts within "
                                   "this comparison. Where gap is smallest, the secondary bottleneck dominates.",
                "next_skill": "nsys-timeline-sql on both small-gap and large-gap regimes; diff them",
            })
    return out


KIND_DETECTORS = [
    detect_failed_cells,
    detect_reliability_flags,
    detect_winner_inversion,
    detect_large_uniform_gap,
    detect_regime_dependent_gap,
]

SEVERITY_RANK = {"high": 3, "medium": 2, "low": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-file", required=True)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--min-gap-pct", type=float, default=15.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    try:
        sweep = json.loads(Path(args.sweep_file).read_text())
    except Exception as e:
        Path(args.out).write_text(json.dumps(
            {"schema_version": SCHEMA_VERSION, "ok": False, "error": f"sweep load: {e}"}, indent=2))
        return

    matrix = sweep.get("matrix") or {}
    if len(matrix) < 2:
        Path(args.out).write_text(json.dumps(
            {"schema_version": SCHEMA_VERSION, "ok": False,
             "error": "need ≥2 configs in matrix"}, indent=2))
        return

    # Reliability ratio (over all OK cells)
    total = 0; reliable = 0
    for cell in matrix.values():
        if not cell.get("ok"): continue
        for r in (cell.get("regimes") or {}).values():
            total += 1
            if r.get("reliable"): reliable += 1
    ratio = reliable / total if total else 0.0
    if ratio < 0.6 and total > 0:
        Path(args.out).write_text(json.dumps(
            {"schema_version": SCHEMA_VERSION, "ok": False,
             "error": f"reliability_ratio={ratio:.2f} below 0.60 — re-sweep with --num-runs 5",
             "reliability_ratio": ratio}, indent=2))
        return

    ctx = {"min_gap_pct": args.min_gap_pct}
    findings = []
    for det in KIND_DETECTORS:
        findings.extend(det(matrix, ctx))

    # Sort by severity then by max gap magnitude
    def _sort_key(f):
        ev = f.get("evidence") or {}
        gap_magnitude = max(
            abs(ev.get("max_gap_pct", 0) or 0),
            abs(ev.get("min_gap_pct", 0) or 0),
            *[abs(v) for v in (ev.get("gap_pcts_per_regime") or {}).values()],
            ev.get("stddev_pct", 0) or 0,
        )
        return (-SEVERITY_RANK.get(f.get("severity", "low"), 0), -gap_magnitude)

    findings.sort(key=_sort_key)
    for i, f in enumerate(findings[:args.top_n], 1):
        f["rank"] = i

    warnings = []
    if not findings:
        warnings.append(f"no anomalies above min_gap_pct={args.min_gap_pct} — "
                        "try lower threshold (--min-gap-pct 5)")

    Path(args.out).write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "sweep_file": args.sweep_file,
        "reliability_ratio": round(ratio, 3),
        "findings": findings[:args.top_n],
        "warnings": warnings,
    }, indent=2))
    print(f"[cross-regime-anomaly] wrote {args.out} — {len(findings[:args.top_n])} findings")


if __name__ == "__main__":
    main()
