#!/usr/bin/env python3
"""Minimal Stage B config-agent.

Reads one problem package, applies one knob change from its suggested
strategies (single value per attempt), runs the full benchmark suite
(target + neighbors + controls), compares against the baseline metrics
already in the package, and writes a decision.json + plan.md under
problem_dir/attempts/attempt_NNN/.

Usage:
    python scripts/solver/config_agent.py \\
        --problem experiments/problems_moe/P001 \\
        --strategy S001 \\
        --value 64

If --value is omitted, picks the first entry from the strategy's
values_to_try.

Currently knob-edit only (no kernel modifications). Does NOT modify the
problem package's root files; only writes under attempts/.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))
from utils import load_json, load_yaml, save_json, save_yaml, now_str  # noqa: E402
from logging_setup import setup_logger  # noqa: E402


def next_attempt_dir(problem_dir: Path) -> Path:
    attempts = problem_dir / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        int(p.name.split("_")[-1])
        for p in attempts.glob("attempt_*")
        if p.is_dir() and p.name.split("_")[-1].isdigit()
    )
    n = (existing[-1] + 1) if existing else 1
    d = attempts / f"attempt_{n:03d}"
    d.mkdir()
    return d


def run_workload(server_cfg: Path, workload: Path, out_dir: Path, log) -> dict | None:
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "run_experiment.py"),
        "--config", str(server_cfg),
        "--workload", str(workload),
        "--mode", "quick",
        "--out-dir", str(out_dir),
        "--server-start-timeout", "600",
        "--benchmark-timeout", "1200",
    ]
    log.info(f"  CMD: {' '.join(cmd)}")
    t0 = time.monotonic()
    orch_log = out_dir / "orchestrator.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(orch_log, "ab") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
    dur = time.monotonic() - t0
    metrics_path = out_dir / "quick_metrics.json"
    metrics = load_json(metrics_path) if metrics_path.exists() else None
    log.info(f"  → rc={rc} dur={dur:.1f}s passed={(metrics or {}).get('passed')}")
    return metrics


def percentage_change(new, old):
    if new is None or old is None or old == 0:
        return None
    return (new - old) / old * 100.0


def pick_strategy(problem: dict, strategy_id: str | None) -> dict:
    strategies = problem.get("suggested_strategies") or []
    if not strategies:
        sys.exit("[config-agent] problem has no suggested_strategies")
    if strategy_id:
        for s in strategies:
            if s["strategy_id"] == strategy_id:
                return s
        sys.exit(f"[config-agent] strategy_id {strategy_id} not in suggested_strategies")
    return strategies[0]


def build_candidate_config(baseline_config_path: Path, knob: str, value: Any,
                           candidate_path: Path) -> dict:
    cfg = load_yaml(baseline_config_path)
    cfg[knob] = value
    save_yaml(cfg, candidate_path)
    return cfg


def evaluate(target_baseline, target_new, neighbors, controls, accept, log):
    primary_cfg = accept["primary"]
    metric = primary_cfg["metric"]
    direction = primary_cfg["direction"]
    required_pct = float(primary_cfg["required_improvement_pct"])

    old = target_baseline.get(metric)
    new = (target_new or {}).get(metric)
    delta = percentage_change(new, old)
    improvement = -(delta or 0.0) if direction == "lower" else (delta or 0.0)
    log.info(f"  primary {metric}: baseline={old} new={new} improvement={improvement:+.1f}% (need ≥{required_pct}%)")

    violations = []
    for c in accept.get("constraints", []):
        m, on = c["metric"], c["on"]
        if on == "target":
            base_v, new_v = target_baseline.get(m), (target_new or {}).get(m)
        elif on == "any":
            worst = 0
            for nm, base, nw in [("target", target_baseline, target_new),
                                  *[(n, b, w) for n, b, w in neighbors],
                                  *[(n, b, w) for n, b, w in controls]]:
                v = (nw or {}).get(m)
                if isinstance(v, bool):
                    worst = max(worst, 1 if v else 0)
                elif isinstance(v, (int, float)) and v is not None:
                    worst = max(worst, v)
            new_v, base_v = worst, 0
        elif on.startswith("controls."):
            name = on.split(".", 1)[1]
            base_v = new_v = None
            for nm, base, nw in controls:
                if nm == name or nm.startswith(name):
                    base_v, new_v = (base or {}).get(m), (nw or {}).get(m)
                    break
        elif on.startswith("neighbors."):
            name = on.split(".", 1)[1]
            base_v = new_v = None
            for nm, base, nw in neighbors:
                if nm == name or nm.startswith(name):
                    base_v, new_v = (base or {}).get(m), (nw or {}).get(m)
                    break
        else:
            continue

        if "max" in c and new_v is not None:
            v_int = 1 if isinstance(new_v, bool) and new_v else (new_v if not isinstance(new_v, bool) else 0)
            if v_int > c["max"]:
                violations.append({"metric": m, "on": on, "value": new_v, "max": c["max"]})
        elif "max_regression_pct" in c and new_v is not None and base_v is not None:
            pct = percentage_change(new_v, base_v) or 0.0
            regression = max(0.0, pct) if c["direction"] == "lower" else max(0.0, -pct)
            if regression > c["max_regression_pct"]:
                violations.append({"metric": m, "on": on,
                                   "regression_pct": round(regression, 2),
                                   "max_allowed_pct": c["max_regression_pct"]})

    if violations:
        return "revert", f"Constraint violations: {violations}. Primary change {improvement:+.1f}%.", violations, improvement
    if improvement >= required_pct:
        return "keep", f"Primary metric improved {improvement:+.1f}% ≥ {required_pct}% threshold; no constraint violations.", [], improvement
    if improvement > 0:
        return "needs_more_evidence", f"Primary improved {improvement:+.1f}% but below {required_pct}% threshold.", [], improvement
    return "revert", f"Primary did not improve (change {improvement:+.1f}%, need ≥{required_pct}%).", [], improvement


def run_one_attempt(problem_dir: Path, problem: dict, accept: dict,
                    baseline_cfg_path: Path, strategy: dict, value: Any,
                    note: str | None = None) -> dict:
    """Run a single attempt with strategy.knob=value. Returns the decision_doc."""
    knob = strategy["knob"]
    attempt_dir = next_attempt_dir(problem_dir)
    log = setup_logger(f"cfg_agent_{attempt_dir.name}", attempt_dir / "config_agent.log")
    log.info(f"=== config-agent attempt @ {now_str()} ===")
    log.info(f"problem_dir: {problem_dir}")
    log.info(f"strategy: {strategy['strategy_id']} → knob={knob} value={value}")
    log.info(f"attempt_dir: {attempt_dir}")

    candidate_path = attempt_dir / "candidate_config.yaml"
    build_candidate_config(baseline_cfg_path, knob, value, candidate_path)
    log.info(f"candidate written: {knob}={value}")

    plan_md = f"""# Attempt {attempt_dir.name} — config-agent

**Problem**: {problem['problem_id']} ({problem.get('regime_id')})
**Strategy**: {strategy['strategy_id']} — {strategy.get('rationale', '')}
**Knob**: `{knob}` → `{value}`
**Expected**: +{strategy.get('expected_improvement_pct', '?')}% on `{problem['symptom']['metric']}`
**Risk**: {strategy.get('risk', '')}
**Note**: {note or '(none)'}
"""
    (attempt_dir / "plan.md").write_text(plan_md)

    target_dir = attempt_dir / "verification" / "target"
    target_workload = problem_dir / problem["target"]["workload"]
    log.info("--- target ---")
    target_new = run_workload(candidate_path, target_workload, target_dir, log)

    neighbors_results = []
    for nb in problem.get("benchmark_suite", {}).get("neighbors", []):
        wl = problem_dir / nb["workload"]
        base = load_json(problem_dir / nb["baseline_metrics"]) if nb.get("baseline_metrics") else None
        nb_dir = attempt_dir / "verification" / "neighbors" / nb["name"]
        log.info(f"--- neighbor {nb['name']} ---")
        nw = run_workload(candidate_path, wl, nb_dir, log)
        neighbors_results.append((nb["name"], base, nw))

    controls_results = []
    for c in problem.get("benchmark_suite", {}).get("controls", []):
        wl = problem_dir / c["workload"]
        base = load_json(problem_dir / c["baseline_metrics"]) if c.get("baseline_metrics") else None
        c_dir = attempt_dir / "verification" / "controls" / c["name"]
        log.info(f"--- control {c['name']} ---")
        nw = run_workload(candidate_path, wl, c_dir, log)
        controls_results.append((c["name"], base, nw))

    target_baseline = load_json(problem_dir / problem["target"]["baseline_metrics"])
    if target_new is None:
        decision, reasoning, violations, improvement = (
            "revert", "Target benchmark did not produce metrics; reverting.", [], None
        )
    else:
        decision, reasoning, violations, improvement = evaluate(
            target_baseline, target_new, neighbors_results, controls_results, accept, log
        )

    pm = accept["primary"]["metric"]
    pd = accept["primary"]["direction"]
    also_solved = []
    for nm, base, nw in neighbors_results + controls_results:
        if not (base and nw):
            continue
        delta = percentage_change(nw.get(pm), base.get(pm))
        if delta is None:
            continue
        imp = -delta if pd == "lower" else delta
        if imp >= 10.0:
            also_solved.append({
                "ref": nm, "metric": pm,
                "delta_pct": round(imp, 2),
                "note": f"Side improvement under the same knob change.",
            })

    decision_doc = {
        "schema_version": 1,
        "attempt_id": attempt_dir.name,
        "attempt_dir": str(attempt_dir),
        "solver_agent": "config-agent",
        "knob": knob,
        "value": value,
        "strategy_id": strategy["strategy_id"],
        "decision": decision,
        "reasoning": reasoning,
        "primary_metric": pm,
        "primary_direction": pd,
        "primary_baseline": target_baseline.get(pm),
        "primary_new": (target_new or {}).get(pm),
        "primary_delta_pct": improvement,
        "constraint_violations": violations,
        "also_solved": also_solved,
        "new_findings_filed": [],
        "verification": {
            "target": {"baseline": target_baseline.get(pm),
                       "new": (target_new or {}).get(pm)},
            "neighbors": [
                {"name": nm, "baseline": (base or {}).get(pm), "new": (nw or {}).get(pm)}
                for nm, base, nw in neighbors_results
            ],
            "controls": [
                {"name": nm, "baseline": (base or {}).get(pm), "new": (nw or {}).get(pm)}
                for nm, base, nw in controls_results
            ],
        },
    }
    save_json(decision_doc, attempt_dir / "decision.json")
    log.info(f"=== DECISION: {decision} ===")
    log.info(reasoning)
    if also_solved:
        log.info(f"also_solved: {also_solved}")
    return decision_doc


def rank_attempts(attempt_docs: list[dict]) -> list[dict]:
    """Sort by: no violations > primary_delta_pct (best improvement) > smaller value."""
    def key(d):
        # in our codebase, "improvement" is already sign-corrected by evaluate(),
        # and decision_doc.primary_delta_pct == improvement_pct (higher = better).
        imp = d.get("primary_delta_pct")
        violated = 1 if d.get("constraint_violations") else 0
        # Smaller |value| as tiebreaker assumes numeric; fall back to 0 otherwise
        try:
            v_tiebreak = abs(float(d["value"]))
        except (TypeError, ValueError):
            v_tiebreak = 0
        # Sort key: (violations asc, improvement desc → use -imp, value asc)
        return (violated, -(imp if imp is not None else -999), v_tiebreak)
    return sorted(attempt_docs, key=key)


def write_search_summary(problem_dir: Path, problem: dict, strategy: dict,
                         attempt_docs: list[dict], best: dict | None) -> Path:
    """Write a Markdown solution.md ranking the attempts."""
    sol_path = problem_dir / "solution.md"
    lines = []
    lines.append(f"# Solution for {problem['problem_id']}\n")
    lines.append("**Auto-written by `config_agent.py --exhaustive`.**\n")
    lines.append(f"**Problem**: {problem.get('regime_id')} on `{problem.get('model_path')}`.\n")
    lines.append(f"**Strategy**: `{strategy['strategy_id']}` — knob `{strategy['knob']}`. "
                 f"Rationale: {strategy.get('rationale', '')}\n")
    lines.append(f"**Values swept**: {[d['value'] for d in attempt_docs]}\n")
    pm = problem.get("symptom", {}).get("metric") or attempt_docs[0]["primary_metric"]
    base = attempt_docs[0].get("primary_baseline")
    lines.append(f"**Baseline {pm}**: {base:.2f}\n" if base else "")

    lines.append("\n## Per-value results\n")
    lines.append("| Attempt | Value | Decision | New | Δ% | Violations | also_solved |")
    lines.append("|---|---:|---|---:|---:|---|---:|")
    for d in attempt_docs:
        v = d.get("value")
        new = d.get("primary_new")
        imp = d.get("primary_delta_pct")
        viol = len(d.get("constraint_violations") or [])
        nas = len(d.get("also_solved") or [])
        new_s = f"{new:.2f}" if isinstance(new, (int, float)) else "—"
        imp_s = f"{imp:+.1f}" if isinstance(imp, (int, float)) else "—"
        lines.append(f"| {d['attempt_id']} | {v} | {d['decision']} | {new_s} | {imp_s} | {viol} | {nas} |")

    lines.append("\n## Best attempt")
    if best is None:
        lines.append("\n**No attempt passed acceptance criteria.** See per-value table above.\n")
    else:
        # detect if other passing attempts are within noise
        passing = [d for d in attempt_docs
                   if not d.get("constraint_violations")
                   and (d.get("primary_delta_pct") or 0) >= 0]
        near = [d for d in passing
                if abs((d.get("primary_delta_pct") or 0) - (best["primary_delta_pct"] or 0)) <= 1.0
                and d["attempt_id"] != best["attempt_id"]]
        note = ""
        if near:
            note = (f"\n> **Note**: {len(near)+1} values "
                    f"({sorted([best['value']] + [d['value'] for d in near])}) "
                    f"give effectively the same improvement (within ±1%). "
                    f"Picked the smallest value to minimize memory/risk.\n")
        lines.append(
            f"\n**`{best['attempt_id']}`** — set `{best['knob']} = {best['value']}`\n"
            f"{note}\n"
            f"- {pm}: **{best['primary_baseline']:.2f} → {best['primary_new']:.2f}** "
            f"({best['primary_delta_pct']:+.1f}%)\n"
            f"- Decision: **{best['decision']}**\n"
            f"- Constraint violations: {best.get('constraint_violations') or 'none'}\n"
        )
        if best.get("also_solved"):
            lines.append("- Side-solved problems:\n")
            for a in best["also_solved"]:
                lines.append(f"  - `{a['ref']}` ({a['metric']}: {a['delta_pct']:+.1f}%)\n")
        lines.append(
            f"\n## Recommended config change\n\n"
            f"Edit `{problem.get('baseline_config')}` to set:\n\n"
            f"```yaml\n{best['knob']}: {best['value']}\n```\n"
        )

    lines.append("\n## Reproducibility\n")
    lines.append(f"```bash\n# Apply the recommended fix and re-run target:\n"
                 f"python scripts/run_experiment.py \\\n"
                 f"    --config {best['attempt_dir']}/candidate_config.yaml \\\n"
                 f"    --workload {problem_dir / problem['target']['workload']} \\\n"
                 f"    --mode quick\n```\n" if best else "")
    sol_path.write_text("\n".join(lines))
    return sol_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--value", default=None,
                    help="single value to try (overrides exhaustive)")
    ap.add_argument("--exhaustive", action="store_true",
                    help="sweep ALL values in the strategy's values_to_try")
    ap.add_argument("--note", default=None)
    args = ap.parse_args()

    problem_dir = Path(args.problem).resolve()
    if not (problem_dir / "problem.json").exists():
        sys.exit(f"[config-agent] not a problem package: {problem_dir}")

    problem = load_json(problem_dir / "problem.json")
    accept = load_json(problem_dir / problem["acceptance_criteria_file"])
    baseline_cfg_path = Path(problem["baseline_config"])
    strategy = pick_strategy(problem, args.strategy)

    # Decide values to try
    def cast(rv: str) -> Any:
        if rv.lower() in ("true", "false"):
            return rv.lower() == "true"
        try:
            return int(rv)
        except ValueError:
            try:
                return float(rv)
            except ValueError:
                return rv

    if args.value is not None:
        values = [cast(args.value)]
    elif args.exhaustive:
        values = list(strategy["values_to_try"])
    else:
        values = [strategy["values_to_try"][0]]

    # Reuse prior attempts that match (strategy_id, value)
    prior_docs: dict = {}  # (strategy_id, str(value)) -> doc
    attempts_root = problem_dir / "attempts"
    if attempts_root.exists():
        for ap in sorted(attempts_root.glob("attempt_*")):
            dj = ap / "decision.json"
            if dj.exists():
                try:
                    d = load_json(dj)
                    d["attempt_dir"] = str(ap)
                    key = (d.get("strategy_id"), str(d.get("value")))
                    prior_docs[key] = d
                except Exception:
                    pass

    print(f"[config-agent] sweeping {len(values)} value(s): {values}")
    attempt_docs: list[dict] = []
    for v in values:
        key = (strategy["strategy_id"], str(v))
        if key in prior_docs:
            d = prior_docs[key]
            print(f"[config-agent] reusing {d['attempt_id']} for {strategy['strategy_id']}={v}")
            attempt_docs.append(d)
            continue
        doc = run_one_attempt(problem_dir, problem, accept, baseline_cfg_path,
                              strategy, v, note=args.note)
        attempt_docs.append(doc)

    # Rank + pick best
    # If multiple attempts pass acceptance and their primary_delta_pct are
    # within `noise_tol_pct` of the leader, prefer the SMALLEST value
    # (least disruptive change, smaller memory footprint, lower risk).
    noise_tol_pct = 1.0
    ranked = rank_attempts(attempt_docs)
    passing = [d for d in ranked
               if not d.get("constraint_violations")
               and (d.get("primary_delta_pct") or 0) >= float(
                   accept["primary"]["required_improvement_pct"])]
    best = None
    if passing:
        leader_imp = passing[0]["primary_delta_pct"]
        near_leader = [d for d in passing
                       if (leader_imp - (d.get("primary_delta_pct") or 0)) <= noise_tol_pct]
        # Tie-break within noise: smaller numeric value wins
        def _val_key(d):
            try:
                return float(d["value"])
            except (TypeError, ValueError):
                return 0
        best = sorted(near_leader, key=_val_key)[0]
        if len(near_leader) > 1:
            print(f"[config-agent] {len(near_leader)} values within ±{noise_tol_pct}% "
                  f"of leader; picked smallest value={best['value']}")

    # Write search summary (problem-level)
    if len(values) > 1:
        sol_path = write_search_summary(problem_dir, problem, strategy, ranked, best)
        print(f"[config-agent] wrote solution summary: {sol_path}")

    summary = {
        "mode": "exhaustive" if args.exhaustive else "single",
        "strategy": strategy["strategy_id"],
        "knob": strategy["knob"],
        "values_tried": values,
        "attempts": [
            {"attempt_id": d["attempt_id"], "value": d["value"],
             "decision": d["decision"], "delta_pct": d["primary_delta_pct"],
             "violations": len(d.get("constraint_violations") or [])}
            for d in attempt_docs
        ],
        "best_attempt": best["attempt_id"] if best else None,
        "best_value": best["value"] if best else None,
        "best_delta_pct": best["primary_delta_pct"] if best else None,
    }
    print(json.dumps(summary, indent=2))

    return 0 if best else (2 if any(d["decision"] == "needs_more_evidence" for d in attempt_docs) else 1)


if __name__ == "__main__":
    sys.exit(main())
