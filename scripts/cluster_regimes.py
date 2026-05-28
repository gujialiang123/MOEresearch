#!/usr/bin/env python3
"""Stage 1: cluster suspicious cases into regimes and emit a human-readable map.

For v0.2 MVP this is rule-based:

  1. Group passed runs by `regime_hint` (the seed-provided label).
  2. Within each group, summarize: count, median primary metric, top tail ratio,
     top suspicion score.
  3. Failed runs become their own "failure" cluster with the failure kind in
     the label.

Writes:
  regime_scout/outputs/regime_map.md     human-readable
  regime_scout/outputs/regime_map.json   machine-readable (used by select_cases)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from utils import load_yaml, read_jsonl, save_json


def median(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def build_clusters(raw_rows: list[dict], scored_rows: list[dict]) -> list[dict]:
    score_by_run = {s["run_id"]: s for s in scored_rows}
    by_hint: dict[str, list[dict]] = defaultdict(list)
    for r in raw_rows:
        by_hint[r.get("regime_hint") or "unknown"].append(r)

    clusters = []
    for hint, members in by_hint.items():
        passed = [m for m in members if m.get("status") == "pass"]
        failed = [m for m in members if m.get("status") == "fail"]

        primary_metric = None
        primary_dir = None
        if passed:
            top = score_by_run.get(passed[0]["run_id"], {})
            primary_metric = top.get("primary_metric")
            primary_dir = top.get("primary_direction")

        primary_vals = []
        ttft_p99_p50_ratios = []
        scores = []
        for m in passed:
            sc = score_by_run.get(m["run_id"], {})
            scores.append(sc.get("score") or 0.0)
            primary_vals.append(sc.get("primary_value"))
            metrics = m.get("metrics") or {}
            p50 = metrics.get("ttft_p50_ms")
            p99 = metrics.get("ttft_p99_ms")
            if p50 and p99 and p50 > 0:
                ttft_p99_p50_ratios.append(p99 / p50)

        clusters.append({
            "cluster_id": f"R_{hint}",
            "regime_hint": hint,
            "passed": len(passed),
            "failed": len(failed),
            "members": [m["workload_name"] for m in members],
            "passed_members": [m["workload_name"] for m in passed],
            "failed_members": [
                {"workload_name": m["workload_name"], "error": m.get("error")}
                for m in failed
            ],
            "primary_metric": primary_metric,
            "primary_direction": primary_dir,
            "median_primary": median(primary_vals),
            "max_score": max(scores) if scores else None,
            "median_ttft_p99_over_p50": median(ttft_p99_p50_ratios),
            "top_workload_by_score": (
                max(passed, key=lambda m: score_by_run.get(m["run_id"], {}).get("score") or 0.0)
                ["workload_name"]
                if passed else None
            ),
        })

    # sort clusters: most-suspicious first
    clusters.sort(key=lambda c: c.get("max_score") or 0.0, reverse=True)
    return clusters


def render_md(clusters: list[dict], scored_rows: list[dict], meta: dict) -> str:
    lines = []
    lines.append("# Regime Map\n")
    lines.append(f"- Generated at: {meta.get('generated_at')}")
    lines.append(f"- Server config: `{meta.get('server_config')}`")
    lines.append(f"- Model: `{meta.get('model')}`")
    lines.append(f"- Hardware: {meta.get('hardware')}")
    lines.append(f"- SGLang version: {meta.get('sglang_version', '(unknown)')}")
    lines.append("")
    lines.append("## Overview\n")
    total = sum(c["passed"] + c["failed"] for c in clusters)
    n_pass = sum(c["passed"] for c in clusters)
    n_fail = sum(c["failed"] for c in clusters)
    lines.append(f"- Workloads run: **{total}**")
    lines.append(f"- Passed: **{n_pass}**")
    lines.append(f"- Failed: **{n_fail}**")
    lines.append(f"- Clusters: **{len(clusters)}**")
    lines.append("")

    lines.append("## Regime clusters\n")
    for c in clusters:
        lines.append(f"### {c['cluster_id']}  ({c['regime_hint']})\n")
        lines.append(f"- Passed members ({c['passed']}): {', '.join(c['passed_members']) or '(none)'}")
        if c["failed_members"]:
            fm = ", ".join(f"{m['workload_name']}({m['error']})" for m in c["failed_members"])
            lines.append(f"- Failed members ({c['failed']}): {fm}")
        if c["primary_metric"]:
            lines.append(f"- Primary metric: `{c['primary_metric']}` ({c['primary_direction']}-is-better)")
        if c["median_primary"] is not None:
            lines.append(f"- Median primary: **{c['median_primary']:.2f}**")
        if c["median_ttft_p99_over_p50"] is not None:
            lines.append(f"- Median TTFT p99/p50 ratio: **{c['median_ttft_p99_over_p50']:.2f}**")
        if c["max_score"] is not None:
            lines.append(f"- Max suspicion score: **{c['max_score']:.3f}**")
        if c["top_workload_by_score"]:
            lines.append(f"- Top workload by score: `{c['top_workload_by_score']}`")
        lines.append("")

    lines.append("## Top suspicious workloads (overall)\n")
    lines.append("| Rank | Workload | Regime | Status | Score | Primary metric | Value |")
    lines.append("|---:|---|---|---|---:|---|---:|")
    for i, s in enumerate(scored_rows[:15], start=1):
        v = s.get("primary_value")
        v_s = f"{v:.2f}" if isinstance(v, (int, float)) else "—"
        sc = s.get("score")
        sc_s = f"{sc:.3f}" if isinstance(sc, (int, float)) else "—"
        lines.append(f"| {i} | `{s['workload_name']}` | {s['regime_hint']} | "
                     f"{s['status']} | {sc_s} | {s['primary_metric']} | {v_s} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw",          default="regime_scout/outputs/raw_results.jsonl")
    ap.add_argument("--suspicious",   default="regime_scout/outputs/suspicious_cases.jsonl")
    ap.add_argument("--server-config",default="configs/base.yaml")
    ap.add_argument("--out-md",       default="regime_scout/outputs/regime_map.md")
    ap.add_argument("--out-json",     default="regime_scout/outputs/regime_map.json")
    args = ap.parse_args()

    raw = read_jsonl(args.raw)
    if not raw:
        print(f"[cluster_regimes] no raw rows in {args.raw}", file=sys.stderr)
        return 1
    scored = read_jsonl(args.suspicious)

    clusters = build_clusters(raw, scored)

    server_cfg = load_yaml(args.server_config) if Path(args.server_config).exists() else {}
    meta = {
        "generated_at": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "server_config": args.server_config,
        "model": server_cfg.get("model-path"),
        "hardware": "H200",
    }

    out_json = {
        "meta": meta,
        "clusters": clusters,
    }
    save_json(out_json, args.out_json)

    md = render_md(clusters, scored, meta)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(md)

    print(f"[cluster_regimes] {len(clusters)} clusters → {args.out_md}")
    print(f"[cluster_regimes] {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
