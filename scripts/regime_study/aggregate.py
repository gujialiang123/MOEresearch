#!/usr/bin/env python3
"""Aggregate regime-study runs across models + repeats into CSV + Markdown reports.

Inputs:
  results/regime_bench/raw/<model>_rep<N>.jsonl   — one row per workload
  experiments/tmp/regime_study/<model>_rep<N>/run_*/server.log  — server logs

Outputs:
  results/regime_bench/parsed_results.csv    — one row per (model,regime,rep)
  results/regime_bench/summary_table.csv     — aggregated mean over reps
  results/regime_bench/summary.md            — meeting-ready md
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT / ".github" / "skills" / "server-log-mining" / "impl"))

from utils import load_yaml, read_jsonl  # noqa: E402

try:
    from parse_server_log import mine  # type: ignore  # noqa: E402
except ImportError:
    mine = None  # gracefully degrade


REGIME_ORDER = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]


def regime_id(name: str) -> str:
    return name.split("_", 1)[0] if name else "?"


def load_workload_descriptors(dir_path: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(dir_path.glob("*.yaml")):
        d = load_yaml(p)
        rid = regime_id(d.get("name", p.stem))
        out[rid] = d
    return out


def mine_or_default(server_log: Path) -> dict:
    if mine is None or not server_log.exists():
        return {}
    try:
        res = mine(server_log)
        return res.get("fields", {}) if isinstance(res, dict) else {}
    except Exception as e:  # noqa: BLE001
        return {"_mine_error": str(e)}


def fmt_input_len(ds: dict) -> str:
    if ds.get("name") == "generated-shared-prefix":
        sys_p = ds.get("gsp_system_prompt_len", 0)
        q = ds.get("gsp_question_len", 0)
        return f"{sys_p}+{q}"
    rir = ds.get("random_input_len")
    rr = ds.get("random_range_ratio", 0.0)
    if rir is None:
        return "?"
    return f"{rir}" if not rr else f"{rir}±{int(rr*100)}%"


def fmt_output_len(ds: dict) -> str:
    if ds.get("name") == "generated-shared-prefix":
        return str(ds.get("gsp_output_len", "?"))
    return str(ds.get("random_output_len", "?"))


def collect_rows(raw_dir: Path, regime_dir: Path, run_root: Path) -> list[dict]:
    """Read all <model>_rep<N>.jsonl files and produce per-run rows."""
    descriptors = load_workload_descriptors(regime_dir)
    rows: list[dict] = []
    for jsonl in sorted(raw_dir.glob("*_rep*.jsonl")):
        stem = jsonl.stem  # e.g. "dense_rep1"
        try:
            model_tag, rep_tag = stem.rsplit("_", 1)
            rep = int(rep_tag.replace("rep", ""))
        except (ValueError, IndexError):
            continue
        for entry in read_jsonl(jsonl):
            name = entry.get("workload_name") or ""
            rid = regime_id(name)
            desc = descriptors.get(rid, {})
            ds = desc.get("dataset", {})
            traffic = desc.get("traffic", {})

            m = entry.get("metrics") or {}
            server_log = (Path(entry["run_dir"]) / "server.log") if entry.get("run_dir") else Path()
            sf = mine_or_default(server_log)

            row = {
                "model": model_tag,
                "regime_id": rid,
                "regime_name": name,
                "regime_hint": entry.get("regime_hint"),
                "rep": rep,
                "input_len_spec": fmt_input_len(ds),
                "output_len_spec": fmt_output_len(ds),
                "dataset_name": ds.get("name"),
                "max_concurrency": traffic.get("max_concurrency"),
                "num_prompts": traffic.get("num_prompts"),
                "request_rate": traffic.get("request_rate", "inf"),
                "status": entry.get("status"),
                "error": entry.get("error"),
                "duration_s": entry.get("duration_s"),
                # metrics
                "completed": m.get("completed"),
                "failed_requests": m.get("failed_requests"),
                "request_throughput": m.get("request_throughput"),
                "input_throughput": m.get("input_throughput"),
                "output_throughput": m.get("output_throughput"),
                "ttft_mean_ms": m.get("ttft_mean_ms"),
                "ttft_p50_ms": m.get("ttft_p50_ms"),
                "ttft_p95_ms": m.get("ttft_p95_ms"),
                "ttft_p99_ms": m.get("ttft_p99_ms"),
                "tpot_mean_ms": m.get("tpot_mean_ms"),
                "tpot_p50_ms": m.get("tpot_p50_ms"),
                "tpot_p99_ms": m.get("tpot_p99_ms"),
                "itl_p50_ms": m.get("itl_p50_ms"),
                "itl_p95_ms": m.get("itl_p95_ms"),
                "itl_p99_ms": m.get("itl_p99_ms"),
                "e2e_mean_ms": m.get("e2e_mean_ms"),
                "e2e_p50_ms": m.get("e2e_p50_ms"),
                "e2e_p99_ms": m.get("e2e_p99_ms"),
                "oom": m.get("oom"),
                "server_crash": m.get("server_crash"),
                # server-log features
                "attention_backend": sf.get("attention_backend"),
                "schedule_policy": sf.get("schedule_policy"),
                "chunked_prefill_size": sf.get("chunked_prefill_size"),
                "max_prefill_tokens": sf.get("max_prefill_tokens"),
                "max_running_requests": sf.get("max_running_requests"),
                "mem_fraction_static": sf.get("mem_fraction_static"),
                "kv_cache_size_gb_total": sf.get("kv_cache_size_gb_total"),
                "max_total_num_tokens": sf.get("max_total_num_tokens"),
                "cuda_graph_bs_captured_max": (
                    max(sf.get("cuda_graph_bs_captured") or [0]) or None
                ),
                "peak_running_reqs": sf.get("peak_running_reqs"),
                "peak_queue_reqs": sf.get("peak_queue_reqs"),
                "prefill_batch_count": sf.get("prefill_batch_count"),
                "decode_batch_count": sf.get("decode_batch_count"),
                "kv_pool_full_events": sf.get("kv_pool_full_events"),
                "retract_events": sf.get("retract_events"),
                "concurrency_capped": sf.get("concurrency_capped"),
                "at_capacity": sf.get("at_capacity"),
            }
            rows.append(row)
    return rows


def mean_or_none(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.mean(xs), 4) if xs else None


def aggregate(rows: list[dict]) -> list[dict]:
    """Aggregate over reps. Per (model, regime_id)."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["model"], r["regime_id"]), []).append(r)
    agg = []
    for (model, rid), grp in groups.items():
        passed = [r for r in grp if r["status"] == "pass"]
        any_row = passed[0] if passed else grp[0]
        out = {
            "model": model,
            "regime_id": rid,
            "regime_name": any_row["regime_name"],
            "regime_hint": any_row["regime_hint"],
            "n_reps": len(grp),
            "n_pass": len(passed),
            "input_len_spec": any_row["input_len_spec"],
            "output_len_spec": any_row["output_len_spec"],
            "max_concurrency": any_row["max_concurrency"],
            "num_prompts": any_row["num_prompts"],
            "dataset_name": any_row["dataset_name"],
            "errors": ";".join(sorted({str(r["error"]) for r in grp if r["error"]})) or None,
        }
        for k in [
            "request_throughput", "input_throughput", "output_throughput",
            "ttft_mean_ms", "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms",
            "tpot_mean_ms", "tpot_p50_ms", "tpot_p99_ms",
            "itl_p50_ms", "itl_p95_ms", "itl_p99_ms",
            "e2e_mean_ms", "e2e_p50_ms", "e2e_p99_ms",
        ]:
            out[k] = mean_or_none([r[k] for r in passed])
        # server-side stats: take from any passed run (constant within model)
        for k in [
            "attention_backend", "schedule_policy", "chunked_prefill_size",
            "max_prefill_tokens", "max_running_requests", "mem_fraction_static",
            "kv_cache_size_gb_total", "max_total_num_tokens",
            "cuda_graph_bs_captured_max",
        ]:
            out[k] = any_row.get(k)
        # per-regime runtime observed:
        out["peak_running_reqs"] = mean_or_none([r["peak_running_reqs"] for r in passed])
        out["peak_queue_reqs"] = mean_or_none([r["peak_queue_reqs"] for r in passed])
        out["prefill_batch_count"] = mean_or_none(
            [r["prefill_batch_count"] for r in passed])
        out["decode_batch_count"] = mean_or_none(
            [r["decode_batch_count"] for r in passed])
        out["concurrency_capped"] = any(
            bool(r.get("concurrency_capped")) for r in passed)
        out["retract_events"] = sum(int(r.get("retract_events") or 0) for r in passed)
        out["kv_pool_full_events"] = sum(
            int(r.get("kv_pool_full_events") or 0) for r in passed)
        agg.append(out)
    # stable ordering: model, then regime
    agg.sort(key=lambda r: (r["model"], REGIME_ORDER.index(r["regime_id"])
                            if r["regime_id"] in REGIME_ORDER else 99))
    return agg


def pct(new, base):
    if new is None or base is None or base == 0:
        return None
    return round((new - base) / base * 100, 2)


def compute_gaps(agg: list[dict], baseline_id: str = "R1") -> list[dict]:
    """Append gap-vs-baseline and gap-vs-best columns."""
    by_model: dict[str, list[dict]] = {}
    for r in agg:
        by_model.setdefault(r["model"], []).append(r)
    out: list[dict] = []
    for model, rows in by_model.items():
        baseline = next((r for r in rows if r["regime_id"] == baseline_id), None)
        passed = [r for r in rows if r["n_pass"] > 0]
        if not passed:
            out.extend(rows)
            continue
        best_thr = max(passed, key=lambda r: r.get("output_throughput") or -1)
        best_ttft = min(
            [r for r in passed if r.get("ttft_p50_ms") is not None],
            key=lambda r: r["ttft_p50_ms"], default=None,
        )
        best_tpot = min(
            [r for r in passed if r.get("tpot_mean_ms") is not None],
            key=lambda r: r["tpot_mean_ms"], default=None,
        )
        for r in rows:
            r2 = dict(r)
            if baseline is not None:
                r2["throughput_gap_vs_R1_pct"] = pct(
                    r.get("output_throughput"), baseline.get("output_throughput"))
                r2["ttft_gap_vs_R1_pct"] = pct(
                    r.get("ttft_mean_ms"), baseline.get("ttft_mean_ms"))
                r2["tpot_gap_vs_R1_pct"] = pct(
                    r.get("tpot_mean_ms"), baseline.get("tpot_mean_ms"))
                r2["p95_lat_gap_vs_R1_pct"] = pct(
                    r.get("e2e_p99_ms"), baseline.get("e2e_p99_ms"))
            else:
                for k in ("throughput_gap_vs_R1_pct", "ttft_gap_vs_R1_pct",
                          "tpot_gap_vs_R1_pct", "p95_lat_gap_vs_R1_pct"):
                    r2[k] = None
            r2["throughput_vs_best_pct"] = pct(
                r.get("output_throughput"), best_thr.get("output_throughput"))
            r2["ttft_vs_best_pct"] = pct(
                r.get("ttft_p50_ms"),
                best_ttft.get("ttft_p50_ms") if best_ttft else None)
            r2["tpot_vs_best_pct"] = pct(
                r.get("tpot_mean_ms"),
                best_tpot.get("tpot_mean_ms") if best_tpot else None)
            out.append(r2)
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})


def fmt(v, prec=1):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{prec}f}"
    return str(v)


def fmt_pct(v):
    if v is None:
        return "n/a"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.1f}%"


def write_summary_md(agg: list[dict], path: Path) -> None:
    by_model: dict[str, list[dict]] = {}
    for r in agg:
        by_model.setdefault(r["model"], []).append(r)

    lines: list[str] = []
    lines.append("# Regime benchmark — meeting summary")
    lines.append("")
    lines.append("> Generated by `scripts/regime_study/aggregate.py`.")
    lines.append("> Same `configs/base.yaml` / `configs/moe_qwen3_30b.yaml` "
                 "for all regimes — no per-regime config tuning.")
    lines.append("")

    # Per-model big table
    for model in sorted(by_model.keys()):
        rows = by_model[model]
        lines.append(f"## Model: `{model}`")
        lines.append("")
        # backend / config summary card
        any_r = rows[0]
        lines.append(
            f"- Attention backend: **{fmt(any_r.get('attention_backend'))}** · "
            f"schedule policy: **{fmt(any_r.get('schedule_policy'))}** · "
            f"chunked_prefill_size: **{fmt(any_r.get('chunked_prefill_size'))}** · "
            f"max_prefill_tokens: **{fmt(any_r.get('max_prefill_tokens'))}** · "
            f"max_running_requests: **{fmt(any_r.get('max_running_requests'))}** · "
            f"mem_fraction_static: **{fmt(any_r.get('mem_fraction_static'))}**"
        )
        lines.append(
            f"- KV cache total: **{fmt(any_r.get('kv_cache_size_gb_total'))} GB**, "
            f"max_total_num_tokens: **{fmt(any_r.get('max_total_num_tokens'))}**, "
            f"cuda_graph captured up to bs **{fmt(any_r.get('cuda_graph_bs_captured_max'))}**"
        )
        lines.append("")
        # Table 1 — per-regime metrics
        lines.append("| Regime | InLen | OutLen | Conc | NumPrompts | Req/s | Out tok/s | TTFT mean | TTFT p95 | TPOT mean | ITL p95 | E2E p99 | n_pass | Gap vs R1 (out tok/s) | Notes |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            notes = []
            if r["n_pass"] == 0:
                notes.append(f"FAIL: {r.get('errors')}")
            if r.get("concurrency_capped"):
                notes.append("hit max_running cap")
            if r.get("retract_events"):
                notes.append(f"{r['retract_events']} retracts")
            if r.get("kv_pool_full_events"):
                notes.append(f"{r['kv_pool_full_events']} KV-pool-full")
            gap = r.get("throughput_gap_vs_R1_pct")
            lines.append(
                f"| {r['regime_id']} | "
                f"{r['input_len_spec']} | {r['output_len_spec']} | "
                f"{r['max_concurrency']} | {r['num_prompts']} | "
                f"{fmt(r.get('request_throughput'), 2)} | "
                f"{fmt(r.get('output_throughput'), 0)} | "
                f"{fmt(r.get('ttft_mean_ms'), 1)} | "
                f"{fmt(r.get('ttft_p95_ms'), 1)} | "
                f"{fmt(r.get('tpot_mean_ms'), 2)} | "
                f"{fmt(r.get('itl_p95_ms'), 2)} | "
                f"{fmt(r.get('e2e_p99_ms'), 0)} | "
                f"{r['n_pass']}/{r['n_reps']} | "
                f"{fmt_pct(gap)} | "
                f"{'; '.join(notes) if notes else ''} |"
            )
        lines.append("")

        # Table 2 — best vs worst
        passed = [r for r in rows if r["n_pass"] > 0]
        if passed:
            best = max(passed, key=lambda r: r.get("output_throughput") or -1)
            worst = min(passed, key=lambda r: r.get("output_throughput") or float("inf"))
            best_ttft = min(
                [r for r in passed if r.get("ttft_p50_ms") is not None],
                key=lambda r: r["ttft_p50_ms"], default=None,
            )
            worst_ttft = max(
                [r for r in passed if r.get("ttft_p50_ms") is not None],
                key=lambda r: r["ttft_p50_ms"], default=None,
            )
            lines.append("**Best/worst summary**:")
            lines.append(f"- Highest output throughput: **{best['regime_id']}** "
                         f"({fmt(best.get('output_throughput'),0)} tok/s)")
            lines.append(f"- Lowest output throughput: **{worst['regime_id']}** "
                         f"({fmt(worst.get('output_throughput'),0)} tok/s)")
            if best_ttft and worst_ttft:
                lines.append(f"- Lowest TTFT p50: **{best_ttft['regime_id']}** "
                             f"({fmt(best_ttft.get('ttft_p50_ms'),1)} ms)")
                lines.append(f"- Highest TTFT p50: **{worst_ttft['regime_id']}** "
                             f"({fmt(worst_ttft.get('ttft_p50_ms'),1)} ms)")
            lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="results/regime_bench/raw")
    ap.add_argument("--regime-dir", default="regime_scout/candidates_regime_study")
    ap.add_argument("--run-root", default="experiments/tmp/regime_study")
    ap.add_argument("--out-dir", default="results/regime_bench")
    ap.add_argument("--baseline", default="R1")
    args = ap.parse_args()

    raw_dir = PROJECT_ROOT / args.raw_dir
    regime_dir = PROJECT_ROOT / args.regime_dir
    run_root = PROJECT_ROOT / args.run_root
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(raw_dir, regime_dir, run_root)
    if not rows:
        print(f"[aggregate] no rows found under {raw_dir}", file=sys.stderr)
        return 1

    agg = aggregate(rows)
    agg_with_gaps = compute_gaps(agg, baseline_id=args.baseline)

    write_csv(rows, out_dir / "parsed_results.csv")
    write_csv(agg_with_gaps, out_dir / "summary_table.csv")
    write_summary_md(agg_with_gaps, out_dir / "summary.md")

    print(f"[aggregate] {len(rows)} per-run rows → parsed_results.csv")
    print(f"[aggregate] {len(agg_with_gaps)} (model,regime) → summary_table.csv")
    print(f"[aggregate] markdown → summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
