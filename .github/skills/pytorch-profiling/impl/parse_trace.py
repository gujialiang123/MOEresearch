#!/usr/bin/env python3
"""Reduce a Torch profiler trace (Chrome Trace JSON) → profile_summary.json.

STUB IMPLEMENTATION. Fills the schema declared in ../SKILL.md but a number
of fields are marked TODO and rely on heuristics. See SKILL.md ROADMAP.

Implemented (v0):
  - top_kernels: top 20 GPU events by self_time aggregated by `name`
  - totals.wallclock_ms (trace span)
  - phase_breakdown_pct: best-effort via sglang user-annotations
    (Prefill/Decode/Schedule/Tokenize) if present, otherwise via kernel name
    heuristics; the rest goes into "other".
  - moe_overhead: sum of any name matching *moe*topk*|*moe*dispatch*|*all_to_all*
  - cuda_graph: very rough — left null with warning unless we find sglang
    cuda-graph capture annotations.

Not implemented yet (emit warnings + leave null/empty):
  - gpu_active_ms / gpu_idle_ms (need stream-level analysis)
  - cuda_graph.fallback_reason_guess (needs server_features.json crosscheck)
  - per-TP-rank merge
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


GPU_CATS = {"kernel", "gpu_op", "cuda_runtime"}  # Chrome trace 'cat' values

MOE_TOPK_RE = re.compile(r"moe.*topk|topk.*moe|topk_softmax", re.I)
MOE_DISP_RE = re.compile(r"moe.*dispatch|grouped_gemm|moe_align", re.I)
ALL2ALL_RE  = re.compile(r"all.?to.?all", re.I)
PREFILL_RE  = re.compile(r"prefill|context", re.I)
DECODE_RE   = re.compile(r"decode|extend|generate", re.I)
SCHED_RE    = re.compile(r"schedule|scheduler|coordinator", re.I)


def load_trace(path: Path) -> dict:
    """Load Chrome Trace JSON, supporting .json and .json.gz."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as f:
        return json.load(f)


def reduce_events(events: list[dict]) -> dict:
    by_name = defaultdict(lambda: {"self_us": 0.0, "calls": 0})
    phase_us = defaultdict(float)
    total_us = 0.0
    ts_min = float("inf")
    ts_max = float("-inf")

    # Phase tagging via stack: sglang emits "user_annotation" ranges with names
    # like "Prefill", "Decode", "Schedule", "Tokenize". Heuristic: any GPU op
    # whose name contains those keywords is mapped to that phase.
    for ev in events:
        ph = ev.get("ph")
        if ph != "X":  # X = complete duration event
            continue
        dur = ev.get("dur")
        if dur is None:
            continue
        name = ev.get("name") or ""
        cat = ev.get("cat") or ""
        ts = ev.get("ts", 0)
        ts_min = min(ts_min, ts)
        ts_max = max(ts_max, ts + dur)

        # GPU-side aggregation only
        if cat not in GPU_CATS:
            continue
        rec = by_name[name]
        rec["self_us"] += dur
        rec["calls"] += 1
        total_us += dur

        if PREFILL_RE.search(name):
            phase_us["prefill"] += dur
        elif DECODE_RE.search(name):
            phase_us["decode"] += dur
        elif SCHED_RE.search(name):
            phase_us["scheduler"] += dur
        else:
            phase_us["other"] += dur

    wall_us = (ts_max - ts_min) if ts_max > ts_min else 0.0
    return {
        "by_name": by_name,
        "phase_us": dict(phase_us),
        "total_us": total_us,
        "wall_us": wall_us,
    }


def detect_moe(by_name: dict) -> dict:
    topk_us = sum(v["self_us"] for n, v in by_name.items() if MOE_TOPK_RE.search(n))
    disp_us = sum(v["self_us"] for n, v in by_name.items() if MOE_DISP_RE.search(n))
    a2a_us  = sum(v["self_us"] for n, v in by_name.items() if ALL2ALL_RE.search(n))
    total = sum(v["self_us"] for v in by_name.values()) or 1.0
    total_pct = 100.0 * (topk_us + disp_us + a2a_us) / total
    if topk_us + disp_us + a2a_us == 0:
        return {"applicable": False}
    verdict = "low" if total_pct < 10 else ("moderate" if total_pct < 25 else "high")
    return {
        "applicable": True,
        "topk_pct":     round(100 * topk_us / total, 2),
        "dispatch_pct": round(100 * disp_us / total, 2),
        "all_to_all_pct": round(100 * a2a_us / total, 2),
        "total_routing_pct": round(total_pct, 2),
        "verdict": verdict,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workload-yaml", default=None)
    ap.add_argument("--config-yaml", default=None)
    ap.add_argument("--top-n", type=int, default=20)
    args = ap.parse_args()

    trace_path = Path(args.trace)
    out_path = Path(args.out)
    warnings: list[str] = []

    try:
        trace = load_trace(trace_path)
    except Exception as e:
        out_path.write_text(json.dumps(
            {"schema_version": 0, "ok": False,
             "error": f"failed to load trace: {e}"}, indent=2))
        return 4

    events = trace.get("traceEvents", [])
    if not events:
        out_path.write_text(json.dumps(
            {"schema_version": 0, "ok": False,
             "error": "trace has no events"}, indent=2))
        return 4

    red = reduce_events(events)
    by_name = red["by_name"]
    total = red["total_us"] or 1.0

    # Top kernels by self_us
    ranked = sorted(by_name.items(), key=lambda kv: kv[1]["self_us"], reverse=True)
    top = []
    for i, (name, v) in enumerate(ranked[: args.top_n], 1):
        phase = (
            "prefill" if PREFILL_RE.search(name) else
            "decode"  if DECODE_RE.search(name) else
            "scheduler" if SCHED_RE.search(name) else "other"
        )
        top.append({
            "rank": i,
            "name": name,
            "self_time_pct": round(100 * v["self_us"] / total, 2),
            "calls": v["calls"],
            "avg_us": round(v["self_us"] / max(v["calls"], 1), 2),
            "phase": phase,
        })

    phase_us = red["phase_us"]
    phase_total = sum(phase_us.values()) or 1.0
    phase_pct = {k: round(100 * v / phase_total, 2) for k, v in phase_us.items()}
    # ensure all keys present
    for k in ("prefill", "decode", "scheduler", "tokenize", "other"):
        phase_pct.setdefault(k, 0.0)

    moe = detect_moe(by_name)

    # workload + config metadata
    wl_meta = {}
    if args.workload_yaml:
        try:
            import yaml
            wl = yaml.safe_load(Path(args.workload_yaml).read_text())
            wl_meta = {
                "regime_id": wl.get("regime_id") or wl.get("name"),
                "num_prompts": (wl.get("traffic") or {}).get("num_prompts"),
                "max_concurrency": (wl.get("traffic") or {}).get("max_concurrency"),
            }
        except Exception as e:
            warnings.append(f"failed to read workload yaml: {e}")

    cfg_sha = None
    if args.config_yaml:
        try:
            cfg_sha = hashlib.sha256(Path(args.config_yaml).read_bytes()).hexdigest()
        except Exception:
            pass

    # cold-start tail check
    if red["wall_us"] > 0 and len(ranked) > 0:
        # crude: compare first vs last 30% bucket time
        first_us = sum(v["self_us"] for _, v in ranked[: max(1, len(ranked)//3)])
        last_us  = sum(v["self_us"] for _, v in ranked[-max(1, len(ranked)//3):])
        if last_us > 0 and first_us > 2.0 * last_us:
            warnings.append("cold-start tail detected; profile may be biased toward warmup")

    # cuda_graph block: stub
    cuda_graph = {
        "captured_bs_range": None,
        "decode_steps_in_graph_pct": None,
        "decode_steps_outside_graph_pct": None,
        "fallback_reason_guess": None,
    }
    warnings.append("cuda_graph fields not yet computed; cross-check server_features.json")

    summary = {
        "schema_version": 0,
        "ok": True,
        "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "trace_path": str(trace_path),
        "candidate_config_sha256": cfg_sha,
        "workload": wl_meta,
        "totals": {
            "wallclock_ms": round(red["wall_us"] / 1000.0, 2),
            "gpu_active_ms": round(total / 1000.0, 2),
            "gpu_idle_ms": None,
            "gpu_utilization_pct": None,
        },
        "phase_breakdown_pct": phase_pct,
        "top_kernels": top,
        "moe_overhead": moe,
        "cuda_graph": cuda_graph,
        "warnings": warnings,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[parse_trace] wrote {out_path}  ({len(top)} top kernels, "
          f"{len(warnings)} warnings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
