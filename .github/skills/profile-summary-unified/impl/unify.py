#!/usr/bin/env python3
"""Skill impl: profile-summary-unified.

Merges outputs from upstream profiling skills into one canonical
profile_unified.json. See ../SKILL.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 0


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _load_yaml(p: Path):
    import yaml
    return yaml.safe_load(p.read_text())


# ---------------------------------------------------------------------------
# Adapters — each returns (partial_dict, list_of_evidence_chain_entries)
# ---------------------------------------------------------------------------

def _from_bench_summary(path: Path, regime_id: str):
    """Pull e2e block from e2e-bench-runner output."""
    d = json.loads(path.read_text())
    if not d.get("ok"):
        return ({}, [{"field": "e2e", "source_skill": "e2e-bench-runner",
                      "source_file": str(path), "ok": False,
                      "gap_reason": d.get("error", "bench_summary not ok")}])
    r = (d.get("regimes") or {}).get(regime_id)
    if not r:
        return ({}, [{"field": "e2e", "source_skill": "e2e-bench-runner",
                      "source_file": str(path), "ok": False,
                      "gap_reason": f"regime '{regime_id}' not in bench_summary"}])
    rs = r.get("req_per_s", {})
    e2e = {
        "req_per_s": {
            "mean": rs.get("mean"),
            "stddev_pct": rs.get("stddev_pct"),
            "reliable": r.get("reliable"),
        },
        "tokens_per_s_mean": (r.get("tokens_per_s") or {}).get("mean"),
        "e2e_p50_ms": (r.get("e2e_ms") or {}).get("p50"),
        "e2e_p99_ms": (r.get("e2e_ms") or {}).get("p99"),
        "completion_rate": r.get("completion_rate"),
    }
    return ({"e2e": e2e},
            [{"field": "e2e.req_per_s", "source_skill": "e2e-bench-runner",
              "source_file": str(path), "ok": True}])


def _from_timeline_summary(path: Path):
    """Pull gpu_macro + kernel_breakdown from nsys-timeline-sql output."""
    d = json.loads(path.read_text())
    if not d.get("ok"):
        return ({}, [{"field": "gpu_macro", "source_skill": "nsys-timeline-sql",
                      "source_file": str(path), "ok": False,
                      "gap_reason": d.get("error", "timeline_summary not ok")}])
    totals = d.get("totals_primary_stream") or {}
    api = d.get("cuda_api") or {}
    gpu_macro = {
        "wall_ms": (d.get("wall_ns") or 0) / 1e6,
        "gpu_active_ms": totals.get("gpu_active_ms"),
        "gpu_idle_ms":   totals.get("gpu_idle_ms"),
        "gpu_util_pct":  totals.get("gpu_util_pct"),
        "launch_count":  totals.get("kernel_count"),
        "launch_ratio_graph_to_eager": api.get("launch_ratio_graph_to_eager"),
        "verdict": api.get("verdict"),
    }
    # kernel breakdown: take top kernels from timeline_summary
    kernels = []
    for k in d.get("top_kernels") or []:
        kernels.append({
            "category": _categorize_kernel(k["short_name"]),
            "kernel_pattern": k["short_name"],
            "self_ms": k.get("self_ms"),
            "self_pct": k.get("self_pct_of_active"),
            "calls": k.get("calls"),
            "avg_us": k.get("avg_us"),
            "source": "nsys-timeline-sql",
        })
    return ({"gpu_macro": gpu_macro, "kernel_breakdown": kernels},
            [{"field": "gpu_macro", "source_skill": "nsys-timeline-sql",
              "source_file": str(path), "ok": True},
             {"field": "kernel_breakdown", "source_skill": "nsys-timeline-sql",
              "source_file": str(path), "ok": True}])


def _from_torch_profile_text(path: Path):
    """Parse vLLM-style torch.profiler text summary into kernel_breakdown."""
    records = []
    for line in path.read_text().splitlines():
        m = re.match(
            r'^\s*(.+?)\s+(\d+\.\d+%)\s+(\S+)\s+(\d+\.\d+%)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+\.\d+%)\s+(\S+)\s+(\S+)\s+(\d+)\s*$',
            line)
        if not m: continue
        name = m.group(1).strip()
        # Skip wrappers / aten / cuda runtime APIs (only keep GPU kernels)
        SKIP = ("execute_context_", "vllm::moe_forward", "aten::", "_vllm_fa3_C::",
                "_C_cache_ops::", "_moe_C::", "cudaEvent", "cudaStream",
                "cuLaunchKernelEx", "Lazy Function", "Buffer Flush", "Activity Buffer")
        if any(name.startswith(s) for s in SKIP): continue
        # Column order: name | Self_CPU% | Self_CPU | CPU_total% | CPU_total | CPU_avg
        #              | Self_CUDA | Self_CUDA% | CUDA_total | CUDA_avg | #calls
        # Groups:        1            2          3            4         5      6
        #                7            8          9            10        11
        records.append({
            "name": name,
            "self_pct": float(m.group(8).rstrip('%')),
            "calls": int(m.group(11)),
            "self_us": _parse_time_us(m.group(7)),  # Self CUDA
            "avg_us":  _parse_time_us(m.group(10)),  # CUDA time avg
        })
    if not records:
        return ({}, [{"field": "kernel_breakdown",
                      "source_skill": "torch.profiler",
                      "source_file": str(path), "ok": False,
                      "gap_reason": "no parseable kernel rows"}])

    # Aggregate by category
    from collections import defaultdict
    by_cat = defaultdict(lambda: {"self_pct": 0.0, "calls": 0, "self_us": 0.0,
                                  "kernels": []})
    for r in records:
        cat = _categorize_kernel(r["name"])
        by_cat[cat]["self_pct"] += r["self_pct"]
        by_cat[cat]["calls"]    += r["calls"]
        by_cat[cat]["self_us"]  += r["self_us"]
        by_cat[cat]["kernels"].append(r["name"])

    kernels = []
    for cat, agg in sorted(by_cat.items(), key=lambda x: -x[1]["self_pct"]):
        kernels.append({
            "category": cat,
            "kernel_pattern": agg["kernels"][0] if agg["kernels"] else "",
            "self_ms": agg["self_us"] / 1000.0,
            "self_pct": agg["self_pct"],
            "calls": agg["calls"],
            "avg_us": (agg["self_us"] / agg["calls"]) if agg["calls"] else None,
            "source": "torch.profiler",
        })
    return ({"kernel_breakdown": kernels},
            [{"field": "kernel_breakdown", "source_skill": "torch.profiler+manual",
              "source_file": str(path), "ok": True,
              "note": "vllm-specific text parsing; categorization heuristic in unify.py"}])


def _from_sglang_profile(path: Path):
    """sglang's pytorch-profiling outputs profile_summary.json (already structured)."""
    d = json.loads(path.read_text())
    if not d.get("ok"):
        return ({}, [{"field": "gpu_macro", "source_skill": "pytorch-profiling",
                      "source_file": str(path), "ok": False,
                      "gap_reason": d.get("error")}])
    totals = d.get("totals") or {}
    gpu_macro = {
        "wall_ms": totals.get("wallclock_ms"),
        "gpu_active_ms": totals.get("gpu_active_ms"),
        "gpu_idle_ms":   totals.get("gpu_idle_ms"),
        "gpu_util_pct":  totals.get("gpu_utilization_pct"),
    }
    kernels = []
    for k in d.get("top_kernels") or []:
        kernels.append({
            "category": _categorize_kernel(k["name"]),
            "kernel_pattern": k["name"],
            "self_pct": k.get("self_time_pct"),
            "calls": k.get("calls"),
            "avg_us": k.get("avg_us"),
            "self_ms": (k.get("avg_us", 0) * k.get("calls", 0)) / 1000.0 if k.get("avg_us") else None,
            "source": "pytorch-profiling",
        })
    return ({"gpu_macro": gpu_macro, "kernel_breakdown": kernels},
            [{"field": "gpu_macro", "source_skill": "pytorch-profiling",
              "source_file": str(path), "ok": True},
             {"field": "kernel_breakdown", "source_skill": "pytorch-profiling",
              "source_file": str(path), "ok": True}])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_time_us(s: str) -> float:
    """Convert torch.profiler time string like '1.107s' or '350.000us' to µs."""
    s = s.strip()
    if s.endswith("us") or s.endswith("µs"):
        return float(s.rstrip("usµ").rstrip())
    if s.endswith("ms"):
        return float(s.rstrip("ms")) * 1000
    if s.endswith("s"):
        return float(s.rstrip("s")) * 1e6
    try:
        return float(s)
    except ValueError:
        return 0.0


def _categorize_kernel(name: str) -> str:
    """Map kernel name to one of the schema's category enum values.
    Names may be truncated by torch.profiler — match on prefixes / known substrings."""
    n = name.lower()
    # Note: truncated cutlass names often look like 'gemmuniv...' so match short prefix
    if "fused_moe::run_global" in name or "fused_moe_kernel" in name:
        return "moe_gemm"
    if ("trtllm_kernels" in name or "tensorrt_llm::kernels::cutlass_kernels" in name
        or "topkgating" in n or "topk_softmax" in n
        or "moe_align" in n or "expandinput" in n or "computestrides" in n
        or "fusedbuildexpertmaps" in n or "blockexp" in n or "finalizemoerout" in n
        or "mergeexpert" in n or "count_and_sort_expert" in n):
        return "moe_routing"
    if ("gemmuniv" in n or "nvjet_sm" in n or "splitkreduce" in n
        or "cublas" in n):
        return "dense_gemm"
    if ("flashattn" in n or "flash::" in n or "flash_fwd" in n
        or ("cutlass" in n and "flash" in n) or "attention" in n
        or "fa3" in n or "prepare_varlen" in n):
        return "attention"
    if name.startswith("triton_") and ("rms_norm" in n or "norm" in n):
        return "norm"
    if name.startswith("triton_"):
        return "elementwise"
    if "reshape_and_cache" in n or "kv_cache" in n:
        return "kv_cache"
    if "memcpy" in n or "memset" in n or "memcpy32" in n:
        return "memcpy"
    if ("elementwise" in n or "vectorized" in n or "reduce_kernel" in n
        or "unrolled" in n or "index_elementwise" in n):
        return "elementwise"
    return "other"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _from_ncu(path: Path):
    """Pull kernel_micro from ncu-microarch ncu_summary.json."""
    d = json.loads(path.read_text())
    if not d.get("ok"):
        return ({}, [{"field": "kernel_micro", "source_skill": "ncu-microarch",
                      "source_file": str(path), "ok": False,
                      "gap_reason": d.get("error", "ncu summary not ok")}])
    kernels = []
    for k in d.get("kernels") or []:
        m = k.get("metrics") or {}
        kernels.append({
            "kernel": k.get("kernel"),
            "sm_occupancy_pct":      m.get("warps_active_pct"),
            "achieved_flops_pct":    m.get("sm_throughput_pct"),
            "dram_bw_pct":           m.get("dram_throughput_pct"),
            "l1_hit_pct":            m.get("l1_hit_pct"),
            "l2_hit_pct":            m.get("l2_hit_pct"),
            "tensor_pipe_active":    m.get("tensor_pipe_active_pct"),
            "top_stall_reason":      f"long_scoreboard={m.get('stall_long_scoreboard_avg'):.2f}, "
                                     f"math_pipe={m.get('stall_math_pipe_throttle_avg'):.2f}"
                                     if (m.get('stall_long_scoreboard_avg') is not None
                                         and m.get('stall_math_pipe_throttle_avg') is not None) else None,
            "verdict":               k.get("verdict"),
            "headroom_estimate_pct": k.get("headroom_estimate_pct"),
            "notes":                 k.get("notes", []),
        })
    return ({"kernel_micro": {"available": True, "kernels": kernels}},
            [{"field": "kernel_micro", "source_skill": "ncu-microarch",
              "source_file": str(path), "ok": True}])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-summary")
    ap.add_argument("--timeline-summary")
    ap.add_argument("--torch-profile-text")
    ap.add_argument("--sglang-profile")
    ap.add_argument("--ncu-summary")
    ap.add_argument("--subject-yaml", required=True,
                    help="YAML with subject info (framework, model, config_summary, hardware)")
    ap.add_argument("--workload-yaml", required=True,
                    help="YAML with at least `regimes:` (we hash the file)")
    ap.add_argument("--regime-id", required=True,
                    help="Which regime in bench_summary/workload to pull")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not any([args.bench_summary, args.timeline_summary,
                args.torch_profile_text, args.sglang_profile, args.ncu_summary]):
        Path(args.out).write_text(json.dumps({
            "schema_version": SCHEMA_VERSION, "ok": False,
            "error": "no profile inputs given"}, indent=2))
        return

    subject = _load_yaml(Path(args.subject_yaml))
    workload_path = Path(args.workload_yaml)
    workload = {
        "regime_id": args.regime_id,
        "regime_yaml_path": str(workload_path),
        "regime_yaml_sha256": sha256_file(workload_path),
    }

    out = {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "subject": subject,
        "workload": workload,
        "e2e": None,
        "gpu_macro": None,
        "kernel_breakdown": [],
        "kernel_micro": {"available": False,
                         "reason": "no ncu input provided — pass --ncu-summary to fill"},
        "evidence_chain": [],
        "warnings": [],
    }

    # Apply adapters
    if args.bench_summary:
        partial, chain = _from_bench_summary(Path(args.bench_summary), args.regime_id)
        out.update({k: v for k, v in partial.items() if v is not None})
        out["evidence_chain"].extend(chain)

    if args.timeline_summary:
        partial, chain = _from_timeline_summary(Path(args.timeline_summary))
        out.update({k: v for k, v in partial.items() if v is not None})
        out["evidence_chain"].extend(chain)

    if args.torch_profile_text:
        partial, chain = _from_torch_profile_text(Path(args.torch_profile_text))
        # If we already had a kernel_breakdown from nsys-timeline-sql, prefer torch.profiler
        # for categorized aggregates (it's more semantic) but flag the override.
        if partial.get("kernel_breakdown") and out.get("kernel_breakdown"):
            out["warnings"].append("kernel_breakdown overridden by torch.profiler "
                                   "(nsys-timeline-sql output also present; kept latter in evidence_chain)")
        out.update({k: v for k, v in partial.items() if v is not None})
        out["evidence_chain"].extend(chain)

    if args.sglang_profile:
        partial, chain = _from_sglang_profile(Path(args.sglang_profile))
        out.update({k: v for k, v in partial.items() if v is not None})
        out["evidence_chain"].extend(chain)

    if args.ncu_summary:
        partial, chain = _from_ncu(Path(args.ncu_summary))
        out.update({k: v for k, v in partial.items() if v is not None})
        out["evidence_chain"].extend(chain)
    else:
        # Add an explicit gap_reason placeholder when NCU was not provided.
        out["evidence_chain"].append({
            "field": "kernel_micro", "source_skill": None,
            "source_file": None, "ok": False,
            "gap_reason": "no --ncu-summary input given; run ncu-microarch skill first",
        })

    # Consistency check: wall time agreement between e2e and gpu_macro
    e2e_wall = None
    if out.get("e2e"):
        # No direct wall_ms in e2e; we can derive nothing here without per_run access.
        pass
    if out.get("gpu_macro") and e2e_wall:
        delta_pct = abs(out["gpu_macro"]["wall_ms"] - e2e_wall * 1000) / (e2e_wall * 1000) * 100
        if delta_pct > 20:
            out["warnings"].append(
                f"wall time inconsistency: e2e={e2e_wall*1000:.0f}ms vs nsys={out['gpu_macro']['wall_ms']:.0f}ms")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[profile-summary-unified] wrote {args.out}")


if __name__ == "__main__":
    main()
