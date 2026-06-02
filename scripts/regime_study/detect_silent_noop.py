#!/usr/bin/env python3
"""Detect silent-no-op cases where a sglang server arg flag was ignored.

Usage:
    python scripts/regime_study/detect_silent_noop.py \
        --server-info results/regime_bench/raw/moe_opt_levels/C4_moe_cutlass/server_info.json \
        --trace experiments/tmp/moe_opt_levels/C4_moe_cutlass/raw_trace/*/p_*.trace.json.gz \
        --reference-trace experiments/tmp/moe_opt_levels/C0_baseline/raw_trace/*/p_*.trace.json.gz

For each *_backend flag in server_info, predict which kernels SHOULD appear if
the flag was honoured, then check the trace against that prediction.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# What kernels each backend choice should produce in the trace.
# Keys = (server_arg_field, requested_value), value = list of regex patterns
# that should match at least one GPU kernel in the trace if the flag was honoured.
BACKEND_FINGERPRINT = {
    ("moe_runner_backend", "cutlass"): [
        r"cutlass.*moe", r"cutlass_fused_experts", r"sm90_xmma_warpspecialized.*moe"
    ],
    ("moe_runner_backend", "deep_gemm"): [r"deep_gemm", r"sm90_xmma_grouped"],
    ("moe_runner_backend", "flashinfer_trtllm"): [r"trtllm.*moe", r"flashinfer.*trtllm"],
    ("moe_runner_backend", "flashinfer_cutlass"): [r"flashinfer.*cutlass", r"cutlass.*flashinfer"],
    ("moe_runner_backend", "triton"): [r"fused_moe_kernel"],
    ("moe_runner_backend", "triton_kernel"): [r"fused_moe_kernel"],

    ("attention_backend", "flashinfer"): [
        r"flashinfer.*(?:Prefill|Decode|Attention)",
        r"BatchPrefillWithRaggedKV", r"BatchDecodeWithPagedKV",
    ],
    ("attention_backend", "fa3"): [r"FlashAttnFwdSm90", r"flash::"],
    ("attention_backend", "trtllm_mha"): [r"trtllm.*mha", r"trtllm.*attention"],
    ("attention_backend", "triton"): [r"_attention.*triton", r"triton.*attention"],

    ("sampling_backend", "flashinfer"): [r"flashinfer.*sampling", r"top_p", r"top_k_sampling"],
    ("sampling_backend", "pytorch"): [],  # generic torch ops; hard to fingerprint
}


def load_trace_kernels(path: str) -> dict[str, int]:
    """Return {kernel_name: total_self_us} for all GPU events."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        trace = json.load(f)
    events = trace.get("traceEvents", [])
    by_name: dict[str, int] = defaultdict(int)
    for e in events:
        if e.get("cat") not in {"kernel", "gpu_op", "cuda_runtime"}:
            continue
        if "dur" not in e:
            continue
        by_name[e.get("name", "")] += e["dur"]
    return dict(by_name)


def check_flag(flag: str, requested: str, kernels: dict[str, int]) -> dict:
    """Check if any kernel matches the expected fingerprint for (flag, requested)."""
    key = (flag, requested)
    if key not in BACKEND_FINGERPRINT:
        return {"flag": flag, "requested": requested,
                "verdict": "unknown_fingerprint",
                "matched_patterns": [], "matched_kernels": []}
    patterns = BACKEND_FINGERPRINT[key]
    if not patterns:
        return {"flag": flag, "requested": requested,
                "verdict": "no_fingerprint_defined",
                "matched_patterns": [], "matched_kernels": []}
    matched = []
    matched_patterns = []
    for p in patterns:
        rx = re.compile(p, re.IGNORECASE)
        for name, dur in kernels.items():
            if rx.search(name):
                matched.append({"kernel": name, "self_us": dur})
                matched_patterns.append(p)
    return {
        "flag": flag,
        "requested": requested,
        "verdict": "honoured" if matched else "IGNORED_OR_FALLBACK",
        "matched_patterns": list(set(matched_patterns)),
        "matched_kernels": sorted(matched, key=lambda r: -r["self_us"])[:3],
    }


def diff_traces(a: dict[str, int], b: dict[str, int], top_n: int = 5) -> dict:
    """Compare two kernel dicts; if top-N kernels are nearly identical, the
    flag in trace A is most likely a no-op vs reference B."""
    all_kernels = set(a) | set(b)
    rows = []
    for k in all_kernels:
        ua = a.get(k, 0)
        ub = b.get(k, 0)
        if ua + ub == 0: continue
        rows.append({"kernel": k, "self_us_a": ua, "self_us_b": ub,
                     "delta_pct": (ua - ub) / max(ub, 1) * 100 if ub else None})
    rows.sort(key=lambda r: -(r["self_us_a"] + r["self_us_b"]))
    top = rows[:top_n]
    # similarity score: how many of top-5 are within 5% by self_us
    near_same = sum(1 for r in top if r["delta_pct"] is not None and abs(r["delta_pct"]) < 5)
    return {
        "top_kernel_similarity": f"{near_same}/{len(top)} kernels within ±5% of reference",
        "top_kernel_diff": top,
        "verdict": "LIKELY_SILENT_NOOP" if near_same >= max(3, len(top) - 1) else "different_kernel_mix",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-info", required=True, help="path to server_info.json")
    ap.add_argument("--trace", required=True, help="path or glob to .trace.json.gz")
    ap.add_argument("--reference-trace", default=None,
                    help="optional baseline trace for kernel-mix similarity check")
    ap.add_argument("--out", default=None, help="write JSON report here")
    args = ap.parse_args()

    si = json.loads(Path(args.server_info).read_text())
    # tolerate glob
    trace_path = sorted(glob.glob(args.trace))[0] if "*" in args.trace else args.trace
    print(f"[detect] loading trace {trace_path}")
    kernels = load_trace_kernels(trace_path)
    print(f"[detect] {len(kernels)} unique GPU kernels in trace")

    report = {"server_info_path": args.server_info, "trace_path": trace_path,
              "checks": []}

    # Check each known backend flag
    for flag in ["moe_runner_backend", "attention_backend", "sampling_backend"]:
        requested = si.get(flag)
        if requested is None or requested == "auto":
            print(f"[detect] {flag}={requested!r} → skip (auto path)")
            continue
        result = check_flag(flag, requested, kernels)
        report["checks"].append(result)
        verdict_pretty = {
            "honoured": "✅ HONOURED",
            "IGNORED_OR_FALLBACK": "⚠️  IGNORED OR FELL BACK",
            "unknown_fingerprint": "❓ NO FINGERPRINT FOR THIS VALUE",
            "no_fingerprint_defined": "❓ FINGERPRINT NOT DEFINED",
        }.get(result["verdict"], result["verdict"])
        print(f"[detect] {flag}={requested!r} → {verdict_pretty}")
        for p in result["matched_patterns"][:3]:
            print(f"          matched pattern: /{p}/")
        for m in result["matched_kernels"][:3]:
            print(f"          {m['self_us']:>8} us  {m['kernel'][:80]}")

    # Optional: kernel-mix similarity vs reference
    if args.reference_trace:
        ref_path = sorted(glob.glob(args.reference_trace))[0] if "*" in args.reference_trace else args.reference_trace
        print(f"\n[detect] reference trace {ref_path}")
        ref_kernels = load_trace_kernels(ref_path)
        diff = diff_traces(kernels, ref_kernels)
        report["kernel_mix_diff"] = diff
        print(f"[detect] {diff['verdict']}")
        print(f"[detect] {diff['top_kernel_similarity']}")
        for r in diff["top_kernel_diff"][:5]:
            d = f"{r['delta_pct']:+.1f}%" if r['delta_pct'] is not None else "N/A"
            print(f"          {r['self_us_a']:>8} vs {r['self_us_b']:>8} us  ({d:>8})  {r['kernel'][:60]}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\n[detect] wrote report → {args.out}")

    # Exit non-zero if any flag was IGNORED
    any_ignored = any(c["verdict"] == "IGNORED_OR_FALLBACK" for c in report["checks"])
    return 2 if any_ignored else 0


if __name__ == "__main__":
    sys.exit(main())
