#!/usr/bin/env python3
"""Aggregate hardware_view.json files across (model, regime) into one CSV + Markdown table."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def fmt(v, prec=1, suffix=""):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{prec}f}{suffix}"
    return f"{v}{suffix}"


def top_kernels_summary(top_kernels: list[dict], n: int = 5) -> str:
    if not top_kernels:
        return "n/a"
    parts = []
    for k in top_kernels[:n]:
        cat = k.get("category", "?")
        pct = k.get("self_time_pct")
        parts.append(f"{cat} {pct:.1f}%" if pct is not None else f"{cat} n/a")
    return "; ".join(parts)


def kernel_categories_str(kcat: dict) -> str:
    if not kcat:
        return "n/a"
    return "; ".join(f"{k}: {v}%" for k, v in list(kcat.items())[:6])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw-view-dir", default="experiments/tmp/hw_view")
    ap.add_argument("--out-dir", default="results/regime_bench")
    args = ap.parse_args()

    hv_root = PROJECT_ROOT / args.hw_view_dir
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in sorted(hv_root.iterdir()):
        if not d.is_dir() or "_smoke" in d.name:
            continue
        hv = d / "hardware_view.json"
        if not hv.exists():
            continue
        try:
            data = json.loads(hv.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"warn: cannot parse {hv}: {e}")
            continue
        # parse model + regime from dir name like "dense_R1"
        try:
            model, regime = d.name.split("_", 1)
        except ValueError:
            continue
        gpu = data.get("gpu_sampling") or {}
        si = data.get("server_info_excerpt") or {}
        prof_t = data.get("profile_totals") or {}
        prof_phase = data.get("profile_phase_breakdown_pct") or {}
        kcat = data.get("kernel_category_pct") or {}
        top_kernels = data.get("top_kernels") or []
        moe_o = data.get("profile_moe_overhead") or {}

        row = {
            "model": model,
            "regime": regime,
            "wall_window_s": data.get("wall_window_s"),
            # backend selection (from /get_server_info, runtime-confirmed)
            "attention_backend": si.get("attention_backend"),
            "sampling_backend": si.get("sampling_backend"),
            "schedule_policy": si.get("schedule_policy"),
            "kv_cache_dtype": si.get("kv_cache_dtype"),
            "max_running_requests": si.get("max_running_requests"),
            "torch_compile_max_bs": si.get("torch_compile_max_bs"),
            # hardware (from nvidia-smi)
            "samples": gpu.get("sample_count"),
            "gpu_mem_peak_GiB": (
                round(gpu.get("memory_used_mib_peak") / 1024, 2)
                if gpu.get("memory_used_mib_peak") is not None else None
            ),
            "gpu_mem_mean_GiB": (
                round(gpu.get("memory_used_mib_mean") / 1024, 2)
                if gpu.get("memory_used_mib_mean") is not None else None
            ),
            "gpu_util_mean_pct": gpu.get("utilization_gpu_pct_mean"),
            "gpu_util_p95_pct": gpu.get("utilization_gpu_pct_p95"),
            "mem_util_mean_pct": gpu.get("utilization_mem_pct_mean"),
            "power_mean_W": gpu.get("power_w_mean"),
            "power_peak_W": gpu.get("power_w_peak"),
            "temp_peak_C": gpu.get("temperature_c_peak"),
            "sm_clock_mhz_mean": gpu.get("sm_clock_mhz_mean"),
            # profile totals
            "trace_wallclock_ms": prof_t.get("wallclock_ms"),
            "trace_gpu_active_ms": prof_t.get("gpu_active_ms"),
            # kernel categories (top-K aggregated)
            "kernel_categories": kernel_categories_str(kcat),
            # top 5 kernels by self time
            "top_kernels_short": top_kernels_summary(top_kernels, n=5),
            "top_kernel_1_name": (top_kernels[0].get("name_short") if top_kernels else None),
            "top_kernel_1_pct": (top_kernels[0].get("self_time_pct") if top_kernels else None),
            "top_kernel_1_calls": (top_kernels[0].get("calls") if top_kernels else None),
            "top_kernel_2_name": (top_kernels[1].get("name_short") if len(top_kernels) > 1 else None),
            "top_kernel_2_pct": (top_kernels[1].get("self_time_pct") if len(top_kernels) > 1 else None),
            # MoE detection
            "moe_applicable": moe_o.get("applicable"),
            "moe_routing_share_pct": moe_o.get("share_pct"),
        }
        rows.append(row)

    if not rows:
        print("no hardware_view.json found")
        return 1

    # CSV
    csv_path = out_dir / "hardware_view_table.csv"
    keys = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})

    # Markdown
    md_path = out_dir / "hardware_view_table.md"
    lines = []
    lines.append("# Regime hardware view — backend / GPU / kernel selection")
    lines.append("")
    lines.append("> Generated by `scripts/regime_study/aggregate_hw_view.py`.")
    lines.append("> Data source: `/get_server_info` (backend), `nvidia-smi @ 0.5s` "
                 "(GPU stats), `torch.profiler` via `sglang.bench_serving --profile` "
                 "(kernel breakdown).")
    lines.append("")

    # Table 1 — Backend selection (constant per model, but confirmed at runtime)
    lines.append("## 1. Backend selection (runtime-confirmed via `/get_server_info`)")
    lines.append("")
    lines.append("| Model | Regime | Attention | Sampling | Schedule | KV dtype | max_running | torch_compile_max_bs |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['regime']} | "
            f"**{r['attention_backend']}** | {r['sampling_backend']} | "
            f"{r['schedule_policy']} | {r['kv_cache_dtype']} | "
            f"{r['max_running_requests']} | {r['torch_compile_max_bs']} |"
        )
    lines.append("")

    # Table 2 — Hardware
    lines.append("## 2. Hardware view (`nvidia-smi` during bench window)")
    lines.append("")
    lines.append("| Model | Regime | Wall (s) | Samples | Mem peak (GiB) | Mem mean (GiB) | GPU util mean (%) | GPU util p95 (%) | Mem-ctrl util (%) | Power mean (W) | Power peak (W) | Peak temp (°C) | SM clock mean (MHz) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['regime']} | {fmt(r['wall_window_s'],1)} | {r['samples']} | "
            f"{fmt(r['gpu_mem_peak_GiB'],2)} | {fmt(r['gpu_mem_mean_GiB'],2)} | "
            f"{fmt(r['gpu_util_mean_pct'],1)} | {fmt(r['gpu_util_p95_pct'],0)} | "
            f"{fmt(r['mem_util_mean_pct'],1)} | "
            f"{fmt(r['power_mean_W'],0)} | {fmt(r['power_peak_W'],0)} | "
            f"{fmt(r['temp_peak_C'],0)} | {fmt(r['sm_clock_mhz_mean'],0)} |"
        )
    lines.append("")

    # Table 3 — Kernel selection
    lines.append("## 3. Kernel breakdown (`torch.profiler` trace, top-20 GPU events)")
    lines.append("")
    lines.append("| Model | Regime | Trace wall (ms) | GPU active (ms) | Kernel categories (% of top-20 self time) |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['regime']} | "
            f"{fmt(r['trace_wallclock_ms'],1)} | {fmt(r['trace_gpu_active_ms'],1)} | "
            f"{r['kernel_categories']} |"
        )
    lines.append("")
    # Top-2 kernel names per row for evidence
    lines.append("**Top-2 kernels per cell (`self_time_pct` from the trace)**:")
    lines.append("")
    lines.append("| Model | Regime | #1 kernel | #1 % | calls | #2 kernel | #2 % |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['regime']} | "
            f"`{(r['top_kernel_1_name'] or '')[:60]}` | {fmt(r['top_kernel_1_pct'],1)}% | "
            f"{r['top_kernel_1_calls']} | "
            f"`{(r['top_kernel_2_name'] or '')[:60]}` | {fmt(r['top_kernel_2_pct'],1)}% |"
        )
    lines.append("")

    md_path.write_text("\n".join(lines))
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
