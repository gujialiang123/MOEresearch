#!/usr/bin/env python3
"""Skill impl: ncu-microarch.

Wraps `sudo -n ncu` against a target_cmd, captures specific metrics for
a kernel regex, and reduces the CSV to ncu_summary.json. See ../SKILL.md.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 0
SKILL_DIR = Path(__file__).resolve().parent.parent
DEFAULT_METRICS_FILE = SKILL_DIR / "metric_sets" / "default.txt"
NCU_PATH_CANDIDATES = [
    "/home/t-chendili/.conda/pkgs/nsight-compute-2026.1.1.2-h1ff7d1d_0/bin/ncu",
]


def find_ncu() -> str | None:
    for p in NCU_PATH_CANDIDATES:
        if Path(p).exists():
            return p
    p = shutil.which("ncu")
    return p


def load_metrics(path: Path) -> list[str]:
    out = []
    for raw in path.read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def fail(out_dir: Path, msg: str, stderr: str = "") -> None:
    payload = {"schema_version": SCHEMA_VERSION, "ok": False, "error": msg}
    if stderr:
        payload["stderr_tail"] = stderr[-1024:]
    (out_dir / "ncu_summary.json").write_text(json.dumps(payload, indent=2))
    print(f"[ncu-microarch] FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

# Map NCU metric name → short key used in summary.json
METRIC_KEY = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed":            "sm_throughput_pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed":          "dram_throughput_pct",
    "smsp__warps_active.avg.pct_of_peak_sustained_active":         "warps_active_pct",
    "l1tex__t_sector_hit_rate.pct":                                "l1_hit_pct",
    "lts__t_sector_hit_rate.pct":                                  "l2_hit_pct",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": "tensor_pipe_active_pct",
    # ncu auto-expands stall metrics to .max_rate/.pct/.ratio — we want .ratio
    # (warps-per-issue average, comparable across kernels).
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio":    "stall_long_scoreboard_avg",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio": "stall_math_pipe_throttle_avg",
}


def parse_ncu_csv(csv_text: str) -> dict[str, dict[str, list[float]]]:
    """Returns {kernel_name: {metric_short_key: [values across launches]}}.

    NCU writes some preamble lines (==PROF== Connected ..., target stdout, etc.)
    BEFORE the CSV header. We strip everything up to the line starting with `"ID",`.
    """
    lines = csv_text.splitlines()
    start = None
    for i, l in enumerate(lines):
        if l.startswith('"ID","Process ID"'):
            start = i
            break
    if start is None:
        return {}
    csv_only = "\n".join(lines[start:])
    reader = csv.DictReader(io.StringIO(csv_only))
    out: dict[str, dict[str, list[float]]] = {}
    for row in reader:
        kernel = (row.get("Kernel Name") or "").strip()
        metric_raw = (row.get("Metric Name") or "").strip()
        val_raw = (row.get("Metric Value") or "").strip().replace(",", "")
        if not kernel or not metric_raw:
            continue
        try:
            val = float(val_raw)
        except ValueError:
            continue
        short = METRIC_KEY.get(metric_raw)
        if short is None:
            continue   # ignore unmapped variants (e.g. .max_rate of stall metrics)
        out.setdefault(kernel, {}).setdefault(short, []).append(val)
    return out


def avg(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def verdict_for(m: dict) -> tuple[str, list[str]]:
    """Apply rules from SKILL.md to derive a verdict + notes."""
    sm = m.get("sm_throughput_pct") or 0
    dram = m.get("dram_throughput_pct") or 0
    occ = m.get("warps_active_pct") or 0
    tc = m.get("tensor_pipe_active_pct") or 0
    stall_sb = m.get("stall_long_scoreboard_avg") or 0
    stall_mp = m.get("stall_math_pipe_throttle_avg") or 0
    notes = []

    if occ < 30:
        verdict = "low_occupancy"
    elif tc < 10 and sm > 30:
        verdict = "tensor_core_idle"
        notes.append("Tensor Cores firing <10% — likely wrong dtype, layout, or non-TC kernel.")
    elif dram >= 70:
        verdict = "memory_bound"
        notes.append("DRAM ≥70% — algorithmic reuse (tile fusion, persistent kernels) beats tile tuning.")
    elif sm >= 70 and dram < 50:
        verdict = "compute_bound"
        if tc >= 50:
            notes.append(f"Tensor Cores firing at {tc:.1f}% — good for bf16/fp16 on Hopper.")
        else:
            notes.append(f"SM busy ({sm:.1f}%) but Tensor Cores only {tc:.1f}% — may have non-TC fallback.")
    elif sm < 30 and dram < 30:
        verdict = "latency_bound"
        notes.append("Both SM and DRAM under 30% — launch overhead or warp stalls dominate.")
    else:
        verdict = "balanced"

    if stall_sb > 2.0:
        notes.append(f"long_scoreboard stall = {stall_sb:.2f} warps/issue — severe memory-bound signal.")
    if stall_mp > 1.5:
        notes.append(f"math_pipe_throttle stall = {stall_mp:.2f} — tensor cores saturated.")
    return verdict, notes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-cmd", required=True)
    ap.add_argument("--kernel-regex", required=True)
    ap.add_argument("--launch-count", type=int, default=12)
    ap.add_argument("--metrics-file", default=str(DEFAULT_METRICS_FILE))
    ap.add_argument("--gpu-id", type=int, default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ncu = find_ncu()
    if not ncu:
        fail(out_dir, "ncu binary not found in known paths")

    # Preflight: sudo -n ncu --version
    try:
        v = subprocess.run(
            ["sudo", "-n", ncu, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if v.returncode != 0 or "Nsight Compute" not in v.stdout:
            fail(out_dir,
                 "sudo -n ncu failed; NOPASSWD entry missing or wrong path",
                 v.stderr or v.stdout)
        version_line = next((l for l in v.stdout.splitlines() if "Version" in l), "unknown")
    except Exception as e:
        fail(out_dir, f"sudo ncu preflight crashed: {e}")

    metrics = load_metrics(Path(args.metrics_file))
    if not metrics:
        fail(out_dir, f"no metrics in {args.metrics_file}")

    csv_path = out_dir / "ncu_raw.csv"

    env = os.environ.copy()
    if args.gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # Invoke ncu — write to ncu_raw.csv directly.
    # Note: the chendi-issued sudoers entry only allows running ncu (no -E flag).
    # To pass env vars to the target_cmd, the *caller* must inline them like
    # `--target-cmd "CUDA_HOME=/path PATH=/x:/y python my.py"`. sh -c respects that.
    argv = [
        "sudo", "-n", ncu,
        "--target-processes", "all",
        "--launch-count", str(args.launch_count),
        "--kernel-name-base", "demangled",     # match against demangled name (e.g. "cutlass::device_kernel<...>")
        "--kernel-name", f"regex:{args.kernel_regex}",
        "--metrics", ",".join(metrics),
        "--csv",
        "--force-overwrite",
    ]
    # Split target_cmd into argv via shell wrapper to preserve user quoting
    argv.extend(["sh", "-c", args.target_cmd])

    print(f"[ncu-microarch] running ncu (target: {args.target_cmd[:60]} ...)", flush=True)
    proc = subprocess.run(argv, env=env, capture_output=True, text=True,
                          timeout=600)
    csv_path.write_text(proc.stdout)

    if proc.returncode != 0 and not proc.stdout:
        fail(out_dir, f"ncu exited {proc.returncode} with empty stdout",
             proc.stderr or proc.stdout)

    # Parse
    parsed = parse_ncu_csv(proc.stdout)
    if not parsed:
        fail(out_dir,
             f"no kernels parsed from ncu csv — kernel_regex '{args.kernel_regex}' "
             "may not have matched any kernel launches",
             proc.stderr)

    kernels_out = []
    warnings = []
    for kname, mvals in parsed.items():
        samples = max(len(v) for v in mvals.values()) if mvals else 0
        agg = {k: avg(v) for k, v in mvals.items()}
        v_kind, notes = verdict_for(agg)
        headroom = round(100 - max(agg.get("sm_throughput_pct") or 0,
                                   agg.get("dram_throughput_pct") or 0), 1)
        kernels_out.append({
            "kernel": kname,
            "samples": samples,
            "metrics": {k: (round(v, 3) if v is not None else None)
                        for k, v in agg.items()},
            "verdict": v_kind,
            "headroom_estimate_pct": headroom,
            "notes": notes,
        })
        if samples < args.launch_count:
            warnings.append(
                f"kernel '{kname[:50]}...': only {samples} samples (requested {args.launch_count})")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "ncu_path": ncu,
        "ncu_version": version_line,
        "target_cmd": args.target_cmd,
        "kernel_regex": args.kernel_regex,
        "launch_count": args.launch_count,
        "gpu_id": args.gpu_id,
        "metrics_file": args.metrics_file,
        "metrics_collected": metrics,
        "kernels": kernels_out,
        "warnings": warnings,
    }
    (out_dir / "ncu_summary.json").write_text(json.dumps(payload, indent=2))
    print(f"[ncu-microarch] wrote {out_dir/'ncu_summary.json'} — {len(kernels_out)} kernel(s)")


if __name__ == "__main__":
    main()
