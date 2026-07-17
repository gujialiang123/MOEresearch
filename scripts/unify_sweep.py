#!/usr/bin/env python3
"""Per-regime unifier: run profile-summary-unified for each of the 4 regimes.
Idempotent — re-run anytime to refresh unified.json after ncu finishes."""
import json
import subprocess
from pathlib import Path

REPO = Path("/home/t-jialianggu/work/MOEresearch")
BASE = REPO / "results/2026-06-09_sglang_triton_sweep"
REGIMES = ["R_short_decode", "R_medium_balanced", "R_long_prefill", "R_concurrent_decode"]

SUBJECT_YAML = BASE / "subject.yaml"
SUBJECT_YAML.write_text("""framework: sglang
framework_version: "0.5.9"
model: qwen3-30b-a3b-moe
config_summary:
  moe_backend: triton
  cudagraph: false
  autotune: false
  tp_size: 1
hardware:
  name: NVIDIA H200
  sm: "9.0"
  num_sms: 132
""")

for rid in REGIMES:
    out = BASE / "unified" / rid
    out.mkdir(parents=True, exist_ok=True)

    bench = BASE / "bench" / "bench_summary.json"
    nsys = BASE / "nsys" / rid / "timeline_summary.json"
    ncu = BASE / "ncu" / rid / "ncu_summary.json"

    argv = [
        "python3", str(REPO / ".github/skills/profile-summary-unified/impl/unify.py"),
        "--subject-yaml", str(SUBJECT_YAML),
        "--workload-yaml", str(REPO / "regimes/qwen3_30b_moe_sglang_perf_sweep.yaml"),
        "--regime-id", rid,
        "--out", str(out / "profile_unified.json"),
    ]
    if bench.exists():
        argv.extend(["--bench-summary", str(bench)])
    if nsys.exists():
        argv.extend(["--timeline-summary", str(nsys)])
    if ncu.exists():
        argv.extend(["--ncu-summary", str(ncu)])

    print(f">> unifying {rid} ...", flush=True)
    subprocess.run(argv, check=True)

print("DONE", flush=True)
