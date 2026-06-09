#!/usr/bin/env python3
"""Skill impl: regime-sweep-runner.

Iterates configs × regimes by calling e2e-bench-runner once per config (which
internally sweeps regimes). Aggregates results into a single matrix JSON.

CLI:
    python sweep.py --configs-file C.yaml --regimes-file R.yaml \
        --num-runs 3 --out-dir DIR
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 0
SKILL_DIR = Path(__file__).resolve().parent.parent
E2E_BENCH = (SKILL_DIR.parent / "e2e-bench-runner" / "impl" / "run_bench.py").resolve()


def load_yaml(path: Path):
    try:
        import yaml
    except ImportError:
        raise RuntimeError("PyYAML required")
    return yaml.safe_load(path.read_text())


def run_one_cell(out_dir: Path, cfg: dict, regimes_file: Path,
                 num_runs: int) -> dict:
    """Invoke e2e-bench-runner for one config, return the loaded summary."""
    cell_dir = out_dir / "per_config" / cfg["tag"]
    cell_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable, str(E2E_BENCH),
        "--url",          cfg["url"],
        "--backend",      cfg["backend"],
        "--tag",          cfg["tag"],
        "--num-runs",     str(num_runs),
        "--out-dir",      str(cell_dir),
        "--regimes-file", str(regimes_file),
    ]
    if "model_name" in cfg:
        argv.extend(["--model-name", cfg["model_name"]])
    print(f"[regime-sweep] running cell '{cfg['tag']}' ...", flush=True)
    proc = subprocess.run(argv, capture_output=True, text=True)
    summary_path = cell_dir / "bench_summary.json"
    if not summary_path.exists():
        return {"ok": False, "error": "no bench_summary.json produced",
                "stderr_tail": (proc.stderr or "")[-512:]}
    return json.loads(summary_path.read_text())


def compress_cell(summary: dict, cfg_notes: str) -> dict:
    """Compress one e2e-bench-runner summary into matrix-cell shape."""
    if not summary.get("ok", False):
        return {"ok": False, "error": summary.get("error", "unknown"),
                "notes": cfg_notes}
    out = {"ok": True, "notes": cfg_notes, "regimes": {}}
    for r_id, r in summary.get("regimes", {}).items():
        rs  = r.get("req_per_s",    {}) or {}
        ts  = r.get("tokens_per_s", {}) or {}
        e2e = r.get("e2e_ms",       {}) or {}
        out["regimes"][r_id] = {
            "req_per_s_mean":    rs.get("mean"),
            "stddev_pct":        rs.get("stddev_pct"),
            "reliable":          r.get("reliable"),
            "tokens_per_s_mean": ts.get("mean"),
            "e2e_p50_ms":        e2e.get("p50"),
            "e2e_p99_ms":        e2e.get("p99"),
            "completion_rate":   r.get("completion_rate"),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs-file", required=True)
    ap.add_argument("--regimes-file", required=True)
    ap.add_argument("--num-runs", type=int, default=3)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "regime_sweep_summary.json"

    try:
        configs_doc = load_yaml(Path(args.configs_file))
        regimes_doc = load_yaml(Path(args.regimes_file))
    except Exception as e:
        json.dump({"schema_version": SCHEMA_VERSION, "ok": False,
                   "error": f"YAML load failed: {e}"},
                  summary_path.open("w"), indent=2)
        sys.exit(1)

    configs = configs_doc.get("configs") or []
    regimes = list((regimes_doc.get("regimes") or {}).keys())
    if not configs or not regimes:
        json.dump({"schema_version": SCHEMA_VERSION, "ok": False,
                   "error": "configs or regimes empty"},
                  summary_path.open("w"), indent=2)
        sys.exit(1)

    # Validate config entries
    REQ = {"tag", "url", "backend"}
    for i, c in enumerate(configs):
        missing = REQ - set(c)
        if missing:
            json.dump({"schema_version": SCHEMA_VERSION, "ok": False,
                       "error": f"config #{i} missing fields: {missing}"},
                      summary_path.open("w"), indent=2)
            sys.exit(1)

    matrix = {}
    warnings = []
    for cfg in configs:
        cell_summary = run_one_cell(out_dir, cfg,
                                    Path(args.regimes_file), args.num_runs)
        compressed = compress_cell(cell_summary, cfg.get("notes", ""))
        matrix[cfg["tag"]] = compressed
        if not compressed["ok"]:
            warnings.append(f"{cfg['tag']} cell failed: {compressed.get('error')}")
        else:
            for r_id, r in compressed["regimes"].items():
                if r.get("reliable") is False:
                    warnings.append(f"{cfg['tag']}/{r_id}: stddev_pct={r.get('stddev_pct'):.1f}% — reliable=false")

    all_failed = all(not v.get("ok", False) for v in matrix.values())
    out = {
        "schema_version": SCHEMA_VERSION,
        "ok": not all_failed,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "configs_file": args.configs_file,
        "regimes_file": args.regimes_file,
        "regimes": regimes,
        "configs": [c["tag"] for c in configs],
        "matrix": matrix,
        "warnings": warnings,
    }
    if all_failed:
        out["error"] = "all sweep cells failed"

    summary_path.write_text(json.dumps(out, indent=2))
    print(f"[regime-sweep-runner] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
