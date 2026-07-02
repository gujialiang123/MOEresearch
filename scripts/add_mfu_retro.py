#!/usr/bin/env python3
"""Retro-annotate all existing bench summary.json files with MFU/MBU fields.

Walks a results/ subtree, finds every summary.json where ok=true and regimes
is populated, and adds `mfu` dict to each regime (in-place). Skips summaries
that already have mfu.

Usage:
    python scripts/add_mfu_retro.py \\
        --results-root results/2026-06-30_lfm2.5 \\
        --hardware configs/hardware/h200.yaml \\
        --model configs/models/lfm2.5-8b-a1b.yaml \\
        [--force]   # overwrite existing mfu fields
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness.mfu import HardwareConfig, ModelConfig, annotate_summary_with_mfu


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing mfu fields (default: skip if present)")
    args = ap.parse_args()

    hw = HardwareConfig.load(args.hardware)
    mdl = ModelConfig.load(args.model)

    root = Path(args.results_root)
    total = 0
    updated = 0
    skipped_has_mfu = 0
    skipped_no_ok = 0
    errors = 0

    for summary_path in root.rglob("summary.json"):
        total += 1
        try:
            data = json.loads(summary_path.read_text())
        except Exception as e:
            print(f"[skip] {summary_path}: cannot parse ({e})")
            errors += 1
            continue

        if not data.get("ok"):
            skipped_no_ok += 1
            continue

        regimes = data.get("regimes", {})
        if not regimes:
            skipped_no_ok += 1
            continue

        already = all("mfu" in r for r in regimes.values() if isinstance(r, dict))
        if already and not args.force:
            skipped_has_mfu += 1
            continue

        annotate_summary_with_mfu(data, model=mdl, hardware=hw)
        summary_path.write_text(json.dumps(data, indent=2))
        updated += 1
        try:
            display = summary_path.resolve().relative_to(_REPO_ROOT)
        except ValueError:
            display = summary_path
        print(f"[ok] {display}")

    print()
    print(f"total summaries       : {total}")
    print(f"updated with MFU      : {updated}")
    print(f"skipped (already mfu) : {skipped_has_mfu}")
    print(f"skipped (not ok/empty): {skipped_no_ok}")
    print(f"errors                : {errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
