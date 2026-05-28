#!/usr/bin/env python3
"""Materialize regime_scout/seed_suite.yaml → one workload YAML per seed.

Each output file is a self-contained workload that scripts/run_experiment.py
can consume directly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils import load_yaml, save_yaml


def materialize(seed: dict, defaults: dict, idx: int) -> tuple[str, dict]:
    name = seed["name"]
    filename = f"seed_{idx:02d}_{name}.yaml"

    cache = dict(seed.get("cache", {}))
    cache.setdefault("mode", defaults.get("cache_mode", "cold"))
    cache.setdefault("flush_cache", defaults.get("flush_cache", True))

    dataset = dict(seed["dataset"])
    if dataset.get("name") in ("random", "random-ids"):
        dataset.setdefault("random_range_ratio", defaults.get("random_range_ratio", 0.0))

    out = {
        "name": name,
        "regime_hint": seed.get("regime_hint", "unknown"),
        "source": {"generated_by": "generate_seed_suite", "seed_index": idx},
        "dataset": dataset,
        "traffic": dict(seed["traffic"]),
        "cache": cache,
        "seed": seed.get("seed", 1234),
        "notes": seed.get("notes", ""),
    }
    return filename, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="regime_scout/seed_suite.yaml")
    ap.add_argument("--out-dir", default="regime_scout/candidates")
    ap.add_argument("--prune", action="store_true",
                    help="Delete existing seed_*.yaml in --out-dir first.")
    args = ap.parse_args()

    suite = load_yaml(args.seed)
    defaults = suite.get("defaults", {})
    seeds = suite.get("seeds", [])
    if not seeds:
        print("[generate_seed_suite] no seeds found", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.prune:
        for p in out_dir.glob("seed_*.yaml"):
            p.unlink()

    written = []
    for idx, seed in enumerate(seeds):
        fname, doc = materialize(seed, defaults, idx)
        path = out_dir / fname
        save_yaml(doc, path)
        written.append(str(path))
        print(f"[generate_seed_suite] wrote {path}")

    print(f"[generate_seed_suite] total {len(written)} workload(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
