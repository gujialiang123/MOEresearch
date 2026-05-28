#!/usr/bin/env python3
"""Skill impl: boundary-expansion — generate neighbor workloads along one axis."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from utils import load_yaml, save_yaml  # noqa: E402


AXIS_TO_FIELD: dict[str, tuple[str, str]] = {
    "max_concurrency": ("traffic", "max_concurrency"),
    "num_prompts":     ("traffic", "num_prompts"),
    "input_len":       ("dataset", "random_input_len"),
    "output_len":      ("dataset", "random_output_len"),
}


def axis_values(search_space: dict, axis: str) -> list:
    axes = search_space.get("axes", {})
    if axis in axes and "values" in axes[axis]:
        return list(axes[axis]["values"])
    raise ValueError(f"axis '{axis}' not in search_space.axes")


def current_value(parent: dict, axis: str):
    section, field = AXIS_TO_FIELD[axis]
    return (parent.get(section) or {}).get(field)


def neighbor_values(values: list, current, strategy: str, count: int = 4) -> list:
    if current is None or current not in values:
        # fallback: use whole list filtered by strategy
        if strategy == "bracket":
            return values[:count]
        if strategy == "downward":
            return values[:count]
        if strategy == "upward":
            return values[-count:]
        return values[:count]
    idx = values.index(current)
    if strategy == "bracket":
        below = values[max(0, idx - count // 2):idx]
        above = values[idx + 1:idx + 1 + count // 2 + count % 2]
        return below + above
    if strategy == "downward":
        return values[max(0, idx - count):idx]
    if strategy == "upward":
        return values[idx + 1:idx + 1 + count]
    if strategy == "geometric":
        # log-uniform-ish: every (len/count)-th sample
        if count >= len(values):
            return [v for v in values if v != current]
        step = max(1, len(values) // count)
        return [values[i] for i in range(0, len(values), step) if values[i] != current][:count]
    raise ValueError(f"unknown strategy: {strategy}")


def make_neighbor(parent: dict, parent_path: Path, axis: str, axis_val,
                  strategy: str) -> tuple[str, dict]:
    section, field = AXIS_TO_FIELD[axis]
    child = json.loads(json.dumps(parent))  # deep copy via JSON
    child.setdefault(section, {})[field] = axis_val
    parent_name = parent.get("name", parent_path.stem)
    short_axis = axis.replace("max_", "")[:3]
    name = f"{parent_name}__{short_axis}_{axis_val}"
    child["name"] = name
    child["regime_hint"] = parent.get("regime_hint")
    child["source"] = {
        "generated_by": "boundary-expansion",
        "parent": str(parent_path),
        "axis": axis,
        "axis_value": axis_val,
        "strategy": strategy,
    }
    # Cap num_prompts proportionally to keep wall time sane when max_concurrency rises
    if axis == "max_concurrency":
        cur_mc = (parent.get("traffic") or {}).get("max_concurrency") or 1
        cur_np = (parent.get("traffic") or {}).get("num_prompts") or 32
        scale = axis_val / max(cur_mc, 1)
        # Keep total work in [0.5x, 2x] of parent
        scale = max(0.5, min(2.0, scale))
        child.setdefault("traffic", {})["num_prompts"] = max(8, int(cur_np * scale))
    return name, child


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True, help="parent workload YAML")
    ap.add_argument("--axis", required=True, choices=list(AXIS_TO_FIELD.keys()))
    ap.add_argument("--strategy", default="bracket",
                    choices=["bracket", "downward", "upward", "geometric"])
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--search-space", default="regime_scout/search_space.yaml")
    ap.add_argument("--neighbors-out", required=True,
                    help="dir to write generated YAMLs into")
    ap.add_argument("--summary-json", default=None)
    args = ap.parse_args()

    parent_path = Path(args.parent).resolve()
    parent = load_yaml(parent_path)
    space = load_yaml(args.search_space)

    try:
        values = axis_values(space, args.axis)
    except ValueError as e:
        out = {"ok": False, "error": str(e)}
        if args.summary_json:
            Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.summary_json).write_text(json.dumps(out))
        print(json.dumps(out))
        return 1

    cur = current_value(parent, args.axis)
    neighbors = neighbor_values(values, cur, args.strategy, count=args.count)
    if not neighbors:
        out = {"ok": False, "error": "no neighbors", "parent_value": cur, "values": values}
        if args.summary_json:
            Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.summary_json).write_text(json.dumps(out))
        print(json.dumps(out, indent=2))
        return 1

    out_dir = Path(args.neighbors_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for v in neighbors:
        name, child = make_neighbor(parent, parent_path, args.axis, v, args.strategy)
        path = out_dir / f"{name}.yaml"
        if path.exists():
            print(f"[expand] skip existing: {path}", file=sys.stderr)
            continue
        save_yaml(child, path)
        generated.append({"path": str(path), "axis_value": v, "name": name})

    summary = {
        "ok": True,
        "parent": str(parent_path),
        "axis": args.axis,
        "strategy": args.strategy,
        "parent_value": cur,
        "neighbor_count": len(generated),
        "generated": generated,
    }
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
