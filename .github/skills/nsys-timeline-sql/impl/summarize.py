#!/usr/bin/env python3
"""Skill impl: nsys-timeline-sql.

Three sub-commands:
  (default)   --sqlite ... --out-dir ...    Produce timeline_summary.json
  diff        --baseline ... --patched ...  Diff two summaries
  query       --sqlite ... --sql "..."      Run arbitrary read-only SQL

See ../SKILL.md for output contracts.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

SCHEMA_VERSION = 0
RECIPES_DIR = Path(__file__).parent.parent / "recipes"


def load_recipe(name: str) -> str:
    return (RECIPES_DIR / f"{name}.sql").read_text()


def query_sqlite(conn: sqlite3.Connection, sql: str, params: dict | None = None) -> list[dict]:
    """Execute SQL using sqlite3's native named-parameter binding (:name).
    SQLite ignores :name inside `--` comments, so the dict can contain extras.
    """
    cur = conn.execute(sql, params or {})
    cols = [d[0] for d in cur.description] if cur.description else []
    return [dict(zip(cols, row)) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Default mode: build timeline_summary.json
# ---------------------------------------------------------------------------

def summarize(sqlite_path: Path, out_dir: Path, top_n: int = 10,
              stream_id: int | None = None,
              window_ns: tuple[int, int] | None = None) -> dict:
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    warnings: list[str] = []

    # Sanity check required tables
    have = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    for t in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_RUNTIME", "StringIds"):
        if t not in have:
            return {"schema_version": SCHEMA_VERSION, "ok": False,
                    "error": f"table {t} not found in sqlite"}

    win_start, win_end = (window_ns or (None, None))
    bind = {"stream_id": stream_id, "top_n": top_n,
            "win_start": win_start, "win_end": win_end}

    # GPU info
    gpu_info = {}
    try:
        g = conn.execute("SELECT name, smCount, smMajor, smMinor FROM TARGET_INFO_GPU LIMIT 1").fetchone()
        if g:
            gpu_info = {"device_name": g["name"],
                        "sm_arch": f"{g['smMajor']}.{g['smMinor']}",
                        "num_sms": g["smCount"]}
    except sqlite3.Error:
        pass

    # Per-stream breakdown
    stream_rows = query_sqlite(conn, load_recipe("per_stream_breakdown"), bind)
    if not stream_rows:
        return {"schema_version": SCHEMA_VERSION, "ok": False,
                "error": "no GPU kernels in profile (CUPTI_ACTIVITY_KIND_KERNEL empty)"}
    active_streams = [int(r["streamId"]) for r in stream_rows]
    primary = stream_rows[0]
    primary_stream = int(primary["streamId"])
    if stream_id is None:
        bind["stream_id"] = primary_stream

    per_stream = {
        str(int(r["streamId"])): {
            "active_ms": (r["active_ns"] or 0) / 1e6,
            "idle_ms":   (r["idle_ns"]   or 0) / 1e6,
            "kernel_count": int(r["kernel_count"] or 0),
        } for r in stream_rows
    }

    primary_active_ns = primary["active_ns"] or 0
    primary_idle_ns   = primary["idle_ns"]   or 0
    wall_ns = primary_active_ns + primary_idle_ns
    util = (primary_active_ns / wall_ns * 100.0) if wall_ns else 0.0

    # Top kernels
    kernel_rows = query_sqlite(conn, load_recipe("top_kernels_by_self_time"), bind)
    top_kernels = []
    for i, r in enumerate(kernel_rows, 1):
        self_ns = r["self_ns"] or 0
        top_kernels.append({
            "rank": i,
            "short_name": r["short_name"],
            "self_ms": self_ns / 1e6,
            "self_pct_of_active": (self_ns / primary_active_ns * 100.0)
                                  if primary_active_ns else 0.0,
            "calls": int(r["calls"] or 0),
            "avg_us": (r["avg_ns"] or 0) / 1e3,
            "max_us": (r["max_ns"] or 0) / 1e3,
            "register_per_thread": r["max_reg"],
            "max_grid": [r["max_grid_x"], r["max_grid_y"], r["max_grid_z"]],
            "max_block": [r["max_block_x"], r["max_block_y"], r["max_block_z"]],
        })

    # Largest idle gaps
    gap_rows = query_sqlite(conn, load_recipe("largest_idle_gaps"), bind)
    largest_gaps = []
    for i, r in enumerate(gap_rows, 1):
        gap_ns = r["gap_ns"] or 0
        before = r["before_kernel"] or ""
        likely = None
        if "memcpy" in before.lower() or "Memcpy" in before:
            likely = "host roundtrip (memcpy precedes idle)"
        elif "Sync" in before or "sync" in before:
            likely = "explicit synchronization"
        largest_gaps.append({
            "rank": i,
            "gap_ms": gap_ns / 1e6,
            "start_ns": r["gap_start_ns"],
            "end_ns":   r["gap_end_ns"],
            "before_kernel": before,
            "after_kernel":  r["after_kernel"],
            "likely_cause":  likely,
        })

    # CUDA API counts
    api_rows = query_sqlite(conn, load_recipe("api_launch_counts"), bind)
    launch_kernel_count = 0
    launch_kernel_total = 0
    graph_launch_count = 0
    graph_launch_total = 0
    for r in api_rows:
        name = (r["api_name"] or "").lower()
        c, t = int(r["calls"] or 0), int(r["total_ns"] or 0)
        if "graphlaunch" in name:
            graph_launch_count += c
            graph_launch_total += t
        elif "launchkernel" in name:
            launch_kernel_count += c
            launch_kernel_total += t
    total_launches = launch_kernel_count + graph_launch_count
    graph_ratio = (graph_launch_count / total_launches) if total_launches else 0.0
    if graph_ratio < 0.1:
        verdict = "eager_dominated"
    elif graph_ratio < 0.9:
        verdict = "mixed"
    else:
        verdict = "graph_dominated"

    # Memcpy (table may be absent if no transfers)
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in have:
        memcpy_rows = query_sqlite(conn, load_recipe("memcpy_aggregate"), bind)
    else:
        memcpy_rows = []
        warnings.append("CUPTI_ACTIVITY_KIND_MEMCPY table missing — no memcpy events recorded")
    # copyKind 1 = H2D, 2 = D2H, 3 = D2D
    mcpy = {"h2d_bytes": 0, "h2d_ms": 0.0, "d2h_bytes": 0, "d2h_ms": 0.0,
            "d2d_bytes": 0, "d2d_ms": 0.0}
    for r in memcpy_rows:
        ck = int(r["copyKind"] or 0)
        b, t = int(r["total_bytes"] or 0), int(r["total_ns"] or 0)
        if   ck == 1: mcpy["h2d_bytes"] += b; mcpy["h2d_ms"] += t / 1e6
        elif ck == 2: mcpy["d2h_bytes"] += b; mcpy["d2h_ms"] += t / 1e6
        elif ck == 3: mcpy["d2d_bytes"] += b; mcpy["d2d_ms"] += t / 1e6
    if mcpy["h2d_ms"]:
        mcpy["h2d_avg_gb_s"] = mcpy["h2d_bytes"] / (mcpy["h2d_ms"] / 1e3) / 1e9
    if mcpy["d2h_ms"]:
        mcpy["d2h_avg_gb_s"] = mcpy["d2h_bytes"] / (mcpy["d2h_ms"] / 1e3) / 1e9

    # Sanity warnings
    if len([s for s in per_stream.values() if s["active_ms"] > 0.05 * primary["active_ns"] / 1e6]) > 5:
        warnings.append(">5 streams have non-trivial activity; per-stream breakdown is worth inspecting")
    if util < 60:
        warnings.append(f"gpu_util_pct={util:.1f} — primary stream is mostly idle; check largest_idle_gaps + launch_kernel_count")
    if top_kernels and top_kernels[0]["self_pct_of_active"] > 60:
        warnings.append(f"top kernel ({top_kernels[0]['short_name']}) dominates >60% of active time")

    return {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "captured_from": str(sqlite_path),
        "analysis_window_ns": window_ns,
        "wall_ns": wall_ns,
        "gpu": gpu_info,
        "streams": {
            "primary_stream_id": primary_stream,
            "active_streams": active_streams,
            "per_stream": per_stream,
        },
        "totals_primary_stream": {
            "gpu_active_ms": primary_active_ns / 1e6,
            "gpu_idle_ms":   primary_idle_ns / 1e6,
            "gpu_util_pct":  util,
            "kernel_count":  int(primary["kernel_count"] or 0),
        },
        "top_kernels": top_kernels,
        "largest_idle_gaps": largest_gaps,
        "cuda_api": {
            "launch_kernel_count": launch_kernel_count,
            "launch_kernel_total_ms": launch_kernel_total / 1e6,
            "launch_kernel_avg_us": (launch_kernel_total / launch_kernel_count / 1e3)
                                    if launch_kernel_count else 0.0,
            "graph_launch_count": graph_launch_count,
            "graph_launch_total_ms": graph_launch_total / 1e6,
            "launch_ratio_graph_to_eager": graph_ratio,
            "verdict": verdict,
        },
        "memcpy": mcpy,
        "recipes_run": ["top_kernels_by_self_time", "largest_idle_gaps",
                        "api_launch_counts", "memcpy_aggregate",
                        "per_stream_breakdown"],
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Diff mode
# ---------------------------------------------------------------------------

def diff_summaries(base: dict, patched: dict) -> dict:
    base_k = {k["short_name"]: k for k in base.get("top_kernels", [])}
    pat_k  = {k["short_name"]: k for k in patched.get("top_kernels", [])}
    all_names = sorted(set(base_k) | set(pat_k))
    rows = []
    for n in all_names:
        b = base_k.get(n, {}).get("self_ms", 0.0)
        p = pat_k.get(n, {}).get("self_ms", 0.0)
        delta = p - b
        pct = (delta / b * 100.0) if b > 0 else None
        rows.append({"kernel": n, "baseline_ms": b, "patched_ms": p,
                     "delta_ms": delta, "delta_pct": pct})
    rows.sort(key=lambda r: abs(r["delta_ms"]), reverse=True)
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "baseline_active_ms": base.get("totals_primary_stream", {}).get("gpu_active_ms"),
        "patched_active_ms":  patched.get("totals_primary_stream", {}).get("gpu_active_ms"),
        "kernel_diff": rows,
    }


# ---------------------------------------------------------------------------
# Query mode
# ---------------------------------------------------------------------------

DENY_RE = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|pragma|replace)\b",
                     re.IGNORECASE)


def query_mode(sqlite_path: Path, sql: str, limit: int) -> dict:
    if DENY_RE.search(sql):
        return {"ok": False, "error": "only SELECT/WITH allowed"}
    if not re.match(r"^\s*(select|with)\b", sql, re.IGNORECASE):
        return {"ok": False, "error": "query must start with SELECT or WITH"}
    if not re.search(r"\blimit\b", sql, re.IGNORECASE):
        sql = f"{sql.rstrip().rstrip(';')} LIMIT {limit}"
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        rows = query_sqlite(conn, sql)
        return {"ok": True, "row_count": len(rows), "rows": rows}
    except sqlite3.Error as e:
        return {"ok": False, "error": f"sqlite error: {e}", "sql": sql}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    # default has no subcommand; we treat absence as "summarize"
    ap.add_argument("--sqlite")
    ap.add_argument("--out-dir")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--stream-id", type=int, default=None)
    ap.add_argument("--window-ns", default=None, help="start_ns,end_ns")

    diff_ap = sub.add_parser("diff")
    diff_ap.add_argument("--baseline", required=True)
    diff_ap.add_argument("--patched",  required=True)
    diff_ap.add_argument("--out",      required=True)

    q_ap = sub.add_parser("query")
    q_ap.add_argument("--sqlite", required=True)
    q_ap.add_argument("--sql",    required=True)
    q_ap.add_argument("--limit",  type=int, default=50)

    args = ap.parse_args()

    if args.cmd == "diff":
        base = json.loads(Path(args.baseline).read_text())
        pat  = json.loads(Path(args.patched).read_text())
        out  = diff_summaries(base, pat)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"[nsys-timeline-sql:diff] wrote {args.out}")
        return

    if args.cmd == "query":
        out = query_mode(Path(args.sqlite), args.sql, args.limit)
        print(json.dumps(out, indent=2, default=str))
        return

    # Default = summarize
    if not args.sqlite or not args.out_dir:
        ap.error("--sqlite and --out-dir required for summarize mode")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    win = None
    if args.window_ns:
        a, b = args.window_ns.split(",")
        win = (int(a), int(b))
    out = summarize(Path(args.sqlite), out_dir, top_n=args.top_n,
                    stream_id=args.stream_id, window_ns=win)
    summary_path = out_dir / "timeline_summary.json"
    summary_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[nsys-timeline-sql] wrote {summary_path}")


if __name__ == "__main__":
    main()
