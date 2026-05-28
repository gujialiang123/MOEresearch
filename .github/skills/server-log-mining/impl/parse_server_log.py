#!/usr/bin/env python3
"""Skill impl: server-log-mining.

Parse a sglang server.log into a structured features JSON. See `../SKILL.md`
for the output contract.

Safe to call on any log: never raises. Missing fields become null, warnings
list which patterns failed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path


SCHEMA_VERSION = 1


def _server_args_field(text: str, key: str, cast=str):
    m = re.search(rf"\b{re.escape(key)}=([^,)\s]+)", text)
    if m is None:
        return None
    raw = m.group(1)
    try:
        if cast is int:
            return int(raw)
        if cast is float:
            return float(raw)
        if cast is bool:
            return raw.lower() == "true"
        return raw.strip("'\"")
    except ValueError:
        return None


def parse_cuda_graph_configured(text: str) -> tuple[list[int] | None, str | None]:
    m = re.search(r"cuda_graph_bs=\[([^\]]+)\]", text)
    if not m:
        return None, "no cuda_graph_bs in ServerArgs"
    try:
        return [int(x) for x in m.group(1).split(",")], None
    except ValueError as e:
        return None, f"cuda_graph_bs parse error: {e}"


def parse_cuda_graph_captured(text: str) -> tuple[list[int] | None, str | None]:
    m = re.search(r"Capture cuda graph bs \[([^\]]+)\]", text)
    if not m:
        return None, "no Capture cuda graph bs line"
    try:
        return [int(x.strip()) for x in m.group(1).split(",")], None
    except ValueError as e:
        return None, f"capture cuda graph bs parse error: {e}"


def parse_cuda_graph_capture_seconds(text: str) -> tuple[float | None, str | None]:
    m = re.search(r"Capture cuda graph end\. Time elapsed: ([0-9.]+) s", text)
    return (float(m.group(1)) if m else None,
            None if m else "no Capture cuda graph end line")


def parse_kv_cache_size(text: str) -> tuple[float | None, str | None]:
    m = re.search(
        r"KV Cache is allocated\. #tokens: \d+, K size: ([0-9.]+) GB, V size: ([0-9.]+) GB", text)
    if not m:
        return None, "no KV Cache is allocated line"
    return float(m.group(1)) + float(m.group(2)), None


def parse_max_total_num_tokens(text: str) -> int | None:
    m = re.search(r"\bmax_total_num_tokens=(\d+)", text)
    return int(m.group(1)) if m else None


def parse_context_len_runtime(text: str) -> int | None:
    m = re.search(r"\bcontext_len=(\d+)", text)
    return int(m.group(1)) if m else None


def parse_model_load_seconds(text: str) -> tuple[float | None, str | None]:
    m = re.search(r"Load weight end\. elapsed=([0-9.]+) s", text)
    return (float(m.group(1)) if m else None,
            None if m else "no Load weight end line")


_TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")


def parse_ts(s: str) -> datetime | None:
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def parse_server_startup_seconds(text: str) -> tuple[float | None, str | None]:
    first_ts = None
    ready_ts = None
    for line in text.splitlines():
        m = _TS_RE.search(line)
        if not m:
            continue
        ts = parse_ts(m.group(1))
        if first_ts is None and "server_args=" in line:
            first_ts = ts
        if "Uvicorn running" in line or "fired up and ready to roll" in line:
            ready_ts = ts
            break
    if first_ts is None or ready_ts is None:
        return None, "couldn't bracket startup window"
    return (ready_ts - first_ts).total_seconds(), None


_DECODE_RE = re.compile(
    r"#running-req: (\d+),\s*#token: (\d+),\s*token usage: ([0-9.]+),"
    r".*?#queue-req: (\d+)"
)

_PREFILL_RE = re.compile(
    r"Prefill batch,\s*#new-seq: (\d+),\s*#new-token: (\d+),"
    r".*?#running-req: (\d+),\s*#queue-req: (\d+)"
)


def scan_batch_lines(text: str) -> dict:
    peak_token_usage = 0.0
    peak_running = 0
    peak_queue = 0
    decode_count = 0
    prefill_count = 0
    for line in text.splitlines():
        if "Decode batch" in line:
            m = _DECODE_RE.search(line)
            if m:
                decode_count += 1
                run, tok, tu, q = int(m.group(1)), int(m.group(2)), float(m.group(3)), int(m.group(4))
                peak_running = max(peak_running, run)
                peak_queue = max(peak_queue, q)
                peak_token_usage = max(peak_token_usage, tu)
        elif "Prefill batch" in line:
            m = _PREFILL_RE.search(line)
            if m:
                prefill_count += 1
                run, q = int(m.group(3)), int(m.group(4))
                peak_running = max(peak_running, run)
                peak_queue = max(peak_queue, q)
    return {
        "peak_token_usage": peak_token_usage,
        "peak_running_reqs": peak_running,
        "peak_queue_reqs": peak_queue,
        "decode_batch_count": decode_count,
        "prefill_batch_count": prefill_count,
    }


_KV_FULL_RE = re.compile(r"KV cache pool is full", re.IGNORECASE)
_RETRACT_RE = re.compile(r"retract", re.IGNORECASE)
_OOM_RE = re.compile(r"out of memory|OutOfMemoryError|CUDA out of memory", re.IGNORECASE)
_CRASH_RE = re.compile(r"Traceback \(most recent call last\)|FATAL|core dumped|Segmentation fault",
                        re.IGNORECASE)


def count(text: str, regex: re.Pattern) -> int:
    return sum(1 for _ in regex.finditer(text))


def mine(server_log: Path, max_bytes: int = 4_000_000) -> dict:
    out: dict = {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "input_log": str(server_log),
        "input_log_bytes": None,
        "fields": {},
        "warnings": [],
        "errors": [],
    }
    if not server_log.exists():
        out["ok"] = False
        out["errors"].append(f"file not found: {server_log}")
        return out

    try:
        size = server_log.stat().st_size
        out["input_log_bytes"] = size
        with open(server_log, "rb") as f:
            if size > max_bytes:
                head = f.read(max_bytes // 2)
                f.seek(size - max_bytes // 2)
                tail = f.read()
                raw = head + b"\n... [TRUNCATED MIDDLE] ...\n" + tail
            else:
                raw = f.read()
        text = raw.decode("utf-8", errors="ignore")
    except OSError as e:
        out["ok"] = False
        out["errors"].append(f"read error: {e}")
        return out

    f: dict = out["fields"]
    f["model_path"] = _server_args_field(text, "model_path", str)
    f["chunked_prefill_size"] = _server_args_field(text, "chunked_prefill_size", int)
    f["max_prefill_tokens"] = _server_args_field(text, "max_prefill_tokens", int)
    f["max_running_requests"] = _server_args_field(text, "max_running_requests", int)
    f["mem_fraction_static"] = _server_args_field(text, "mem_fraction_static", float)
    f["tp_size"] = _server_args_field(text, "tp_size", int)
    f["schedule_policy"] = _server_args_field(text, "schedule_policy", str)
    f["schedule_conservativeness"] = _server_args_field(text, "schedule_conservativeness", float)
    f["attention_backend"] = _server_args_field(text, "attention_backend", str)
    f["disable_radix_cache"] = _server_args_field(text, "disable_radix_cache", bool)
    f["disable_cuda_graph"] = _server_args_field(text, "disable_cuda_graph", bool)
    f["num_continuous_decode_steps"] = _server_args_field(text, "num_continuous_decode_steps", int)
    f["page_size"] = _server_args_field(text, "page_size", int)

    cgs_cfg, w = parse_cuda_graph_configured(text)
    f["cuda_graph_bs_configured"] = cgs_cfg
    if w:
        out["warnings"].append(w)
    cgs_cap, w = parse_cuda_graph_captured(text)
    f["cuda_graph_bs_captured"] = cgs_cap
    if w:
        out["warnings"].append(w)
    cgs_t, w = parse_cuda_graph_capture_seconds(text)
    f["cuda_graph_capture_seconds"] = cgs_t
    if w:
        out["warnings"].append(w)

    f["kv_cache_size_gb_total"], w = parse_kv_cache_size(text)
    if w:
        out["warnings"].append(w)
    f["max_total_num_tokens"] = parse_max_total_num_tokens(text)
    f["context_len"] = parse_context_len_runtime(text)
    f["model_load_seconds"], w = parse_model_load_seconds(text)
    if w:
        out["warnings"].append(w)
    f["server_startup_seconds"], w = parse_server_startup_seconds(text)
    if w:
        out["warnings"].append(w)

    f.update(scan_batch_lines(text))

    f["kv_pool_full_events"] = count(text, _KV_FULL_RE)
    f["retract_events"] = count(text, _RETRACT_RE)
    f["oom_events"] = count(text, _OOM_RE)
    f["crash_events"] = count(text, _CRASH_RE)

    f["cuda_graph_too_small"] = bool(
        cgs_cap and f["peak_running_reqs"] and f["peak_running_reqs"] > max(cgs_cap)
    )
    f["at_capacity"] = bool(f["peak_token_usage"] and f["peak_token_usage"] >= 0.95)
    f["near_capacity"] = bool(f["peak_token_usage"] and f["peak_token_usage"] >= 0.80)
    # Concurrency-capped: server admits up to max_running_requests, queue holds
    # the rest. Strong signal that lifting max_running_requests would help tail.
    mrr = f.get("max_running_requests")
    pr  = f.get("peak_running_reqs") or 0
    pq  = f.get("peak_queue_reqs") or 0
    f["concurrency_capped"] = bool(
        mrr is not None and pr >= int(0.95 * mrr) and pq >= 0.5 * mrr
    )
    # Captured cuda_graph upper bound is also relevant: if max_running_requests
    # > captured max, runtime falls off the fast path even before queueing.
    if cgs_cap and mrr is not None:
        f["max_running_above_cuda_graph"] = mrr > max(cgs_cap)
    else:
        f["max_running_above_cuda_graph"] = None

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-bytes", type=int, default=4_000_000)
    args = ap.parse_args()

    result = mine(Path(args.server_log), max_bytes=args.max_bytes)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    summary = {
        "ok": result["ok"],
        "warnings": len(result["warnings"]),
        "cuda_graph_too_small": result["fields"].get("cuda_graph_too_small"),
        "at_capacity": result["fields"].get("at_capacity"),
        "peak_running_reqs": result["fields"].get("peak_running_reqs"),
        "cuda_graph_bs_captured_max": (
            max(result["fields"].get("cuda_graph_bs_captured") or [0])
        ),
        "max_running_requests": result["fields"].get("max_running_requests"),
        "peak_token_usage": result["fields"].get("peak_token_usage"),
    }
    print(json.dumps(summary, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
