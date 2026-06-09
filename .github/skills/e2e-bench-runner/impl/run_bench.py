#!/usr/bin/env python3
"""Skill impl: e2e-bench-runner.

Reuses the prompt generation logic from
results/4way_bench/scripts/run_bench_4way.py but writes a structured
bench_summary.json per the contract in ../SKILL.md.

CLI:
    python run_bench.py --url URL --backend {sglang,vllm} --tag TAG \
        --num-runs N --out-dir DIR [--regimes R_short,R_medium,R_long]

All failures are written to bench_summary.json as {"ok": false, ...}.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

SCHEMA_VERSION = 0

REGIME_DEFS = {
    "R_short":  {"num_prompts":  8, "prompt_words":  200, "max_new":  64, "concurrency":  1},
    "R_medium": {"num_prompts": 16, "prompt_words":  800, "max_new": 256, "concurrency":  8},
    "R_long":   {"num_prompts":  8, "prompt_words": 2000, "max_new": 256, "concurrency": 16},
}

WORDS = ("machine learning artificial intelligence deep neural network "
         "training inference optimization performance benchmark profiling "
         "kernel implementation source code framework library backend "
         "transformer attention encoder decoder embedding tokenizer batch "
         "processing efficiency throughput latency memory bandwidth").split()

RELIABLE_STDDEV_PCT = 8.0
TIMEOUT_S = 600


def make_prompts(n: int, words: int, seed: int = 2026) -> list[str]:
    rng = random.Random(seed)
    return [" ".join(rng.choice(WORDS) for _ in range(words)) for _ in range(n)]


def send_sglang(url: str, prompt: str, max_new: int) -> dict:
    r = requests.post(f"{url}/generate", json={
        "text": prompt,
        "sampling_params": {"max_new_tokens": max_new, "temperature": 0.0, "ignore_eos": True},
    }, timeout=TIMEOUT_S, stream=False)
    r.raise_for_status()
    out = r.json()
    return {"text": out.get("text", ""), "meta": out.get("meta_info", {})}


def send_vllm(url: str, prompt: str, max_new: int, model_name: str) -> dict:
    r = requests.post(f"{url}/v1/completions", json={
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_new,
        "temperature": 0.0,
        "ignore_eos": True,
    }, timeout=TIMEOUT_S)
    r.raise_for_status()
    out = r.json()
    return {"text": out["choices"][0]["text"], "meta": out.get("usage", {})}


def run_single_regime(url: str, backend: str, regime: dict, prompts: list[str],
                      model_name: str) -> dict:
    """Run one regime once. Returns per-request records + regime aggregates."""
    send_fn = send_sglang if backend == "sglang" else (
        lambda u, p, m: send_vllm(u, p, m, model_name))

    records: list[dict] = []
    errors: list[str] = []

    def worker(idx: int, prompt: str):
        t0 = time.perf_counter()
        try:
            res = send_fn(url, prompt, regime["max_new"])
            t1 = time.perf_counter()
            out_tokens = (res["meta"].get("completion_tokens")
                          or res["meta"].get("output_tokens")
                          or len(res["text"].split()))
            return {"idx": idx, "ok": True, "e2e_s": t1 - t0,
                    "output_tokens": out_tokens, "error": None}
        except Exception as e:
            return {"idx": idx, "ok": False, "e2e_s": None,
                    "output_tokens": 0, "error": f"{type(e).__name__}: {e}"}

    wall_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=regime["concurrency"]) as ex:
        futures = [ex.submit(worker, i, p) for i, p in enumerate(prompts)]
        for f in concurrent.futures.as_completed(futures):
            rec = f.result()
            records.append(rec)
            if not rec["ok"]:
                errors.append(rec["error"])
    wall_s = time.perf_counter() - wall_start

    ok_records = [r for r in records if r["ok"]]
    completion_rate = len(ok_records) / max(1, len(records))
    total_out_tokens = sum(r["output_tokens"] for r in ok_records)

    return {
        "wall_s": wall_s,
        "req_per_s": len(ok_records) / wall_s if wall_s > 0 else 0.0,
        "tokens_per_s": total_out_tokens / wall_s if wall_s > 0 else 0.0,
        "total_out_tokens": total_out_tokens,
        "completion_rate": completion_rate,
        "errors": errors[:5],
        "records": records,
    }


def pct(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    k = int(round((p / 100.0) * (len(s) - 1)))
    return s[max(0, min(len(s) - 1, k))]


def aggregate_runs(regime_runs: list[dict]) -> dict:
    """Drop run 0, compute mean/stddev/p50/p99 across remaining runs."""
    used = regime_runs[1:] if len(regime_runs) >= 2 else regime_runs
    req_per_s = [r["req_per_s"] for r in used]
    tok_per_s = [r["tokens_per_s"] for r in used]
    wall = [r["wall_s"] for r in used]

    # TTFT/ITL: we don't have token-arrival timestamps in non-streaming mode,
    # so e2e_s is our only latency. p50/p99 over all requests across used runs.
    all_e2e = [rec["e2e_s"] for r in used for rec in r["records"]
               if rec.get("ok") and rec.get("e2e_s") is not None]

    def stat(xs: list[float]) -> dict:
        if not xs:
            return {"mean": None, "stddev": None, "stddev_pct": None, "runs": xs}
        m = statistics.fmean(xs)
        sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        return {"mean": m, "stddev": sd,
                "stddev_pct": (sd / m * 100.0) if m else None,
                "runs": xs}

    req_stat = stat(req_per_s)
    reliable = (req_stat["stddev_pct"] is not None
                and req_stat["stddev_pct"] <= RELIABLE_STDDEV_PCT)

    return {
        "req_per_s":    req_stat,
        "tokens_per_s": stat(tok_per_s),
        "wall_s":       stat(wall),
        "e2e_ms":       {"p50": (pct(all_e2e, 50) or 0) * 1000,
                         "p99": (pct(all_e2e, 99) or 0) * 1000,
                         "count": len(all_e2e)},
        "completion_rate": (sum(r["completion_rate"] for r in used) / len(used)
                            if used else 0.0),
        "reliable": reliable,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--backend", choices=["sglang", "vllm"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--num-runs", type=int, default=3)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--regimes", default="R_short,R_medium,R_long",
                    help="Comma-separated regime IDs; used only when --regimes-file is absent")
    ap.add_argument("--regimes-file", default=None,
                    help="YAML file defining regimes (overrides --regimes and built-in defaults). "
                         "Schema: {regimes: {<id>: {num_prompts,prompt_words,max_new,concurrency}}}")
    ap.add_argument("--model-name", default="qwen3-30b-a3b-moe",
                    help="vLLM endpoint requires this; sglang ignores it")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "per_run").mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "bench_summary.json"

    # Resolve regime definitions (built-in defaults OR loaded from YAML)
    regime_defs = dict(REGIME_DEFS)
    regimes_source = "built_in"
    if args.regimes_file:
        try:
            import yaml  # PyYAML is a transitive dep of vllm/sglang envs
        except ImportError:
            json.dump({"schema_version": SCHEMA_VERSION, "ok": False,
                       "error": "PyYAML required for --regimes-file"},
                      summary_path.open("w"), indent=2)
            sys.exit(1)
        try:
            user_defs = yaml.safe_load(Path(args.regimes_file).read_text()) or {}
            user_regimes = user_defs.get("regimes", {})
            if not isinstance(user_regimes, dict) or not user_regimes:
                raise ValueError("YAML must contain top-level 'regimes:' mapping with ≥1 entry")
            # Validate each regime has required fields
            REQUIRED = {"num_prompts", "prompt_words", "max_new", "concurrency"}
            for r_id, r_def in user_regimes.items():
                missing = REQUIRED - set(r_def or {})
                if missing:
                    raise ValueError(f"regime '{r_id}' missing fields: {missing}")
            regime_defs = {str(k): dict(v) for k, v in user_regimes.items()}
            regimes_source = str(args.regimes_file)
            regimes = list(regime_defs.keys())
        except Exception as e:
            json.dump({"schema_version": SCHEMA_VERSION, "ok": False,
                       "error": f"regimes-file load failed: {e}"},
                      summary_path.open("w"), indent=2)
            sys.exit(1)
    else:
        regimes = [r.strip() for r in args.regimes.split(",")]
        for r in regimes:
            if r not in regime_defs:
                json.dump({"schema_version": SCHEMA_VERSION, "ok": False,
                           "error": f"unknown regime '{r}'; known: {list(regime_defs)}"},
                          summary_path.open("w"), indent=2)
                sys.exit(1)

    # Pre-flight: server health
    try:
        # sglang exposes /health, vllm exposes /health too on most builds; if not, try /v1/models
        h = requests.get(f"{args.url}/health", timeout=10)
        if h.status_code != 200:
            raise RuntimeError(f"health returned {h.status_code}")
    except Exception as e:
        try:
            requests.get(f"{args.url}/v1/models", timeout=10).raise_for_status()
        except Exception as e2:
            json.dump({"schema_version": SCHEMA_VERSION, "ok": False,
                       "error": f"server not ready at {args.url}: {e2}"},
                      summary_path.open("w"), indent=2)
            sys.exit(1)

    warnings: list[str] = []
    if args.num_runs < 2:
        warnings.append("num_runs < 2: stddev unavailable, results NOT reliable")

    out = {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "tag": args.tag,
        "url": args.url,
        "backend": args.backend,
        "num_runs_total": args.num_runs,
        "num_runs_used":  max(1, args.num_runs - 1),
        "regimes_source": regimes_source,
        "regimes": {},
        "warnings": warnings,
    }

    for r_id in regimes:
        regime = regime_defs[r_id]
        prompts = make_prompts(regime["num_prompts"], regime["prompt_words"])
        per_run_results = []
        for run_idx in range(args.num_runs):
            t = time.perf_counter()
            try:
                res = run_single_regime(args.url, args.backend, regime, prompts,
                                        args.model_name)
            except Exception as e:
                res = {"wall_s": 0, "req_per_s": 0, "tokens_per_s": 0,
                       "total_out_tokens": 0, "completion_rate": 0.0,
                       "errors": [str(e)], "records": []}
            per_run_results.append(res)
            # Dump per-run records
            run_path = out_dir / "per_run" / f"{r_id}_run{run_idx}.json"
            json.dump({
                "regime": r_id, "run": run_idx, "elapsed_wall_s": time.perf_counter() - t,
                **{k: v for k, v in res.items() if k != "records"},
                "records": res["records"],
            }, run_path.open("w"), indent=2)

        agg = aggregate_runs(per_run_results)
        out["regimes"][r_id] = {**regime, **agg}

    summary_path.write_text(json.dumps(out, indent=2))
    print(f"[e2e-bench-runner] wrote {summary_path}")


if __name__ == "__main__":
    main()
