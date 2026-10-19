#!/usr/bin/env python3
"""Pin GSM8K parquet, export stable rows, audit saved history, and seal candidates.

Requires requests and pyarrow; never imports a model or Hugging Face datasets.
Downloads and generated data default to scratch. Historical IDs are evidence of
exposure, not proof that an unversioned historical dataset matched today's rows.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

DATASET = "openai/gsm8k"
REVISION = "740312add88f781978c0658806c59bc2815b9866"
RANGES = {"C_fit": [0, 128], "C_diag": [128, 256],
          "D_select": [256, 512], "D_validate": [512, 1024]}
RECORD_KEYS = {"gold", "gen_token_ids", "generated_tokens", "gen_len", "glen",
               "kl_series", "kl_p8_pk", "eos_logp_at_eos", "t_ready"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def get_json(url):
    import requests
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def download(url: str, target: Path):
    import requests
    if target.is_file():
        return
    temporary = target.with_name(target.name + ".partial")
    with requests.get(url, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        with temporary.open("wb") as output:
            for chunk in response.iter_content(1 << 20):
                output.write(chunk)
    temporary.replace(target)


def prepare_dataset(out: Path):
    import pyarrow.parquet as pq
    api_url = f"https://huggingface.co/api/datasets/{DATASET}/revision/{REVISION}"
    api_path = out / "source_api.json"
    api = json.loads(api_path.read_text()) if api_path.exists() else get_json(api_url)
    if api.get("sha") != REVISION or api.get("id") != DATASET:
        raise ValueError("Pinned API identity/revision does not match")
    if not api_path.exists():
        atomic_json(api_path, api)
    tree_url = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}/main"
    tree_path = out / "source_tree.json"
    tree = json.loads(tree_path.read_text()) if tree_path.exists() else get_json(tree_url)
    if not tree_path.exists():
        atomic_json(tree_path, tree)
    expected_sha = {item["path"]: item.get("lfs", {}).get("oid") for item in tree}
    rows, sources = {}, {}
    for split, expected in (("train", 7473), ("test", 1319)):
        name = f"main/{split}-00000-of-00001.parquet"
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{name}"
        parquet = out / name.replace("/", "_")
        download(url, parquet)
        actual_sha = sha(parquet.read_bytes())
        if not expected_sha.get(name) or actual_sha != expected_sha[name]:
            raise ValueError(f"Parquet hash differs from pinned upstream LFS object: {name}")
        records = pq.read_table(parquet).to_pylist()
        if len(records) != expected:
            raise ValueError(f"Unexpected {split} row count: {len(records)}")
        output = out / f"{split}.jsonl"
        exported = []
        for index, record in enumerate(records):
            question_hash = sha(record["question"].encode("utf-8"))
            exported.append({"sample_id": index, "id": index, "split": split,
                             "question": record["question"], "answer": record["answer"],
                             "question_hash": question_hash, "question_sha256": question_hash,
                             "dataset_revision": REVISION})
        content = b"".join(canonical(row) + b"\n" for row in exported)
        if output.exists() and output.read_bytes() != content:
            raise ValueError(f"Refusing to overwrite inconsistent dataset export: {output}")
        if not output.exists():
            output.write_bytes(content)
        rows[split] = exported
        sources[split] = {"url": url, "path": str(parquet),
                          "parquet_sha256": actual_sha,
                          "upstream_lfs_sha256": expected_sha[name],
                          "jsonl_path": str(output), "jsonl_sha256": sha(content),
                          "rows": expected, "question_hash_list_sha256":
                          sha(canonical([r["question_hash"] for r in exported]))}
    return rows, {"dataset": DATASET, "subset": "main", "revision": REVISION,
                  "api_url": api_url, "source_api_sha256": sha(api_path.read_bytes()),
                  "tree_api_url": tree_url, "source_tree_sha256": sha(tree_path.read_bytes()),
                  "sources": sources, "question_hash_definition": "SHA256(exact question UTF-8 bytes)"}


def dictionaries(value):
    """Traverse JSON containers without revisiting scalar token/float arrays."""
    if isinstance(value, dict):
        yield value
        for item in value.values():
            if isinstance(item, (dict, list)):
                yield from dictionaries(item)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                yield from dictionaries(item)


def compact_ranges(ids):
    ids = sorted(set(ids))
    result = []
    for index in ids:
        if result and result[-1][1] == index:
            result[-1][1] = index + 1
        else:
            result.append([index, index + 1])
    return result


def context(metadata):
    result = {}
    for value in metadata:
        for item in (value, value.get("args", {}), value.get("config", {})):
            if not isinstance(item, dict):
                continue
            for key in ("split", "range", "calib_range", "calib_n", "offset", "limit", "sample_ids"):
                if key in item:
                    result[key] = item[key]
    return result


def source_split(path: Path):
    """Only checked repository script families; inference is labeled in outputs."""
    run = path.parts[0]
    rules = [
        (r"_v17_gsm8k", "test", "scripts/run_v17_gsm8k_topk.py"),
        (r"_v18_dynamic_topk", "test", "scripts/run_v18_dynamic_topk.py"),
        (r"_v20_free_gen|_v21_", "test", "scripts/run_v20_dynamic_topk_free_generation.py"),
        (r"_v22_", "test", "scripts/run_v22_teacher_forced_eos.py"),
        (r"_v23_|_v24|_v28", "test", "scripts/run_gsm8k_configs.py"),
        (r"_v25_", "test", "scripts/run_v25_answer_readiness.py"),
        (r"_v26_", "test", "scripts/run_v26_direct_effect_probe.py"),
        (r"_v29_partial", "test", "scripts/run_v29_partial_renorm.py"),
        (r"_v30_gain_controls$", "test", "scripts/run_v30_gain_controls.py"),
        (r"_v31_pulse", "test", "scripts/run_v31_pulse_recovery.py"),
        (r"_v32_", "train", "scripts/run_v32_pulse_recovery.py"),
    ]
    for pattern, split, source in rules:
        if re.search(pattern, run):
            return split, source
    return None, None


def numeric_gold(text):
    if text is None:
        return None
    return str(text).split("####")[-1].strip().replace(",", "")


def audit_history(repo: Path, rows):
    result_root = repo / "results"
    all_files = sorted(p for p in result_root.rglob("*")
                       if p.suffix in (".json", ".jsonl") and
                       "state_calibration_study" not in str(p.relative_to(result_root)))
    file_inventory, failures = [], []
    by_run, cache = {}, {}
    exact_hashes, inferred_ids, declared_ids = set(), defaultdict(set), defaultdict(set)
    unresolved = []
    lookup = defaultdict(list)
    for split, records in rows.items():
        for row in records:
            lookup[row["question_hash"]].append((split, row["sample_id"]))
    def metadata_for(file):
        values = []
        for parent in (file.parent, *file.parents):
            if parent == result_root:
                break
            for name in ("summary.json", "manifest.json"):
                p = parent / name
                if p not in cache:
                    try:
                        cache[p] = json.loads(p.read_text()) if p.exists() else {}
                    except (OSError, ValueError):
                        cache[p] = {}
                if isinstance(cache[p], dict):
                    values.append(cache[p])
        return list(reversed(values))
    for file in all_files:
        rel = file.relative_to(result_root)
        run = rel.parts[0]
        known_gsm = bool(re.search(r"_v(?:17_gsm8k|18_dynamic_topk|20_free_gen|21_|22_|23_|24|25_|26_|28|29|30_|31_|32_)", run))
        own_metadata = []
        if file.suffix == ".json":
            try:
                own = json.loads(file.read_text())
                if isinstance(own, dict):
                    own_metadata.append(own)
            except (OSError, ValueError):
                pass  # The full parse below records the error.
        ctx = context(metadata_for(file) + own_metadata)
        fallback_split, source = source_split(rel)
        split = ctx.get("split", fallback_split)
        split_basis = "saved_metadata" if ctx.get("split") else "current_script_family" if source else None
        entry = {"path": str(file.relative_to(repo)), "sha256": sha(file.read_bytes()),
                 "bytes": file.stat().st_size, "gsm8k_family": known_gsm,
                 "records": 0, "sample_records": 0, "split": split,
                 "split_basis": split_basis, "source_script": source,
                 "sample_ids": [], "question_hash_records": 0,
                 "gold_checked": 0, "gold_mismatches": 0, "unresolved_sample_records": 0}
        ids, schemas = set(), Counter()
        if known_gsm:
            stats = by_run.setdefault(run, {"files": 0, "sample_records": 0,
                "observed_ids": defaultdict(set), "declared_ids": defaultdict(set),
                "exact_hash_records": 0, "gold_checked": 0, "gold_mismatches": 0,
                "unresolved_sample_records": 0})
            stats["files"] += 1
        try:
            if file.suffix == ".jsonl":
                def values():
                    with file.open() as handle:
                        for line_number, line in enumerate(handle, 1):
                            if line.strip():
                                try:
                                    yield json.loads(line)
                                except ValueError as exc:
                                    failures.append({"file": str(rel), "line": line_number, "error": str(exc)})
            else:
                def values():
                    yield json.loads(file.read_text())
            for value in values():
                entry["records"] += 1
                for record in dictionaries(value):
                    # Store key names, never arbitrary historical secrets or full output text.
                    if known_gsm and len(schemas) < 32:
                        schemas[tuple(sorted(record))] += 1
                    qh = record.get("question_hash", record.get("question_sha256"))
                    if isinstance(record.get("question"), str):
                        qh = sha(record["question"].encode("utf-8"))
                    if isinstance(qh, str) and qh in lookup:
                        exact_hashes.add(qh)
                        entry["question_hash_records"] += 1
                    sid = record.get("sample_id", record.get("qi", record.get("id")))
                    sample = (isinstance(sid, int) and not isinstance(sid, bool)
                              and (bool(RECORD_KEYS.intersection(record)) or "sample_id" in record))
                    if not sample or not known_gsm:
                        continue
                    entry["sample_records"] += 1
                    row_split = record.get("split", split)
                    # The generic old harness wrote indices relative to args.offset.
                    offset = int(ctx.get("offset", 0)) if source == "scripts/run_gsm8k_configs.py" else 0
                    mapped = sid + offset
                    if isinstance(qh, str) and qh in lookup:
                        for matched_split, matched_id in lookup[qh]:
                            inferred_ids[matched_split].add(matched_id)
                            stats["observed_ids"][matched_split].add(matched_id)
                    elif row_split in rows and 0 <= mapped < len(rows[row_split]):
                        inferred_ids[row_split].add(mapped)
                        stats["observed_ids"][row_split].add(mapped)
                        ids.add(mapped)
                        if "gold" in record:
                            entry["gold_checked"] += 1
                            if numeric_gold(record["gold"]) != numeric_gold(rows[row_split][mapped]["answer"]):
                                entry["gold_mismatches"] += 1
                    else:
                        entry["unresolved_sample_records"] += 1
            if known_gsm:
                # Declarations are exposure exclusions, not a claim of successful execution.
                declared = ctx.get("sample_ids", [])
                if not declared and isinstance(ctx.get("range"), list) and len(ctx["range"]) == 2:
                    declared = range(*ctx["range"])
                if not declared and isinstance(ctx.get("limit"), int) and split in rows:
                    start = int(ctx.get("offset", 0))
                    declared = range(start, start + ctx["limit"])
                if split in rows:
                    valid = {x for x in declared if isinstance(x, int) and 0 <= x < len(rows[split])}
                    declared_ids[split].update(valid)
                    stats["declared_ids"][split].update(valid)
                calib = ctx.get("calib_range")
                if isinstance(calib, list) and len(calib) == 2:
                    declared_ids["train"].update(range(*calib))
                    stats["declared_ids"]["train"].update(range(*calib))
                if isinstance(ctx.get("calib_n"), int):
                    declared_ids["train"].update(range(ctx["calib_n"]))
                    stats["declared_ids"]["train"].update(range(ctx["calib_n"]))
                stats["sample_records"] += entry["sample_records"]
                stats["exact_hash_records"] += entry["question_hash_records"]
                for key in ("gold_checked", "gold_mismatches", "unresolved_sample_records"):
                    stats[key] += entry[key]
                if entry["unresolved_sample_records"]:
                    unresolved.append({"file": str(rel), "reason": "sample IDs lack resolvable split/range", "records": entry["unresolved_sample_records"]})
        except (OSError, ValueError, TypeError) as exc:
            failures.append({"file": str(rel), "error": f"{type(exc).__name__}: {exc}"})
        entry["sample_ids"] = sorted(ids)
        entry["schemas"] = [{"keys": list(keys), "count": count} for keys, count in schemas.items()]
        file_inventory.append(entry)
    # Known calibration artifact has no persisted prompt IDs or fitting count.
    if (result_root / "2026-07-20_v24_weight_ablation/norm_match_scalars.json").exists():
        unresolved.append({"file": "2026-07-20_v24_weight_ablation/norm_match_scalars.json",
            "reason": "Calibration exists, but original fitting IDs/count/revision were not saved; current script default n=64 is not execution evidence."})
    for run in ("2026-07-15_v17_gsm8k_topk", "2026-07-15_v18_dynamic_topk"):
        if run in by_run and by_run[run]["sample_records"] == 0:
            unresolved.append({"file": run,
                "reason": "Aggregate-only GSM8K results save limit but no per-question IDs or hashes; declaration is inferred using the current checked source's test-prefix convention."})
    consulted = sorted({v["source_script"] for v in file_inventory if v.get("source_script")} |
                        {"scripts/_harness.py", "scripts/calibrate_norm_match.py", "scripts/calibrate_v29_decode_norm.py"})
    sources = {name: sha((repo / name).read_bytes()) for name in consulted if (repo / name).exists()}
    for stats in by_run.values():
        for key in ("observed_ids", "declared_ids"):
            stats[key] = {split: {"count": len(ids), "ranges_half_open": compact_ranges(ids)}
                          for split, ids in stats[key].items()}
    audit = {"schema_version": 1, "scope": "All current-checkout results/**/*.json and results/**/*.jsonl, excluding state_calibration_study; no deleted/out-of-checkout files or complete Git-history reconstruction.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_revision": REVISION, "files_scanned": len(all_files),
        "historical_dataset_identity": "UNVERIFIED: old manifests/raw rows generally omit dataset revision and question hash; numeric IDs are mapped under documented split/source semantics only.",
        "independence_status": "INDEPENDENCE_UNVERIFIED",
        "limitations": ["No absence-of-use guarantee for missing, deleted, external, or aggregate-only results.",
            "Gold agreement is a consistency check, not unique question identification.",
            "Current script-family inference is weaker than a historical source snapshot.",
            "Declared ranges may include attempted but unfinished samples; they are excluded conservatively.",
            "Files were audited as saved data, not only manifests; exact per-file schemas/IDs are retained below."],
        "observed_id_coverage": {split: {"ids": sorted(ids), "ranges_half_open": compact_ranges(ids)}
                                 for split, ids in inferred_ids.items()},
        "declared_id_coverage": {split: {"ids": sorted(ids), "ranges_half_open": compact_ranges(ids)}
                                 for split, ids in declared_ids.items()},
        "exact_historical_question_hashes": sorted(exact_hashes),
        "unresolved": unresolved, "parse_failures": failures,
        "source_script_sha256": sources, "by_run": by_run, "files": file_inventory}
    return audit, inferred_ids, declared_ids, exact_hashes


def seal(out, rows, audit, observed, declared, exact_hashes):
    exposed_hashes = set(exact_hashes)
    for split in rows:
        for index in observed[split] | declared[split]:
            exposed_hashes.add(rows[split][index]["question_hash"])
    reserved = {r["question_hash"] for r in rows["train"][:1024]}
    candidates, excluded = [], []
    for row in rows["train"][1024:2048]:
        item = {"split": "train", "sample_id": row["sample_id"], "question_hash": row["question_hash"]}
        if row["question_hash"] in exposed_hashes or row["question_hash"] in reserved:
            excluded.append(item)
        else:
            candidates.append(item)
    result = {"schema_version": 1, "dataset": DATASET, "subset": "main", "revision": REVISION,
        "candidate_range_half_open": [1024, 2048], "candidate_count": len(candidates),
        "excluded_observed_or_reserved": excluded, "samples": candidates,
        "selection_rule": "ascending sample_id within train[1024,2048), exclude observed/declared exposure and exact question-hash overlap with train[0,1024)",
        "candidate_hash": sha(canonical(candidates)), "independence_status": "INDEPENDENCE_UNVERIFIED",
        "interpretation": "Prospective exclusion/seal only; not historically verified fresh data and not a formal P6 sample-size decision.",
        "unseal_gate": "Freeze primary comparisons, endpoints, quality constraint and final sample-size/power plan before any candidate model evaluation; final N must use a preregistered prefix without outcome-driven resampling.",
        "historical_audit_inventory_sha256": sha(canonical(audit["files"])),
        "created_utc": datetime.now(timezone.utc).isoformat()}
    target = out / "confirmation_seal.json"
    if target.exists():
        previous = json.loads(target.read_text())
        # Keep the original seal and its original inventory fingerprint. Later
        # audits can add evidence without rewriting this prospective candidate set.
        for key in ("revision", "candidate_hash"):
            if previous.get(key) != result[key]:
                raise ValueError(f"Existing seal {key} differs; preserve it and create a new explicitly reviewed study")
        return previous
    atomic_json(target, result)
    return result


def write_report(path, manifest, audit, confirmation):
    lines = ["# GSM8K 数据来源与历史使用审计", "",
        f"数据固定为 `{DATASET}` / `main` / `{REVISION}`。题目哈希使用原始 question 的 UTF-8 字节，不做空白归一化。",
        "", "## 固定划分", "", "|集合|train 半开区间|样本数|", "|---|---|---:|"]
    for name, (lo, hi) in RANGES.items():
        lines.append(f"|{name}|[{lo},{hi})|{hi-lo}|")
    lines += ["", "上述集合对本次拟合互斥；C_diag/D_select 不能宣称历史未使用。",
        "", "## 原始文件与校验", ""]
    for split, source in manifest["sources"].items():
        lines += [f"- {split}: {source['rows']} 条；parquet SHA256 `{source['parquet_sha256']}`；JSONL SHA256 `{source['jsonl_sha256']}`。",
                  f"  [固定 revision 原始文件]({source['url']})；本地 `{source['jsonl_path']}`。"]
    lines += ["", "## 已保存的历史证据", "",
        f"审计当前 checkout 中 {audit['files_scanned']} 份 JSON/JSONL，包含逐样本生成、逐位置 probe、分组 raw、汇总和 manifest。逐文件 SHA256、schema、ID、来源判定与错误见 history_audit.json。",
        "", "|历史目录|观测 ID（按保存的 split/源码语义映射）|声明范围|原始样本记录|", "|---|---|---|---:|"]
    for name, stats in audit["by_run"].items():
        if not stats["sample_records"] and not stats["declared_ids"]:
            continue
        def show(field):
            return "; ".join(f"{s}: {v['count']} / {v['ranges_half_open']}" for s,v in stats[field].items()) or "未保存"
        lines.append(f"|{name}|{show('observed_ids')}|{show('declared_ids')}|{stats['sample_records']}|")
    lines += ["", "记录数包含不同方法及同一题的多位置记录，不能当独立样本量。声明范围与实际观测分列；声明不是成功运行证明。旧数据缺少 revision/题目哈希，因此这些映射属于有来源依据的 ID 推断。gold 一致也不能单独确认题目身份。",
        "", "## 确认候选封存与不确定性", "",
        f"已封存 train[1024,2048) 的 {confirmation['candidate_count']} 个候选题目哈希；排除 {len(confirmation['excluded_observed_or_reserved'])} 个已观测/已声明/与核心划分重题的候选。候选列表哈希 `{confirmation['candidate_hash']}`。",
        "", "**INDEPENDENCE_UNVERIFIED**：这只证明对当前可见记录的排除和选配置前的候选固定，不能证明历史从未使用。旧校准参数部分未保存拟合 ID/count；未恢复所有 Git 历史、外部或已删除日志。完整 test 已有历史使用，不可重新称为全新确认集。",
        "", "候选池不是正式 P6 的最终样本量。解封前还须冻结比较、主要指标、质量约束与功效/精度计划。不得根据候选结果换题。",
        "", "未解决记录："]
    for item in audit["unresolved"]:
        lines.append(f"- `{item['file']}`: {item['reason']}" + (f" ({item['records']} 条)" if "records" in item else ""))
    lines += [f"- JSON 解析错误：{len(audit['parse_failures'])}。", "",
        "## 重现命令", "", "```bash",
        "/home/jgu7/work/e2e-runtime/venv/bin/python scripts/audit_state_calibration_data.py \\",
        "  --repo /home/jgu7/work/MOEresearch --out /scratch/jgu7/moe-state-study/data",
        "```", "", "脚本不加载模型；下载、parquet、导出、manifest 和封存文件均在 scratch。已有不同内容的导出或封存会报错，不能静默覆盖。"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=Path("/scratch/jgu7/moe-state-study/data"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    if not str(out).startswith("/scratch/jgu7/"):
        parser.error("Dataset/cache outputs must be under /scratch/jgu7/")
    out.mkdir(parents=True, exist_ok=True)
    rows, manifest = prepare_dataset(out)
    audit, observed, declared, hashes = audit_history(args.repo.resolve(), rows)
    confirmation = seal(out, rows, audit, observed, declared, hashes)
    sealed_audit_path = out / "history_audit.sealed.json"
    if not sealed_audit_path.exists():
        prior_path = out / "history_audit.json"
        sealed_audit = json.loads(prior_path.read_text()) if prior_path.exists() else audit
        if sha(canonical(sealed_audit["files"])) != confirmation["historical_audit_inventory_sha256"]:
            raise ValueError("Original sealed inventory is unavailable; refusing to erase its provenance")
        atomic_json(sealed_audit_path, sealed_audit)
    manifest.update({"created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip(),
        "fixed_train_ranges": RANGES, "confirmation_seal": str(out / "confirmation_seal.json"),
        "confirmation_candidate_hash": confirmation["candidate_hash"],
        "sealed_history_audit_path": str(sealed_audit_path),
        "latest_history_audit_sha256": sha(canonical(audit)),
        "historical_independence": "INDEPENDENCE_UNVERIFIED",
        "script_sha256": sha(Path(__file__).read_bytes())})
    atomic_json(out / "data_manifest.json", manifest)
    atomic_json(out / "history_audit.json", audit)
    report = args.report or args.repo / "docs/2026-10-19/history_audit.md"
    write_report(report, manifest, audit, confirmation)
    print(json.dumps({"revision": REVISION, "train": manifest["sources"]["train"],
        "files_scanned": audit["files_scanned"], "parse_failures": len(audit["parse_failures"]),
        "confirmation_count": confirmation["candidate_count"], "report": str(report)}, indent=2))


if __name__ == "__main__":
    main()
