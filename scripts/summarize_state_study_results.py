#!/usr/bin/env python3
"""Read-only scientific report for one frozen state-study run.

The run directory is evidence, never an output target. Descriptive generation
statistics are rebuilt from physical JSONL records. Existing bootstrap results
are quoted only after checking the corresponding raw requests. No model, tensor
package, experiment module, parser revision, or historical run is imported.
NumPy is loaded lazily only for final archive metadata/position verification;
independent prospective statistics have their own hash-bound cached endpoint.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shlex
import socket
import statistics
import sys


ARMS = ("N", "F", "R8", "R16", "M_N", "L_N", "M_NT", "M_ST")
BASELINES = ARMS[:6]
FAILED = {"oom", "error", "timeout"}
IDENTITY = ("question_hash", "protocol_hash", "parser_version", "model_revision",
            "model_hash", "dtype", "max_new_tokens", "rendered_prompt", "prompt_token_ids")
QUESTIONS = (
    "实际跑的是哪一份模型、什么精度、什么实现版本？是否保持原项目科学设定？",
    "修改过哪些关键实现？哪些正确性检查通过，哪些仍未通过？",
    "六个已有方法结果如何？最强已有方法是否已经解决主要问题？",
    "原生校准到了 student 状态上有没有再失准？C−D、B−A 与长后缀说明什么？",
    "M_NT/M_ST 都重新运行后，M_ST 是否仍有额外收益？",
    "相同文字的 native-state 对照、总校准预算与强基线能否解释收益？",
    "答案质量、生成长度、自然结束与实际成本分别如何变化？",
    "当前结论属于 POSITIVE_EXPLORATORY、NEGATIVE、INCONCLUSIVE 还是 BLOCKED？",
    "下一步最值得做的一件事是什么？精确命令和依据是什么？",
)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    lower = int(index)
    return values[lower] + (values[min(lower + 1, len(values)-1)] - values[lower]) * (index-lower)


def distribution(values):
    values = [float(value) for value in values if finite(value)]
    return {"n": len(values), "mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "p90": percentile(values, .9), "sum": sum(values) if values else None,
            "max": max(values) if values else None}


def number(value, percent=False):
    if not finite(value):
        return "PENDING"
    return f"{100*value:.2f}%" if percent else f"{value:.4g}"


def link(path):
    path = Path(path)
    return f"[{path.name}](<{path}>)"


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def sample_sort(row):
    value = row.get("sample_id")
    return str(type(value)), value if isinstance(value, (int,str)) else repr(value)


def run_is_active(run):
    """Auxiliary lock observation only; never establishes process liveness."""
    lock = Path(run)/"study.lock"
    if not lock.is_file():
        return None
    try:
        with lock.open("rb") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(handle, fcntl.LOCK_UN)
        return False
    except OSError:
        return None


def _proc_stat(value):
    # The comm field is parenthesized and may contain spaces or parentheses.
    close = value.rfind(")")
    if close < 0:
        raise ValueError("invalid /proc/PID/stat comm field")
    fields = value[close+1:].split()
    if len(fields) < 20:
        raise ValueError("short /proc/PID/stat record")
    return {"pid":int(value[:value.find("(")].strip()), "state":fields[0],
            "start_ticks":str(int(fields[19]))}


def verify_launch_process(run, launch, *, proc_root=Path("/proc"), hostname=None):
    """Prove one launch PID live locally; absence or remote state stays UNKNOWN.

    Exact boot/start identity is checked before reading cmdline, then stat is
    read again to avoid reporting a PID that exited or changed during the read.
    A lock, missing /proc entry, scheduler allocation, or old PID alone cannot
    certify that the experiment process is live or that its work is terminal.
    """
    hostname = socket.gethostname() if hostname is None else hostname
    result = {"status":"UNKNOWN", "checked_at_utc":datetime.now(timezone.utc).isoformat(),
        "local_hostname":hostname, "expected_hostname":launch.get("hostname"),
        "pid":launch.get("pid"), "expected_proc_start_ticks":launch.get("proc_start_ticks"),
        "expected_boot_id":launch.get("boot_id"), "slurm_job_id":launch.get("slurm_job_id"),
        "checks":{}, "terminal_state_established":False}
    expected_host = launch.get("hostname")
    if not expected_host:
        return {**result,"reason":"launch_hostname_missing"}
    same_host = str(expected_host).lower().rstrip(".") == str(hostname).lower().rstrip(".")
    result["checks"]["same_hostname"] = same_host
    if not same_host:
        return {**result,"reason":"different_host_local_proc_cannot_verify_launch_pid"}
    pid = launch.get("pid")
    if isinstance(pid,bool) or not isinstance(pid,int) or pid <= 0:
        return {**result,"reason":"invalid_or_missing_launch_pid"}
    if not launch.get("boot_id") or launch.get("proc_start_ticks") is None:
        return {**result,"reason":"launch_boot_or_start_identity_missing"}
    try:
        boot = (Path(proc_root)/"sys/kernel/random/boot_id").read_text().strip()
        result["observed_boot_id"] = boot
        result["checks"]["boot_id_matches"] = boot == launch["boot_id"]
        if not result["checks"]["boot_id_matches"]:
            return {**result,"reason":"boot_id_mismatch_original_launch_not_verified"}
        directory = Path(proc_root)/str(pid)
        before_raw = (directory/"stat").read_text()
        before = _proc_stat(before_raw)
        result["observed_start_ticks"] = before["start_ticks"]
        result["observed_process_state"] = before["state"]
        result["checks"]["pid_and_start_ticks_match"] = before["pid"] == pid and before["start_ticks"] == str(int(launch["proc_start_ticks"]))
        if not result["checks"]["pid_and_start_ticks_match"]:
            return {**result,"reason":"pid_start_identity_mismatch_possible_pid_reuse"}
        result["checks"]["non_zombie"] = before["state"] not in ("Z","X","x")
        if not result["checks"]["non_zombie"]:
            return {**result,"reason":"launch_pid_is_zombie_or_dead_not_live_no_scientific_terminal_claim"}
        command_raw = (directory/"cmdline").read_bytes()
        command = [part.decode("utf-8",errors="replace") for part in command_raw.rstrip(b"\0").split(b"\0") if part]
        result["observed_cmdline"] = command
        run_matches = False
        for index,argument in enumerate(command):
            value = command[index+1] if argument == "--run-dir" and index+1 < len(command) else argument.partition("=")[2] if argument.startswith("--run-dir=") else None
            if value and Path(value).is_absolute() and Path(value).resolve() == Path(run).resolve():
                run_matches = True
        result["checks"]["cmdline_run_dir_matches"] = run_matches
        expected_command = launch.get("command")
        result["checks"]["cmdline_matches_launch_command"] = command == expected_command if isinstance(expected_command,list) else None
        if not run_matches or result["checks"]["cmdline_matches_launch_command"] is False:
            return {**result,"reason":"cmdline_does_not_match_launch_run_identity"}
        after_raw = (directory/"stat").read_text()
        after = _proc_stat(after_raw)
        unchanged = after["pid"] == pid and after["start_ticks"] == before["start_ticks"] and after["state"] not in ("Z","X","x")
        result["checks"]["identity_and_non_zombie_rechecked"] = unchanged
        result["proc_stat_snapshot_sha256"] = hashlib.sha256(after_raw.encode()).hexdigest()
        result["cmdline_snapshot_sha256"] = hashlib.sha256(command_raw).hexdigest()
        if not unchanged:
            return {**result,"reason":"process_changed_or_exited_during_verification"}
    except (OSError,ValueError) as exc:
        return {**result,"reason":"local_proc_unreadable_or_identity_invalid", "read_error":str(exc)}
    return {**result,"status":"VERIFIED_LIVE", "reason":"same_host_boot_pid_start_run_cmdline_non_zombie_verified"}


class Evidence:
    def __init__(self, run_dir, verify_diagnostic_hashes=False):
        self.run = Path(run_dir).resolve()
        self.verify_diagnostic_hashes = verify_diagnostic_hashes
        self.inventory = {}
        self.errors = []
        self.warnings = []

    def path(self, value):
        path = Path(value)
        path = path if path.is_absolute() else self.run / path
        path = path.resolve()
        if not path.is_relative_to(self.run):
            raise ValueError(f"Evidence path escapes this run: {path}")
        return path

    def record(self, path, raw=None):
        path = self.path(path)
        if raw is None:
            checksum = hashlib.sha256()
            physical_lines = 0
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024*1024), b""):
                    checksum.update(block)
                    if path.suffix==".jsonl":physical_lines+=block.count(b"\n")
            sha = checksum.hexdigest()
        else:
            sha = hashlib.sha256(raw).hexdigest()
        self.inventory[str(path.relative_to(self.run))] = {
            "path": str(path), "bytes": len(raw) if raw is not None else path.stat().st_size,
            "sha256": sha}
        if raw is None and path.suffix==".jsonl":self.inventory[str(path.relative_to(self.run))]["physical_lines"]=physical_lines
        return sha

    def json(self, value):
        path = self.path(value)
        if not path.is_file():
            return None
        try:
            raw = path.read_bytes()
            self.record(path, raw)
            return json.loads(raw, parse_constant=_reject_constant)
        except (OSError, ValueError) as exc:
            self.errors.append(f"{path}: {exc}")
            return None

    def jsonl(self, value):
        """Read a bounded snapshot using LF records, preserving U+2028 in strings."""
        path = self.path(value)
        if not path.is_file():
            return []
        rows, checksum, size = [], hashlib.sha256(), 0
        before = path.stat()
        try:
            with path.open("rb") as handle:
                remaining = before.st_size
                line = 0
                while remaining:
                    raw = handle.readline(remaining)
                    if not raw:
                        break
                    remaining -= len(raw)
                    size += len(raw)
                    line += 1
                    checksum.update(raw)
                    if not raw.endswith(b"\n"):
                        self.warnings.append(f"{path}:{line}: 未提交的末行；本快照不计入该行")
                        break
                    if not raw.strip():
                        self.errors.append(f"{path}:{line}: 空 JSONL 记录")
                        continue
                    try:
                        row = json.loads(raw, parse_constant=_reject_constant)
                        if not isinstance(row, dict):
                            raise ValueError("JSONL record is not an object")
                        rows.append(row)
                    except (ValueError, UnicodeError) as exc:
                        self.errors.append(f"{path}:{line}: {exc}")
            after = path.stat()
            if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
                self.warnings.append(f"{path}: 汇总期间仍有写入；表格是读取时的前缀快照")
            self.inventory[str(path.relative_to(self.run))] = {
                "path": str(path), "bytes": size, "sha256": checksum.hexdigest(),
                "snapshot_bytes": before.st_size, "complete_records": len(rows)}
        except OSError as exc:
            self.errors.append(f"{path}: {exc}")
        return rows


def describe(rows, expected_ids, evidence, path, protocol_hash, arm):
    ids, unique = [], {}
    for row in rows:
        sid = row.get("sample_id")
        if not isinstance(sid, int) or isinstance(sid, bool):
            evidence.errors.append(f"{path}: 非整数 sample_id {sid!r}")
            continue
        ids.append(sid)
        if sid in unique:
            evidence.errors.append(f"{path}: 重复 sample_id={sid}，不允许正式配对")
        unique[sid] = row
        if row.get("arm") != arm or not protocol_hash or row.get("protocol_hash") != protocol_hash:
            evidence.errors.append(f"{path}: sample_id={sid} 的 arm/protocol 身份不一致")
        question = row.get("question")
        if not isinstance(question, str) or row.get("question_hash") != hashlib.sha256(question.encode()).hexdigest():
            evidence.errors.append(f"{path}: sample_id={sid} 的问题 hash 不一致")
        tokens = row.get("gen_token_ids")
        if not isinstance(tokens, list) or row.get("generated_tokens") != len(tokens):
            evidence.errors.append(f"{path}: sample_id={sid} 的生成 token 计数不一致")
        elif tokens:
            ended = tokens[-1] in row.get("eos_ids", [])
            if any(token in row.get("eos_ids", []) for token in tokens[:-1]):
                evidence.errors.append(f"{path}: sample_id={sid} 在 EOS 后还有 token")
            if (row.get("stop_reason") == "eos") != ended:
                evidence.errors.append(f"{path}: sample_id={sid} 的 EOS/stop_reason 不一致")
            if row.get("stop_reason") == "max_new_tokens" and len(tokens) != row.get("max_new_tokens"):
                evidence.errors.append(f"{path}: sample_id={sid} 的行政截断位置不一致")
    conflicts = [field for field in ("protocol_hash", "parser_version", "model_revision", "arm_hash", "max_new_tokens")
                 if len({json.dumps(row.get(field), sort_keys=True) for row in rows}) > 1]
    if conflicts:
        evidence.errors.append(f"{path}: 单臂混合身份字段 {conflicts}")
    reasons = Counter(row.get("stop_reason", "unknown") for row in rows)
    n = len(rows)
    def accuracy(field):
        values = [0.0 if row.get("stop_reason") in FAILED else float(row[field])
                  for row in rows if row.get("stop_reason") in FAILED or isinstance(row.get(field), bool)]
        return {"n": len(values), "value": statistics.fmean(values) if values else None}
    measured = [row for row in rows if row.get("observer_enabled") is False and row.get("timing_debug_logits") is False]
    disagreements = [row for row in rows if row.get("parser_disagreement") or
                     (isinstance(row.get("strict_correct"), bool) and isinstance(row.get("tolerant_correct"), bool)
                      and row["strict_correct"] != row["tolerant_correct"])]
    expected = set(expected_ids)
    return {"n": n, "n_expected": len(expected), "sample_ids": sorted(set(ids)),
            "pairing_identity_hash": canonical([
                {"sample_id":row.get("sample_id"), **{field:row[field] for field in IDENTITY if field in row}}
                for row in sorted(rows,key=sample_sort)]),
            "missing_ids": sorted(expected-set(ids)), "unexpected_ids": sorted(set(ids)-expected),
            "complete": bool(rows) and len(ids) == len(set(ids)) and set(ids) == expected,
            "strict": accuracy("strict_correct"), "tolerant": accuracy("tolerant_correct"),
            "stop_reason_counts": dict(reasons), "eos_rate": reasons["eos"]/n if n else None,
            "cap_rate": reasons["max_new_tokens"]/n if n else None,
            "length": distribution([row.get("generated_tokens") for row in rows]),
            "generation_seconds": distribution([row.get("generation_seconds") for row in measured]),
            "prefill_seconds": distribution([row.get("prefill_seconds") for row in measured]),
            "decode_seconds": distribution([row.get("decode_seconds") for row in measured]),
            "peak_allocated_bytes": distribution([row.get("peak_allocated_bytes") for row in measured]),
            "peak_reserved_bytes": distribution([row.get("peak_reserved_bytes") for row in measured]),
            "expert_token_assignments_simplified": distribution([row.get("expert_token_assignments_simplified") for row in rows]),
            "repeat_4gram": distribution([row.get("repeated_4gram_fraction") for row in rows]),
            "marker_rate": sum(row.get("has_answer_marker") is True for row in rows)/n if n else None,
            "post_marker_tokens": distribution([row.get("post_marker_tokens") for row in rows]),
            "parser_disagreements": len(disagreements), "max_new_tokens": sorted({row.get("max_new_tokens") for row in rows if finite(row.get("max_new_tokens"))}),
            "raw_path": str(path), "timing_eligible_rows": len(measured)}


def cache_matches(cache, stats, evidence, path):
    if not cache:
        return False
    cached = cache.get("arms", {})
    if set(cached) != set(stats):
        evidence.warnings.append(f"{path}: 摘要中的 arm 集合已过时，不引用其配对区间")
        return False
    for arm, measured in stats.items():
        old = cached[arm]
        fields = ((old.get("n_recorded"), measured["n"]),
                  (old.get("sample_ids"), measured["sample_ids"]),
                  (old.get("pairing_identity_hash"), measured["pairing_identity_hash"]),
                  (old.get("strict_accuracy"), measured["strict"]["value"]),
                  (old.get("tolerant_accuracy"), measured["tolerant"]["value"]),
                  (old.get("length", {}).get("mean"), measured["length"]["mean"]),
                  (old.get("stop_reason_counts"), measured["stop_reason_counts"]))
        if any(not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12) if finite(left) and finite(right)
               else left != right for left, right in fields):
            evidence.warnings.append(f"{path}: {arm} 缓存摘要与实际行不匹配，不引用其配对区间")
            return False
    return True


def load_generation(evidence, scope, expected_ids, protocol_hash, expected_arms=()):
    directory = evidence.run/scope/"generations"
    stats, rows, comparisons = {}, {}, {}
    paths = sorted(directory.glob("*.jsonl")) if directory.is_dir() else []
    for path in paths:
        arm = path.stem
        records = evidence.jsonl(path)
        rows[arm] = records
        stats[arm] = describe(records, expected_ids, evidence, path, protocol_hash, arm)
    summary_path = evidence.run/scope/"summaries/generation_summary.json"
    cache = evidence.json(summary_path)
    cache_valid = cache_matches(cache, stats, evidence, summary_path)
    if cache_valid:
        for name, comparison in cache.get("comparisons", {}).items():
            if "_minus_" not in name:
                continue
            left, right = name.split("_minus_", 1)
            ls, rs = stats.get(left), stats.get(right)
            if not ls or not rs or not ls["complete"] or not rs["complete"]:
                continue
            lm, rm = {r["sample_id"]: r for r in rows[left]}, {r["sample_id"]: r for r in rows[right]}
            if any(lm[sid].get(field) != rm[sid].get(field) for sid in lm for field in IDENTITY):
                evidence.errors.append(f"{scope}/{name}: 配对请求身份不一致")
                continue
            if comparison.get("formal_valid") is True and comparison.get("n_pairs") == len(expected_ids):
                comparisons[name] = comparison
    examples = []
    for arm in ARMS:
        for row in rows.get(arm, []):
            if row.get("parser_disagreement") or row.get("strict_correct") != row.get("tolerant_correct"):
                examples.append({"scope": scope, **{field: row.get(field) for field in
                    ("sample_id", "arm", "question_hash", "gold", "pred_strict", "pred_tolerant",
                     "strict_correct", "tolerant_correct", "strict_parse_status", "tolerant_parse_status", "stop_reason", "raw_output")}})
    examples.sort(key=lambda row: (row.get("strict_correct") == row.get("tolerant_correct"), str(row["sample_id"]), row["arm"]))
    return {"scope": scope, "arms": stats, "expected_ids": expected_ids,
            "expected_arms": list(expected_arms), "summary_cache_validated": cache_valid,
            "complete": bool(expected_arms) and all(stats.get(arm, {}).get("complete", False) for arm in expected_arms),
            "comparisons": comparisons, "parser_examples": examples[:20]}, rows


def load_diagnostic(evidence, scope):
    path = evidence.run/scope/"summaries/diagnostic_summary.json"
    summary = evidence.json(path)
    if summary is None:
        return {"scope": scope, "status": "PENDING", "summary_path": str(path), "summary": None}
    directory = evidence.run/scope/"diagnostics"
    listed = set()
    valid = bool(summary.get("source_archives"))
    for item in summary.get("source_archives", []):
        archive = evidence.path(item["path"])
        listed.add(archive)
        if not archive.is_file() or archive.stat().st_size != item.get("size_bytes"):
            evidence.warnings.append(f"{path}: 诊断来源缺失或大小变化：{archive}")
            valid = False
    current = set(directory.glob("*.npz")) | set(directory.glob("*.complete.json")) | set(directory.glob("*.empty.json"))
    if current != listed:
        evidence.warnings.append(f"{path}: 诊断归档清单变化，需重建诊断摘要")
        valid = False
    for commit_path in sorted(directory.glob("*.complete.json")):
        commit = evidence.json(commit_path)
        if not commit:
            valid = False
            continue
        for field, hash_field in (("archive", "archive_sha256"), ("tokens", "tokens_sha256")):
            filename = commit.get(field, "")
            if not filename or Path(filename).name != filename:
                evidence.errors.append(f"{commit_path}: 无效 commit 文件名")
                valid = False
                continue
            source = directory/filename
            if not source.is_file():
                evidence.errors.append(f"{commit_path}: 已提交证据缺失 {filename}")
                valid = False
            elif evidence.verify_diagnostic_hashes and evidence.record(source) != commit.get(hash_field):
                evidence.errors.append(f"{commit_path}: 证据 SHA256 不一致 {filename}")
                valid = False
    if summary.get("incomplete_uncommitted_archives"):
        valid = False
    return {"scope": scope, "status": summary.get("status", "UNKNOWN"), "summary_path": str(path),
            "summary": summary, "current_inventory_matches": valid,
            "archive_hashes_verified": valid and evidence.verify_diagnostic_hashes}


def state_metric(diagnostic, path=None, contrast=None, weighting="token"):
    if diagnostic.get("summary") and not diagnostic.get("current_inventory_matches"):
        return None
    summary = diagnostic.get("summary") or {}
    block = summary.get("analyses", {}).get("all", {}).get(weighting, {})
    metric = summary.get("chosen_mismatch_metric", "E_moment")
    if contrast:
        row = block.get("comparisons", {}).get(contrast, {})
        return row.get("metrics", {}).get(metric) if row.get("status") == "ok" else None
    return block.get("paths", {}).get(path, {}).get("intervals", {}).get(metric)


def reconcile_state_masks(diagnostic, raw, ids, original_scope, updated_scope):
    """Bind saved all-position coverage to the corresponding free-generation text."""
    coverage = (diagnostic.get("summary") or {}).get("coverage", {}).get("all", {})
    verified = {}
    for path in coverage:
        if path in ("A", "B", "C", "D"):
            arm = "N" if path in ("A", "B") else "L_N"
            rows = raw.get(original_scope, {}).get(arm, [])
        elif path in ("M_NT", "M_ST"):
            rows = raw.get(updated_scope, {}).get(path, [])
        else:
            continue
        traces = {str(row["sample_id"]):row.get("gen_token_ids",[]) for row in rows}
        counts = coverage[path].get("counts_per_prompt_layer", {})
        masks = coverage[path].get("mask_hashes_per_prompt_layer", {})
        empty = set(coverage[path].get("zero_observation_prompt_ids", []))
        valid = set(traces) == {str(i) for i in ids} and (set(counts)|empty) == set(traces)
        for pid, trace in traces.items():
            if len(trace) == 1:
                valid = valid and pid in empty and pid not in counts
            else:
                expected_mask = canonical([[step,token] for step,token in enumerate(trace[:-1],1)])
                valid = valid and set(counts.get(pid,{})) == {str(layer) for layer in range(48)}
                valid = valid and set(masks.get(pid,{})) == {str(layer) for layer in range(48)}
                valid = valid and all(value == len(trace)-1 for value in counts.get(pid,{}).values())
                valid = valid and all(value == expected_mask for value in masks.get(pid,{}).values())
        verified[path] = bool(valid)
    diagnostic["full_trace_masks_reconciled"] = verified


def interval_text(value, generation=False):
    if not value:
        return "PENDING / 不可估计"
    estimate = value.get("mean") if generation else value.get("estimate")
    ci = value.get("ci95") if generation else [value.get("ci_low"), value.get("ci_high")]
    if not finite(estimate) or not ci or not all(finite(x) for x in ci):
        return "不可估计；" + str(value.get("status", "缺少有限区间"))
    return f"{number(estimate)} [{number(ci[0])}, {number(ci[1])}]"


def load_calibrations(evidence, protocol_hash):
    records = {}
    for path in sorted((evidence.run/"formal/calibration").glob("*.json")):
        if path.stem not in ("M_N", "L_N", "M_NT", "M_ST"):
            continue
        value = evidence.json(path)
        if not value:
            continue
        body = {key: item for key, item in value.items() if key != "calibration_hash"}
        if canonical(body) != value.get("calibration_hash"):
            evidence.errors.append(f"{path}: calibration hash 不一致")
        metadata = value.get("metadata", {})
        if metadata.get("protocol_hash") != protocol_hash:
            evidence.errors.append(f"{path}: calibration protocol 身份不一致")
        records[path.stem] = {key: value.get(key) for key in
            ("family", "reduced_branch", "application_phase", "application_dtype", "statistics_dtype", "ddof", "operation_order",
             "numerical_protection", "calibration_hash", "metadata")}
        records[path.stem]["path"] = str(path)
    nt, st = records.get("M_NT"), records.get("M_ST")
    matched = None
    if nt and st:
        fields = ("family", "reduced_branch", "application_phase", "statistics_dtype", "ddof", "operation_order", "numerical_protection")
        meta_fields = ("fit_mask_hash", "prompt_ids", "fit_decode_token_count", "source_student_calibration_hash")
        matched = all(nt.get(key) == st.get(key) for key in fields) and all(
            nt["metadata"].get(key) is not None and nt["metadata"].get(key) == st["metadata"].get(key) for key in meta_fields)
        if not matched:
            evidence.errors.append("M_NT/M_ST 未通过同公式、同参数保护、同 prompt/token mask 身份核对")
    return {"arms": records, "nt_st_matched_text_formula_and_count": matched,
            "native_cost": evidence.json("formal/calibration/calibration_cost.json"),
            "updated_cost_attempts": evidence.jsonl("formal/calibration/updated_calibration_cost_attempts.jsonl")}


def collect(run_dir, verify_diagnostic_hashes=False,quality_plan_path=None,positive_cache_dir=None):
    evidence = Evidence(run_dir, verify_diagnostic_hashes)
    protocol = evidence.json("protocol.json") or {}
    manifest = evidence.json("manifest.json") or {}
    config = protocol.get("config", {})
    phash = protocol.get("protocol_hash")
    if protocol:
        if canonical({key: value for key, value in protocol.items() if key != "protocol_hash"}) != phash:
            evidence.errors.append("protocol.json 的自描述 hash 不一致")
        if manifest and manifest.get("protocol_hash") != phash:
            evidence.errors.append("manifest 与 protocol 身份不一致")
    freeze = evidence.json("validation_freeze.json") or {}
    launch = evidence.json("launch.json") or {}
    process_verification = verify_launch_process(evidence.run,launch)
    if freeze and freeze.get("protocol_hash") != phash:
        evidence.errors.append("validation_freeze 与 protocol 身份不一致")
    preflight = evidence.json("preflight.json") or evidence.json("preflight/preflight.json") or {}
    if preflight.get("protocol_hash") is not None and preflight["protocol_hash"] != phash:
        evidence.errors.append("preflight 与 protocol 身份不一致")
    generations, raw = {}, {}
    for scope, lo, hi, arms in (("smoke/select",256,260,BASELINES), ("pilot/select",256,288,BASELINES),
        ("formal/select",256,512,BASELINES), ("formal/validate",512,1024,freeze.get("finalist_arms", ())),
        ("formal/select_cap4096",256,384,freeze.get("finalist_arms", ())),
        ("formal/fit_native",0,128,("N",)), ("formal/fit_student",0,128,("L_N",)),
        ("formal/diag_rollouts",128,256,("N","L_N")),
        ("formal/updated_diag_rollouts",128,256,("M_NT","M_ST")),
        ("formal/validation_state/diag_rollouts",512,640,("N","L_N")),
        ("formal/validation_state/updated_diag_rollouts",512,640,("M_NT","M_ST"))):
        generations[scope], raw[scope] = load_generation(evidence, scope, list(range(lo,hi)), phash, arms)
    diagnostics = {scope: load_diagnostic(evidence, scope) for scope in ("smoke", "pilot", "formal", "formal/validation_state")}
    reconcile_state_masks(diagnostics["formal"],raw,list(range(128,256)),
        "formal/diag_rollouts","formal/updated_diag_rollouts")
    reconcile_state_masks(diagnostics["formal/validation_state"],raw,list(range(512,640)),
        "formal/validation_state/diag_rollouts","formal/validation_state/updated_diag_rollouts")
    calibration = load_calibrations(evidence, phash)
    failure_attempts = []
    for path in sorted(evidence.run.rglob("failures.jsonl")) if evidence.run.is_dir() else []:
        failure_attempts.extend({"evidence_path":str(path), **row} for row in evidence.jsonl(path))
    failed_gate = preflight.get("status") in ("FAILED_IMPLEMENTATION", "FAILED", "FAIL")
    blocked = bool(evidence.errors) or failed_gate or manifest.get("status") in ("error", "FAILED_IMPLEMENTATION")
    disposition = "BLOCKED" if blocked else "INCONCLUSIVE"
    reason = "实现/证据身份/运行错误尚待解决" if blocked else "证据链尚未全部完成，或尚未建立明确质量约束"
    formal, validation = diagnostics["formal"], diagnostics["formal/validation_state"]
    if not blocked and all(d.get("status") == "NEGATIVE" and d.get("archive_hashes_verified")
        and all(d.get("full_trace_masks_reconciled",{}).get(path) for path in ("A","B","C","D"))
        for d in (formal, validation)) and generations["formal/validate"]["complete"] and preflight.get("protocol_hash") == phash and preflight.get("status") in ("PASS","PASSED"):
        disposition, reason = "NEGATIVE", "已保存的正式诊断和冻结后状态复核均满足预声明实用界限；仅针对本设定内失准假设"
    if __package__:
        from .state_study_positive_evidence import evaluate_positive_evidence
    else:
        script_dir=str(Path(__file__).resolve().parent)
        if script_dir not in sys.path:sys.path.insert(0,script_dir)
        from state_study_positive_evidence import evaluate_positive_evidence
    plan_path=Path(quality_plan_path) if quality_plan_path else Path('/scratch/jgu7/moe-state-study/reports')/evidence.run.name/'quality_plan.json'
    positive=evaluate_positive_evidence({'protocol':protocol,'manifest':manifest,'preflight':preflight,
        'validation_freeze':freeze,'calibration':calibration,'generations':generations,'diagnostics':diagnostics},
        raw,evidence,plan_path,positive_cache_dir)
    if not blocked and positive.get('passed') is True:
        disposition,reason='POSITIVE_EXPLORATORY','依据验证前冻结的独立分析补充，完整状态/同文本重估/自身轨迹/质量优越性与观测请求成本证据链通过；非NI或独立确认'
    examples = [row for scope in ("formal/validate", "formal/select", "formal/select_cap4096")
                for row in generations[scope]["parser_examples"]][:20]
    for group in generations.values():
        group.pop("parser_examples", None)
    # Large P0 shadow tables stay at their evidence path, not duplicated in the
    # report. Keep every case name/sample and the recorded comparison endpoints.
    preflight = {**preflight, "cases":[{key:value for key,value in row.items()
        if key not in ("shadow_real_rows",)} for row in preflight.get("cases",[])]}
    manifest = {**manifest, "stages":{name:{**item,"detail":
        item.get("detail") if len(json.dumps(item.get("detail"),ensure_ascii=False)) < 2000 else
        {"summary":"完整 detail 保留在 manifest.json；此处省略大表", **{key:item["detail"][key]
         for key in ("status","reason","case_count","gate_recalibration") if isinstance(item.get("detail"),dict) and key in item["detail"]}}}
        for name,item in manifest.get("stages",{}).items()}}
    return {"schema_version":"state_study_independent_report.v1",
        "generated_at_utc":datetime.now(timezone.utc).isoformat(), "run_dir":str(evidence.run),
        "runner_lock_held":run_is_active(evidence.run),
        "launch_metadata":launch, "process_verification":process_verification,
        "protocol":protocol, "manifest":manifest, "preflight":preflight, "validation_freeze":freeze,
        "generations":generations, "diagnostics":diagnostics, "calibration":calibration,
        "failure_attempts":failure_attempts, "parser_disagreement_examples":examples,
        "disposition":disposition, "disposition_reason":reason, "errors":evidence.errors,
        "positive_evidence":positive,
        "warnings":evidence.warnings, "evidence_inventory":evidence.inventory,
        "analysis_contract":{"new_model_execution":False,"new_parameter_fitting":False,
            "generation_descriptives":"recomputed from recorded request fields; frozen parser is not rerun",
            "existing_confidence_intervals":"quoted from identity/point-matching saved summaries, not recomputed here; generation summaries lack a cryptographic binding to all row outcomes",
            "prospective_quality_and_relative_token_intervals":"Independent exact-binomial quality bounds and approximate paired-ratio bootstrap only after all structural prerequisites; cached by plan/source/actual inputs.",
            "diagnostic_hash_verification_requested":verify_diagnostic_hashes,
            "prospective_analysis_supplement":positive.get('analysis_supplement'),
            "original_NI_only_condition_not_satisfied_or_silently_rewritten":True,
            "positive_classification_basis":"Explicit prospective quality-superiority supplement, only when all verified requirements pass; not the original NI-only implementation text.",
            "historical_runs_included":False,"bootstrap_uncertainty":"conditional on frozen calibration; does not include refit uncertainty"}}


def render(report):
    run = Path(report["run_dir"])
    protocol, manifest = report["protocol"], report["manifest"]
    config, runtime = protocol.get("config", {}), manifest.get("runtime", {})
    groups, diagnostics = report["generations"], report["diagnostics"]
    formal, validation = diagnostics["formal"], diagnostics["formal/validation_state"]
    freeze, preflight = report["validation_freeze"], report["preflight"]
    scientific_rows = sum(groups[scope]["arms"].get(arm, {}).get("n",0) for scope in ("formal/select","formal/validate") for arm in ARMS)
    lines = ["# 状态来源校准：独立科学进展报告", "",
        f"读取时间：{report['generated_at_utc']}；证据目录：`{run}`。",
        f"当前总判断：**{report['disposition']}**。{report['disposition_reason']}。",
        f"正式选择/复核共读到 {scientific_rows} 条请求记录。零记录表示 PENDING，不能当作负结果。",
        f"具体启动进程核验：`{report['process_verification']['status']}`；PID `{report['process_verification'].get('pid')}`；核验时间 `{report['process_verification']['checked_at_utc']}`；原因 `{report['process_verification']['reason']}`。study.lock 观测 `{report.get('runner_lock_held')}` 仅作辅助，不代替 PID 身份核验。",
        "这是独立分析的完整九问报告，原始 run 路径如上；与 runner 的 STATUS/基础 RUN_REPORT 互补。生成描述统计重算，状态矩/区间引用保存摘要，不重新运行模型或重新拟合参数。新质量/比例成本端点仅在完整先决证据满足后按前瞻补充计算并缓存。",
        "先前 INCONCLUSIVE 表示当时缺少完整正式数据与可执行质量细则；不据已观察 D_validate 结果倒推门槛。原 protocol 的 NI-only 文字及 None 保留，原 NI 条件没有被假称满足。", ""]
    for index, question in enumerate(QUESTIONS, 1):
        lines += [f"## {index}. {question}", ""]
        if index == 1:
            lines += [f"冻结声明：模型 `{config.get('model_id','PENDING')}`，revision `{config.get('model_revision','PENDING')}`，精度 `{config.get('precision','PENDING')}`，attention `{config.get('attention_backend','PENDING')}`，batch `{config.get('batch_size','PENDING')}`。",
                f"实际运行记录：GPU `{runtime.get('gpu','未记录')}`；dtype `{runtime.get('dtype','未记录')}`；PyTorch `{runtime.get('torch','未记录')}`；Transformers `{runtime.get('transformers','未记录')}`；expert backend `{runtime.get('expert_backend','未记录')}`。",
                f"源码指纹 `{protocol.get('source_hash','PENDING')}`；协议 `{protocol.get('protocol_hash','PENDING')}`；数据 revision `{config.get('dataset_revision','PENDING')}`。",
                "只有实际 runtime/预检/逐样本证据支持已运行；配置声明本身不证明执行成功。旧 checkpoint 的本地权重摘要不可核实，旧结果未并入本轮配对比较。",
                f"证据：{link(run/'protocol.json')}、{link(run/'manifest.json')}、{link(run/'source_hashes.json')}。"]
        elif index == 2:
            cases = preflight.get("cases", [])
            names = sorted({row.get("name","unknown") for row in cases})
            lines += [f"实现记录：适配器 `{protocol.get('adapter','PENDING')}`；浮点顺序 `{protocol.get('floating_order','PENDING')}`。",
                "关键控制包括 compute/reference-k 分离、聚合后 FP32 scalar/逐维 affine、decode-only 应用、独立 cache 和相同文本的不同 driver 回放；这些是协议记录，是否通过由以下预检证据决定。",
                f"P0 保存状态 `{preflight.get('status','PENDING')}`；已保存 {len(cases)} 个检查实例；项目：{', '.join(names) or 'PENDING'}。"]
            lines += [f"smoke 六臂固定 4 题记录完整：`{groups['smoke/select']['complete']}`；pilot 六臂固定 32 题记录完整：`{groups['pilot/select']['complete']}`。工程通过不算正式机制证据。"]
            lines += [f"工程 pilot 当前共 {sum(x['n'] for x in groups['pilot/select']['arms'].values())} 条六臂请求记录；该工程样本不与正式 D_select 合并，不用于调整门槛。细节见独立 `pilot_generation_audit.md`，正式六臂表仍只用下节的正式数据。"]
            if preflight.get("error"):
                lines.append(f"预检错误：`{preflight['error'].get('message', preflight['error'])}`。")
            lines += [f"容差记录：`{json.dumps(preflight.get('tolerances',{}),ensure_ascii=False,sort_keys=True)}`。未运行检查不能计为通过。",
                f"证据：{link(run/'preflight.json')}、{link(run/'preflight/preflight.json')}、{link(run/'working_tree.patch')}。"]
        elif index == 3:
            lines += ["正式 D_select 的六个基线如下；不将 smoke/pilot 混入正式表。"]
            lines += generation_table(groups["formal/select"], BASELINES)
            if freeze:
                lines.append(f"D_select 冻结最佳 reference：`{freeze.get('selected_reference','PENDING')}`；最强已有基线：`{freeze.get('strongest_existing_baseline','PENDING')}`；初始校准器：`{freeze.get('initial_calibration','PENDING')}`。")
            else:
                lines.append("尚无 validation_freeze.json：不事后从 D_validate 另选最佳方法。")
            lines += ["是否已解决主要问题须同时查看质量、EOS/截断和完整请求成本；只胜过 scalar，或只缩短长度，均不足以建立新方法必要性。",
                f"证据：{link(run/'formal/select/generations')}、{link(run/'validation_freeze.json')}。"]
        elif index == 4:
            lines += [f"正式诊断保存状态 `{formal['status']}`；判据：`{json.dumps((formal.get('summary') or {}).get('decision',{}),ensure_ascii=False,sort_keys=True)}`。",
                "A=N on T_N；B=S0 on T_N；C=S0 on T_S；D=N on T_S。C−D 固定 student 文字，只改执行 driver；B−A 是 native 文字上的对应对照。C−D 不等于 C−A。",
                "",
                "|完整有效 decode 轨迹|token 加权估计 [95% CI]|prompt 等权敏感性|", "|---|---:|---:|"]
            for name in ("A","B","C","D"):
                lines.append(f"|{name}|{interval_text(state_metric(formal,path=name))}|{interval_text(state_metric(formal,path=name,weighting='prompt'))}|")
            for contrast in ("C_minus_D","B_minus_A"):
                lines.append(f"|{contrast}|{interval_text(state_metric(formal,contrast=contrast))}|{interval_text(state_metric(formal,contrast=contrast,weighting='prompt'))}|")
            coverage = (formal.get("summary") or {}).get("coverage", {})
            lines += ["", "逐位置覆盖（按路径列出有效位置总数；0 或缺项不表示匹配）：", "", "|位置组|A|B|C|D|", "|---|---:|---:|---:|---:|"]
            for group, paths in coverage.items():
                if group.startswith("common"):
                    continue
                lines.append("|"+group+"|"+"|".join(str(paths.get(name,{}).get("effective_position_count","PENDING")) for name in ("A","B","C","D"))+"|")
            lines += ["", "完整轨迹及长后缀与 common mask 分析分开。零 decode 样本没有状态证据；应连同各路径 total_recorded_prompt_count 与逐样本 trace 核对。未完成的后缀不能被前缀结果替代。",
                f"诊断清单与当前文件相符：`{formal.get('current_inventory_matches',False)}`；底层归档 SHA256 已复核：`{formal.get('archive_hashes_verified',False)}`。",
                f"完整 trace 的 prompt/48层/位置/token mask 对账：`{json.dumps(formal.get('full_trace_masks_reconciled',{}),ensure_ascii=False,sort_keys=True)}`。False/缺项表示正式覆盖尚未完整证明。",
                f"摘要的预期 prompt 总数：`{len((formal.get('summary') or {}).get('expected_prompt_ids') or []) or 'PENDING'}`；完整性：`{json.dumps((formal.get('summary') or {}).get('prompt_completeness',{}),ensure_ascii=False,sort_keys=True)}`。零 decode/该 bin 不活跃的 prompt 仍应进入 bootstrap 抽样全集，不虚构 activation；抽到零有效 token 的 draw 应明确不可定义而非丢弃。",
                f"证据：{link(formal['summary_path'])}、{link(run/'formal/diagnostics/replay_checks.jsonl')}、{link(run/'formal/diag_rollouts/generations')}。"]
        elif index == 5:
            for scope in ("formal/updated_diag_rollouts", "formal/validation_state/updated_diag_rollouts"):
                data = groups[scope]
                lines.append(f"`{scope}`："+"；".join(f"{arm} {data['arms'].get(arm,{}).get('n',0)}/{len(data['expected_ids'])}" for arm in ("M_NT","M_ST"))+"。")
            lines += [f"C_diag 自身轨迹 E(M_ST)−E(M_NT)：{interval_text(state_metric(formal,contrast='M_ST_minus_M_NT_own_trajectories'))}。",
                f"D_validate 固定状态复核子集自身轨迹差：{interval_text(state_metric(validation,contrast='M_ST_minus_M_NT_own_trajectories'))}。",
                "两个更新版本分别自由生成再回放自己的新 trace，才属于部署后自身状态检验。参数只在旧 S0 状态上变好不够；统计区间跨零也不证明等效。"]
            lines += comparison_lines(groups["formal/validate"], names=("M_ST_minus_M_NT",))
            lines += [f"证据：{link(run/'formal/updated_diag_rollouts/generations')}、{link(validation['summary_path'])}。"]
        elif index == 6:
            calibration = report["calibration"]
            lines += [f"NT/ST 的同公式、同数值保护、同 prompt/位置清单与拟合 token 数核对：`{calibration['nt_st_matched_text_formula_and_count']}`。None 表示尚未有两个参数文件。",
                "",
                "|校准器|family|拟合 decode tokens|来源 rollout tokens|mask hash|", "|---|---|---:|---:|---|"]
            for arm, data in calibration["arms"].items():
                metadata = data.get("metadata", {})
                lines.append(f"|{arm}|{data.get('family')}|{metadata.get('fit_decode_token_count','PENDING')}|{metadata.get('rollout_token_count','PENDING')}|`{metadata.get('fit_mask_hash','PENDING')}`|")
            lines += ["", "M_NT 控制更换为 student 生成文字后的收益；M_ST−M_NT 才区分同文本上的状态来源。原始 L_N 与 NT/ST 的拟合 token 数不必相同，不能称它们总预算相同。",
                "已记录校准成本（不是完整作业墙钟；失败前未落盘开销、热身、I/O、最终序列化等可能未包含）：",
                "", "```json", json.dumps({"native":calibration["native_cost"], "updated_attempts":calibration["updated_cost_attempts"]},ensure_ascii=False,indent=2), "```",
                "NT/ST 可共享 S0 rollout，不能把该 rollout 成本分别相加后又称共享；也不能从两者成本接近推断已做原生校准等总预算扩样对照。首轮未观察到该对照时，等总预算解释仍未排除。",
                f"证据：{link(run/'formal/calibration')}、{link(run/'formal/fit_native/generations/N.jsonl')}、{link(run/'formal/fit_student/generations/L_N.jsonl')}。"]
        elif index == 7:
            for scope in ("formal/validate", "formal/select_cap4096"):
                lines += [f"**{scope}**", ""] + generation_table(groups[scope])
                lines += cost_table(groups[scope])
                lines += comparison_lines(groups[scope])
            select = groups["formal/select"]["arms"]
            trigger = any(finite(data["cap_rate"]) and data["cap_rate"] > .01 for data in select.values())
            lines += [f"D_select 已记录样本中有臂行政截断 >1%：`{trigger}`。未完成选择集时该触发判断仍可能变化。触发后需所有入围臂在固定前 128 题、cap=4096 单独复验；2048 与 4096 不混合。",
                "cap 处恰好发出 EOS 计自然结束；cap 截断后的自然长度未知。长度下降但质量/作答率恶化不能算改进。Strict/tolerant 同时展示；方向相反时质量结论不确定，最后数字兜底不等于真实解题能力。",
                "时间只纳入显式 observer_enabled=false 且 timing_debug_logits=false 的行；同一 Python 研究后端实测不能直接称为 vLLM/SGLang 加速。专家预算是有标签的简化 assignment 计数，包含 native prefill 与 T−1 次 decode，不是实测总 FLOPs。",
                f"失败尝试记录 {len(report['failure_attempts'])} 条；重复重试与最终成功分开，不能把尝试数当独立请求数，也不能静默丢掉尚未补齐的配对。",
                f"已导出最多 {len(report['parser_disagreement_examples'])} 个正式解析分歧例子到 scientific_summary.json；不据此只修某一臂。"]
        elif index == 8:
            lines += [f"**{report['disposition']}**：{report['disposition_reason']}。",
                f"正式 P3 状态 `{formal['status']}`；冻结后状态分析 `{validation['status']}`。只有更新路径的验证摘要缺少 A/B/C/D 时，其 INCONCLUSIVE 不自动否定已估计的 NT/ST 自身轨迹差。",
                f"原质量非劣性界限：`{freeze.get('quality_noninferiority_margin','未冻结')}`；原 NI-only 条件仍未满足。独立前瞻质量计划状态：`{report['positive_evidence'].get('quality_plan_status','PENDING')}`；它是对实现窄化的分析补充/修正，依据原用户计划11.4允许明确质量优越性，不是默改原 protocol 或 validation_freeze。",
                "POSITIVE_EXPLORATORY 需要全部状态/重估/新自身轨迹/身份覆盖先决条件，加验证集512题上四项同时精确质量下界>0与两项名义同时配对ratio区间下界≥5%。成本 bootstrap 覆盖是近似的；不会由“不显著”推NI/等价。4096无逆向点估计要求属于新前瞻补充的敏感性一致性条件，不冒充原计划已有逐项否决。",
                f"正链要求逐项结果：`{json.dumps(report['positive_evidence'].get('requirements',{}),ensure_ascii=False,sort_keys=True)}`；状态/原因：`{report['positive_evidence'].get('status')}` / `{report['positive_evidence'].get('reason')}`。",
                "NEGATIVE 仅在正式与冻结后复核都记录满足预声明实用界限、数据完整且显式复核底层归档 SHA256 时输出；其作用域限于本设定。诊断支持状态效应仍不等于完整中介、下游收益或独立确认。",
                "历史独立性：C_diag/D_select/D_validate 仅对本次拟合留出。候选确认池的历史独立性未核实；P6 未解封不称通过确认，P7 未运行不声称跨模型/任务。",
                "固定校准参数的 prompt bootstrap 不含校准重拟合不确定性；没有三组校准复验时，该限制仍存在。"]
            lines += positive_gate_tables(report['positive_evidence'])
        elif index == 9:
            command_path = run/"commands.sh"
            process = report["process_verification"]
            if process.get("status") == "VERIFIED_LIVE":
                command = shlex.join([sys.executable,"-m","json.tool",str(run/"manifest.json")])
                lines += ["同主机上的具体 launch PID 已核对 boot ID、启动 ticks、run 命令行及非 zombie 状态，当前 VERIFIED_LIVE：保持这次冻结 auto 继续执行，不另启续跑进程。下一步只读查看持久化进度；此证明只对应读取时刻，不保证未来持续存活。", "精确只读检查命令："]
            elif process.get("status") == "UNKNOWN":
                job = report["launch_metadata"].get("slurm_job_id")
                inspect = run/"launch.json" if (run/"launch.json").is_file() else run/"manifest.json"
                command = shlex.join(["scontrol","show","job",str(job)]) if str(job).isdigit() else shlex.join([sys.executable,"-m","json.tool",str(inspect)])
                lines += ["启动进程状态 UNKNOWN：跨主机、读不到 /proc、PID 身份不符或 zombie 均不能在本报告里直接宣布整个实验 terminal。先在记录的启动主机核验具体进程，必要时只读核对对应 Slurm 作业；作业仍在分配也不等于该 PID 存活。未核实前不重复启动。", "补充核验命令："]
            elif report["disposition"] == "BLOCKED":
                inspect = run/"preflight/preflight.json" if preflight else run/"manifest.json"
                command = shlex.join([sys.executable,"-m","json.tool",str(inspect)])
                lines += ["先定位并修复已记录的唯一优先阻塞，保留当前 run；修改冻结源码后必须新 run 做 P0，不能在此目录混合协议。只读检查命令："]
            elif not groups["formal/validate"]["complete"]:
                command = "sh "+shlex.quote(str(command_path)) if command_path.is_file() else "test -f "+shlex.quote(str(run/"protocol.json"))
                lines += ["优先按既有冻结协议补齐 P0–P5 的下一未完成阶段；不因中间效果方向改样本或阈值。" if command_path.is_file() else "尚无可核验 commands.sh；先建立冻结协议与启动命令。", "精确命令："]
            else:
                command = shlex.join([sys.executable,str(Path(__file__).resolve()),"--run-dir",str(run),"--verify-diagnostic-hashes"])
                lines += ["优先做完整证据复核，并据已冻结比较报告连续效应和质量/成本；当前证据不足时保留 INCONCLUSIVE，先制定新的前瞻质量计划再考虑 P6。只读复核命令："]
            lines += ["", "```sh", command, "```", "", "阶段状态记录（来自 manifest，不替代原始数据完整性）：", ""]
            for name, item in manifest.get("stages", {}).items():
                lines.append(f"- `{name}`：`{item.get('status','UNKNOWN')}`；{json.dumps(item.get('detail'),ensure_ascii=False)}")
            lines += ["", "尚未满足或需明确保留的事项：", ""]
            for name, satisfied in (("P0 实际保存通过",preflight.get("status") in ("PASS","PASSED")),
                ("六基线 D_select 全部固定题目",groups["formal/select"]["complete"]),
                ("正式 A/B/C/D 分析已生成",bool(formal.get("summary"))),
                ("D_validate 入围方法固定题目完整",groups["formal/validate"]["complete"]),
                ("独立前瞻质量优越性计划身份已核实",report['positive_evidence'].get('quality_plan_status')=='VERIFIED_PROSPECTIVE_PLAN')):
                lines.append(f"- {name}：{'已观察到所需记录' if satisfied else 'PENDING / 尚未满足'}。")
            lines += ["- P6 独立确认、P7 跨设置与校准样本重复：本脚本不从 P0–P5 外推完成。"]
        lines += [""]
    lines += ["## 数据完整性与报告边界", ""]
    if report["errors"]:
        lines += ["以下身份/格式错误阻止正式比较：", ""] + ["- "+item for item in report["errors"]]
    if report["warnings"]:
        lines += ["", "快照/缓存提示：", ""] + ["- "+item for item in report["warnings"]]
    if not report["errors"] and not report["warnings"]:
        lines.append("在本次读取范围内未发现身份/格式错误；这不替代模型正确性预检或底层诊断重算。")
    lines += ["", "输入路径、字节数和 SHA256 见 scientific_summary.json 的 evidence_inventory。launch.json 与 process_verification 保留主机、boot ID、PID/start ticks、命令行及重复读取的身份核验结果；UNKNOWN 不表示运行终止，study.lock 也不构成具体 PID 存活证明。生成 CI 的缓存核对覆盖请求身份、样本数和描述点估计；旧 generation_summary 未保存对全部逐样本结局的密码学绑定，本脚本不把该检查写成独立 CI 重算。未显式开启 --verify-diagnostic-hashes 时，仅核对诊断摘要清单/文件大小与提交文件，未重读全部诊断 token/NPZ 内容。输出目录独立于证据目录，原始 run 保持只读。", ""]
    return "\n".join(lines)


def generation_table(group, arms=None):
    selected = arms or [arm for arm in ARMS if arm in group["arms"]]
    if not selected:
        return ["", "PENDING：尚未冻结入围方法或尚无该 scope 的生成记录。", ""]
    lines = ["", "|方法|记录/计划|Strict|Tolerant|平均/中位/p90 token|EOS|行政截断|错误请求|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in selected:
        data = group["arms"].get(arm)
        if not data:
            lines.append(f"|{arm}|0/{len(group['expected_ids'])}|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|")
            continue
        length = data["length"]
        lines.append(f"|{arm}|{data['n']}/{data['n_expected']}|{number(data['strict']['value'],True)} (n={data['strict']['n']})|{number(data['tolerant']['value'],True)} (n={data['tolerant']['n']})|{number(length['mean'])}/{number(length['median'])}/{number(length['p90'])}|{number(data['eos_rate'],True)}|{number(data['cap_rate'],True)}|{sum(data['stop_reason_counts'].get(x,0) for x in FAILED)}|")
    lines += ["", "缺失请求单列，不从完成样本分数推断完整集合质量；失败尝试另见失败账本。"]
    if any(data["missing_ids"] or data["unexpected_ids"] for data in group["arms"].values()):
        lines.append("")
    for arm, data in group["arms"].items():
        if data["missing_ids"] or data["unexpected_ids"]:
            lines.append(f"- {arm}：缺失 {len(data['missing_ids'])} 个 ID（前12个 {data['missing_ids'][:12]}）；额外 {len(data['unexpected_ids'])} 个 ID（前12个 {data['unexpected_ids'][:12]}）。完整 ID 清单见结构化报告。")
    return lines


def cost_table(group):
    if not group["arms"]:
        return ["", "PENDING：尚无该 scope 的逐请求成本测量。", ""]
    lines = ["", "|方法|无探针计时 n|总/Prefill/Decode 平均秒|峰值 allocated/reserved GiB（各请求最大）|重复4-gram均值|####比例|标记后token均值|简化专家assignment均值|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        data = group["arms"].get(arm)
        if not data:
            continue
        allocated, reserved = data["peak_allocated_bytes"]["max"], data["peak_reserved_bytes"]["max"]
        memory = "/".join(number(value/2**30 if finite(value) else None) for value in (allocated,reserved))
        seconds = "/".join(number(data[key]["mean"]) for key in ("generation_seconds","prefill_seconds","decode_seconds"))
        lines.append(f"|{arm}|{data['generation_seconds']['n']}|{seconds}|{memory}|{number(data['repeat_4gram']['mean'])}|{number(data['marker_rate'],True)}|{number(data['post_marker_tokens']['mean'])}|{number(data['expert_token_assignments_simplified']['mean'])}|")
    return lines+[""]


def comparison_lines(group, names=None):
    lines = ["", f"`{group['scope']}` 的保存配对差值（左减右，95% CI）：", ""]
    comparisons = group["comparisons"]
    selected = list(names) if names else list(comparisons)
    if not selected:
        return lines+["PENDING：没有与当前原始请求一致的完整配对区间。", ""]
    for name in selected:
        comparison = comparisons.get(name, {})
        metrics = comparison.get("metrics", {})
        lines.append(f"- `{name}`：Strict {interval_text(metrics.get('strict_accuracy'),True)}；Tolerant {interval_text(metrics.get('tolerant_accuracy'),True)}；token {interval_text(metrics.get('mean_generated_tokens'),True)}；EOS {interval_text(metrics.get('eos_rate'),True)}；请求秒 {interval_text(metrics.get('mean_generation_seconds'),True)}。")
        strict, tolerant = metrics.get("strict_accuracy",{}).get("mean"), metrics.get("tolerant_accuracy",{}).get("mean")
        if finite(strict) and finite(tolerant) and strict*tolerant < 0:
            lines.append("  两种评分方向相反：质量结论不确定，不选有利的一列。")
    return lines+[""]


def positive_gate_tables(positive):
    downstream=positive.get('downstream')
    if not downstream:return ["", "前瞻质量/ratio新统计：PENDING；未因数据缺失运行新的真实成本 bootstrap。", ""]
    lines=["", "前瞻补充在固定 D_validate 的新端点（离线分析，不计入生成时间）：", "",
        "|质量比较/评分|n|准确率差|同时保守精确下界|状态|", "|---|---:|---:|---:|---|"]
    for name,row in downstream.get('quality',{}).items():
        lines.append(f"|{name}|{row.get('n','PENDING')}|{number(row.get('estimate'))}|{number(row.get('exact_lower_bound'))}|{row.get('status')}|")
    lines += ["", "|M_ST 相对 control|相对token减少|名义同时区间|undefined draws|状态|", "|---|---:|---:|---:|---|"]
    for name,row in downstream.get('cost',{}).items():
        lines.append(f"|{name}|{number(row.get('estimate'),True)}|[{number(row.get('lower_bound'),True)}, {number(row.get('upper_bound'),True)}]|{row.get('undefined_draws','PENDING')}|{row.get('status')}|")
    return lines+["",f"数据/计划绑定缓存命中：`{downstream.get('cache_reused',False)}`。质量界是保守二项精确界；成本是 percentile-bootstrap 的近似覆盖。P6 自动授权始终为 `False`。", ""]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", help="Explicit derived-report directory outside run-dir; default prints Markdown")
    parser.add_argument("--verify-diagnostic-hashes", action="store_true", help="Stream all committed diagnostic NPZ/token files through SHA256 (CPU/I/O only)")
    parser.add_argument("--quality-plan",help="Independent prospective supplement; default reports/<run.name>/quality_plan.json")
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    if args.output_dir:
        output = Path(args.output_dir).resolve()
        if output == run or output.is_relative_to(run):
            parser.error("output-dir must be outside the read-only run-dir")
        if output.is_relative_to(Path.home().resolve()):
            parser.error("derived result artifacts must be outside Home, e.g. under /scratch/jgu7/")
    result = collect(run, args.verify_diagnostic_hashes,args.quality_plan,args.output_dir)
    markdown = render(result)
    if args.output_dir:
        output.mkdir(parents=True, exist_ok=True)
        for name, body in (("SCIENTIFIC_REPORT.md",markdown),("RUN_REPORT.md",markdown), ("scientific_summary.json",json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")):
            temporary = output/("."+name+".next")
            temporary.write_text(body, encoding="utf-8")
            temporary.replace(output/name)
        print(output/"SCIENTIFIC_REPORT.md")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
