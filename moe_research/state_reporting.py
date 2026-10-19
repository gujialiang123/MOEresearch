"""CPU-only generation summaries and preregistered paired-prompt comparisons.

Functions return data or Markdown; the caller chooses scratch output paths.
Missing rows never become complete pairs. Errors remain in request-level quality
denominators, while unavailable length/time measurements are never imputed.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

SEED = 20261019
FAILURE_REASONS = {"timeout", "oom", "error"}
COMPLETED_REASONS = {"eos", "max_new_tokens"}
DEFAULT_COMPARISONS = (("F", "N"), ("R8", "F"), ("R16", "R8"),
                       ("L_N", "M_N"), ("M_ST", "M_NT"))
REFERENCE_ORDER = ("F", "R8", "R16")
IDENTITY_FIELDS = ("question_hash", "protocol_hash", "parser_version",
                   "model_revision", "model_hash", "dtype", "max_new_tokens",
                   "rendered_prompt", "prompt_token_ids")
UNITS = {"strict_accuracy": "fraction", "tolerant_accuracy": "fraction",
         "mean_generated_tokens": "tokens", "eos_rate": "fraction",
         "administrative_cap_rate": "fraction", "error_rate": "fraction",
         "mean_generation_seconds": "seconds"}


def _rows(value):
    if isinstance(value, (str, Path)):
        with Path(value).open() as handle:
            result = [json.loads(line) for line in handle if line.strip()]
    elif isinstance(value, Mapping):
        result = list(value.values())
    else:
        result = list(value)
    ids = []
    for row in result:
        if not isinstance(row, Mapping) or "sample_id" not in row:
            raise ValueError("Each generation row must contain sample_id")
        sid = row["sample_id"]
        if isinstance(sid, bool) or not isinstance(sid, (str, int)):
            raise ValueError("sample_id must be an integer or string")
        ids.append(sid)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample_id; paired inference requires one request per arm/prompt")
    return sorted(result, key=lambda row: (str(type(row["sample_id"])), row["sample_id"]))


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _distribution(values):
    values = [float(x) for x in values if _finite(x)]
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None, "max": None, "sum": None}
    array = np.asarray(values, dtype=np.float64)
    return {"n": len(values), "mean": float(array.mean()), "median": float(np.median(array)),
            "p90": float(np.quantile(array, .9)), "max": float(array.max()), "sum": float(array.sum())}


def _request_score(row, field):
    if row.get("stop_reason") in FAILURE_REASONS:
        return 0.0
    value = row.get(field)
    return float(value) if isinstance(value, bool) else None


def _metric(row, name):
    if name == "strict_accuracy":
        return _request_score(row, "strict_correct")
    if name == "tolerant_accuracy":
        return _request_score(row, "tolerant_correct")
    if name == "mean_generated_tokens":
        value = row.get("generated_tokens")
        return float(value) if _finite(value) and value >= 0 else None
    if name in ("eos_rate", "administrative_cap_rate", "error_rate"):
        reason = row.get("stop_reason")
        if reason not in COMPLETED_REASONS | FAILURE_REASONS:
            return None
        target = {"eos_rate": {"eos"}, "administrative_cap_rate": {"max_new_tokens"},
                  "error_rate": FAILURE_REASONS}[name]
        return float(reason in target)
    if name == "mean_generation_seconds":
        # Measured deployment timing requires explicit probe/debug exclusion.
        if row.get("observer_enabled") is not False or row.get("timing_debug_logits") is not False:
            return None
        value = row.get("generation_seconds")
        return float(value) if _finite(value) and value >= 0 else None
    raise KeyError(name)


def summarize_generations(rows, expected_ids=None):
    """Describe recorded requests, preserving missing IDs and every failure.

    Accuracy is per recorded request, with failed requests counted incorrect.
    Missing rows are separately identified; no complete-dataset score is claimed.
    Other statistics use available measurements with explicit counts.
    """
    rows = _rows(rows)
    ids = [row["sample_id"] for row in rows]
    expected = set(ids if expected_ids is None else expected_ids)
    missing, unexpected = expected - set(ids), set(ids) - expected
    reasons = Counter(row.get("stop_reason", "unknown") for row in rows)
    metrics, counts = {}, {}
    for name in UNITS:
        values = [_metric(row, name) for row in rows]
        valid = [x for x in values if x is not None]
        metrics[name] = float(np.mean(valid)) if valid else None
        counts[name] = len(valid)
    disagreements = [row for row in rows if row.get("parser_disagreement") or
        (isinstance(row.get("strict_correct"), bool) and
         isinstance(row.get("tolerant_correct"), bool) and
         row["strict_correct"] != row["tolerant_correct"])]
    # Prioritize scoring conflicts, then preserve ascending IDs; cap exports at20.
    disagreements.sort(key=lambda r: (r.get("strict_correct") == r.get("tolerant_correct"),
                                      str(r["sample_id"])))
    examples = [{key: row.get(key) for key in ("sample_id", "question_hash", "question", "arm",
        "gold", "pred_strict", "pred_tolerant", "strict_correct", "tolerant_correct",
        "strict_parse_status", "tolerant_parse_status", "raw_output", "stop_reason")}
        for row in disagreements[:20]]
    invariant_conflicts = []
    for name in ("protocol_hash", "parser_version", "model_revision", "model_hash", "dtype", "max_new_tokens", "arm_hash"):
        values = {json.dumps(row[name], sort_keys=True) for row in rows if name in row}
        if len(values) > 1:
            invariant_conflicts.append(name)
    successful = [row for row in rows if row.get("stop_reason") in COMPLETED_REASONS]
    caps = sorted({row["max_new_tokens"] for row in rows if _finite(row.get("max_new_tokens"))})
    issues = []
    if missing or unexpected:
        issues.append("expected_sample_ids_incomplete_or_extra")
    if invariant_conflicts:
        issues.append("mixed_protocol_or_arm_identity")
    if reasons.get("unknown") or any(r not in COMPLETED_REASONS | FAILURE_REASONS for r in reasons):
        issues.append("unknown_stop_reason")
    if counts["strict_accuracy"] != len(rows) or counts["tolerant_accuracy"] != len(rows):
        issues.append("missing_request_scores")
    return {"n_recorded": len(rows), "n_expected": len(expected), "sample_ids": ids,
        "pairing_identity_hash": hashlib.sha256(json.dumps([
            {"sample_id":row["sample_id"], **{field:row[field] for field in IDENTITY_FIELDS if field in row}}
            for row in rows], sort_keys=True, ensure_ascii=False, separators=(",",":")).encode()).hexdigest(),
        "missing_ids": sorted(missing, key=str), "unexpected_ids": sorted(unexpected, key=str),
        "complete_expected_ids": bool(rows) and not missing and not unexpected,
        "stop_reason_counts": dict(reasons), "n_failed": sum(reasons[r] for r in FAILURE_REASONS),
        "n_completed": len(successful), **metrics, "metric_counts": counts,
        "length": _distribution([row.get("generated_tokens") for row in rows]),
        "length_completed_requests": _distribution([row.get("generated_tokens") for row in successful]),
        "prompt_tokens": _distribution([row.get("prompt_tokens") for row in rows]),
        "repeated_4gram_fraction": _distribution([row.get("repeated_4gram_fraction") for row in rows]),
        "answer_marker_rate": (sum(row.get("has_answer_marker") is True for row in successful) /
                               len(successful)) if successful else None,
        "first_marker_position": _distribution([row.get("first_marker_position") for row in rows]),
        "post_marker_tokens": _distribution([row.get("post_marker_tokens") for row in rows]),
        "scoring_disagreement_count": sum(row.get("strict_correct") != row.get("tolerant_correct")
            for row in rows if "strict_correct" in row and "tolerant_correct" in row),
        "parser_disagreement_count": len(disagreements), "parser_disagreement_examples": examples,
        "generation_seconds": _distribution([_metric(row, "mean_generation_seconds") for row in rows]),
        "prefill_seconds": _distribution([row.get("prefill_seconds") for row in rows
            if row.get("observer_enabled") is False and row.get("timing_debug_logits") is False]),
        "decode_seconds": _distribution([row.get("decode_seconds") for row in rows
            if row.get("observer_enabled") is False and row.get("timing_debug_logits") is False]),
        "peak_allocated_bytes": _distribution([row.get("peak_allocated_bytes") for row in rows
            if row.get("observer_enabled") is False and row.get("timing_debug_logits") is False]),
        "peak_reserved_bytes": _distribution([row.get("peak_reserved_bytes") for row in rows
            if row.get("observer_enabled") is False and row.get("timing_debug_logits") is False]),
        "expert_token_assignments_simplified": _distribution([
            row.get("expert_token_assignments_simplified") for row in rows]),
        "expert_budget_labels": sorted({row["budget_definition"] for row in rows if row.get("budget_definition")}),
        "max_new_tokens_values": caps, "invariant_conflicts": invariant_conflicts, "issues": issues,
        "accuracy_definition": "Per recorded request; timeout/oom/error count incorrect; missing rows remain missing.",
        "length_definition": "Observed tokens to EOS/cap/failure when measured; cap-censored lengths do not identify natural lengths beyond cap.",
        "timing_definition": "Only explicit observer_enabled=false and timing_debug_logits=false rows; current backend only."}


def paired_generation_bootstrap(leftrows, rightrows, iters=10000, seed=SEED, expected_ids=None):
    """Prompt bootstrap of left-minus-right metrics; never intersect incomplete IDs."""
    if not isinstance(iters, int) or iters < 1:
        raise ValueError("iters must be positive")
    leftrows, rightrows = _rows(leftrows), _rows(rightrows)
    left = {r["sample_id"]: r for r in leftrows}
    right = {r["sample_id"]: r for r in rightrows}
    left_ids, right_ids = set(left), set(right)
    expected = left_ids | right_ids if expected_ids is None else set(expected_ids)
    mismatch = {"left_missing": sorted(expected-left_ids, key=str),
                "right_missing": sorted(expected-right_ids, key=str),
                "left_unexpected": sorted(left_ids-expected, key=str),
                "right_unexpected": sorted(right_ids-expected, key=str)}
    result = {"direction": "left_minus_right", "bootstrap_unit": "prompt", "seed": seed,
              "iterations": iters, "confidence_level": .95, "pairing": mismatch,
              "status": "BLOCKED_INCOMPLETE_PAIRS", "formal_valid": False,
              "n_pairs": 0, "metrics": {}}
    if not left_ids or left_ids != right_ids or any(mismatch.values()):
        return result
    ids = [r["sample_id"] for r in leftrows]
    identity_conflicts = []
    for sid in ids:
        for field in IDENTITY_FIELDS:
            if field in left[sid] or field in right[sid]:
                if left[sid].get(field) != right[sid].get(field):
                    identity_conflicts.append({"sample_id": sid, "field": field})
    left_summary, right_summary = summarize_generations(leftrows), summarize_generations(rightrows)
    if identity_conflicts or left_summary["invariant_conflicts"] or right_summary["invariant_conflicts"]:
        result.update(status="BLOCKED_IDENTITY_MISMATCH", identity_conflicts=identity_conflicts,
            within_arm_conflicts={"left":left_summary["invariant_conflicts"], "right":right_summary["invariant_conflicts"]})
        return result
    metric_names, deltas = [], []
    for name, unit in UNITS.items():
        pairs = [(_metric(left[sid], name), _metric(right[sid], name)) for sid in ids]
        missing = [sid for sid, pair in zip(ids, pairs) if None in pair]
        if missing:
            result["metrics"][name] = {"status":"BLOCKED_MISSING_MEASUREMENTS", "missing_ids": missing,
                                      "mean": None, "ci95": None, "unit": unit}
        else:
            metric_names.append(name)
            deltas.append([a-b for a,b in pairs])
    if deltas:
        matrix = np.asarray(deltas, dtype=np.float64).T
        rng, draws = np.random.default_rng(seed), []
        for start in range(0, iters, 256):
            indexes = rng.integers(0, len(ids), size=(min(256, iters-start), len(ids)))
            draws.append(matrix[indexes].mean(axis=1))
        samples = np.concatenate(draws, axis=0)
        for column, name in enumerate(metric_names):
            interval = np.quantile(samples[:,column], [.025,.975])
            result["metrics"][name] = {"status":"OK", "mean": float(matrix[:,column].mean()),
                "ci95": interval.tolist(), "n":len(ids), "unit":UNITS[name]}
    result.update(status="OK", formal_valid=True, n_pairs=len(ids),
        caveat="These CIs condition on frozen calibration parameters; no equivalence/noninferiority claim or p-value is inferred.")
    return result


def build_generation_summary(dict_arm_rows, expected_ids=None, bootstrap_iters=10000, comparisons=None):
    """Build descriptive tables plus only the requested, predeclared comparisons."""
    loaded = {name:_rows(rows) for name,rows in dict_arm_rows.items()}
    arms = {}
    for name, rows in loaded.items():
        if any(row.get("arm", name) != name for row in rows):
            raise ValueError(f"Arm label mismatch in {name}")
        arms[name] = summarize_generations(rows, expected_ids)
    compared = {}
    for left, right in DEFAULT_COMPARISONS if comparisons is None else comparisons:
        key = f"{left}_minus_{right}"
        if left not in loaded or right not in loaded:
            compared[key] = {"status":"NOT_AVAILABLE", "formal_valid":False,
                             "missing_arms":[a for a in (left,right) if a not in loaded]}
        else:
            compared[key] = paired_generation_bootstrap(loaded[left], loaded[right],
                iters=bootstrap_iters, expected_ids=expected_ids)
    return {"schema_version":1, "arms":arms, "comparisons":compared,
        "all_expected_rows_present": bool(arms) and all(v["complete_expected_ids"] for v in arms.values()),
        "bootstrap":{"unit":"prompt", "iterations":bootstrap_iters, "seed":SEED},
        "statistical_scope":"Descriptive generation data and frozen-parameter paired CIs; no mechanism/causality/noninferiority decision."}


def choose_baseline(summary, *, dataset_role="D_select"):
    """Deterministic reference selection on D_select; S0 remains predefined L_N."""
    if dataset_role != "D_select":
        raise ValueError("Baseline selection is allowed only on D_select")
    arms = summary["arms"]
    if any(arm not in arms for arm in REFERENCE_ORDER):
        raise ValueError("F, R8 and R16 must all be present for reference selection")
    expected = set(arms["F"]["sample_ids"])
    ranked = []
    for order, arm in enumerate(REFERENCE_ORDER):
        data = arms[arm]
        if not data["complete_expected_ids"] or set(data["sample_ids"]) != expected or data["issues"]:
            raise ValueError("Reference selection requires complete, identical, consistent D_select IDs")
        if data["pairing_identity_hash"] != arms["F"]["pairing_identity_hash"]:
            raise ValueError("Reference selection requires matching prompt/protocol identities")
        if data["metric_counts"]["mean_generated_tokens"] != data["n_recorded"]:
            raise ValueError("Reference selection cannot impute missing output lengths")
        quality = min(data["strict_accuracy"], data["tolerant_accuracy"])
        ranked.append((quality, data["length"]["mean"], order, arm))
    ranked.sort(key=lambda x:(-x[0], x[1], x[2]))
    return {"dataset_role":"D_select", "selected_reference":ranked[0][3],
        "selected_calibration":"L_N", "S0":"L_N",
        "rule":"greatest min(strict_accuracy,tolerant_accuracy), then lower mean observed length, then fixed order F/R8/R16",
        "ranking":[{"arm":a,"minimum_accuracy":q,"mean_generated_tokens":length} for q,length,_,a in ranked],
        "sample_ids":arms["F"]["sample_ids"],
        "sample_id_hash":hashlib.sha256(json.dumps(arms["F"]["sample_ids"],separators=(",",":")).encode()).hexdigest(),
        "note":"Selection is exploratory; freeze final comparisons before D_validate. L_N is predefined rather than selected by downstream results."}


def render_generation_report(summary, context=None, selection=None):
    """Render Chinese Markdown from measured data; never creates files or claims a gate passed."""
    def number(x, percent=False):
        if x is None:
            return "未测得"
        return f"{x*100:.2f}%" if percent else f"{x:.3f}"
    lines = ["# 自由生成结果汇总", ""]
    for key,value in (context or {}).items():
        lines.append(f"- {key}：{value}")
    lines += ["", "|方法|已记录/计划|Strict|Tolerant|平均 token|中位数|p90|自然 EOS|行政截断|失败|平均请求秒|",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm,data in summary["arms"].items():
        lines.append(f"|{arm}|{data['n_recorded']}/{data['n_expected']}|{number(data['strict_accuracy'],True)}|{number(data['tolerant_accuracy'],True)}|"
            f"{number(data['length']['mean'])}|{number(data['length']['median'])}|{number(data['length']['p90'])}|"
            f"{number(data['eos_rate'],True)}|{number(data['administrative_cap_rate'],True)}|{data['n_failed']}|"
            f"{number(data['generation_seconds']['mean'])}|")
    lines += ["", "失败请求在请求级准确率中计错；缺失记录单列。EOS 恰在 cap 最后一位仍按自然结束统计。长度是已观测 token 数，不能恢复 cap 之后未知的自然长度。",
              "", "## 数据完整性、重复、格式与内存", ""]
    for arm,data in summary["arms"].items():
        lines.append(f"- {arm}：缺失 {len(data['missing_ids'])}，额外 {len(data['unexpected_ids'])}；停止原因 {data['stop_reason_counts']}；"
            f"重复 4-gram 均值 {number(data['repeated_4gram_fraction']['mean'])}；答案标记率 {number(data['answer_marker_rate'],True)}；"
            f"评分冲突 {data['scoring_disagreement_count']}；峰值 allocated/reserved 最大值 "
            f"{number(data['peak_allocated_bytes']['max'])}/{number(data['peak_reserved_bytes']['max'])} 字节。")
    lines += ["", "## 预先指定的配对差值", "", "差值方向为左方法减右方法；95% CI 按 prompt 成对重采样，校准参数固定。", "",
              "|比较|状态|指标|差值|95% CI|", "|---|---|---|---:|---|"]
    for name,comparison in summary["comparisons"].items():
        if not comparison.get("formal_valid"):
            lines.append(f"|{name}|{comparison['status']}|未作正式配对推断|—|—|")
            continue
        for metric,value in comparison["metrics"].items():
            interval = value.get("ci95")
            interval_text = f"[{number(interval[0])}, {number(interval[1])}]" if interval else "未测得"
            lines.append(f"|{name}|{value['status']}|{metric} ({value['unit']})|{number(value['mean'])}|{interval_text}|")
    if selection:
        lines += ["", "## D_select 选择", "",
            f"reference 方法：`{selection['selected_reference']}`；初始校准器 S0：`{selection['S0']}`。",
            f"规则：{selection['rule']}。"]
    lines += ["", "## 解释边界", "",
        "- 缺失配对或协议/题目身份不一致时，正式比较被阻止；不会仅取有利交集。",
        "- Strict 与 tolerant 并列；方向相反时，质量结论仍不确定。",
        "- CI 包含零不证明等价或非劣；本模块不作无损判断。",
        "- 时延仅使用明确关闭 observer 和 logits 调试的记录，代表当前研究后端。专家预算字段保留简化预算标签。",
        "- 本汇总不判定 H1/H2/H3 或 P6/P7 门槛；这些需要同文本诊断、重估后自身轨迹、冻结方案和独立确认的实际证据。"]
    return "\n".join(lines) + "\n"
