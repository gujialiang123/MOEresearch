#!/usr/bin/env python3
"""CPU-only acceptance and immutable delivery of the observer cost appendix.

Reads actual committed matched replays.  No model, fitting, generation,
bootstrap, or change to a scientific classification is performed here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

if __package__:
    from . import state_study_validation512_report as science
    from .freeze_state_study_quality_plan import canonical, load, sha, publish_once, Invalid, Pending
else:
    import state_study_validation512_report as science
    from freeze_state_study_quality_plan import canonical, load, sha, publish_once, Invalid, Pending

IDS = (128, 129, 130, 131)
MEMORY_KEYS = ("baseline_allocated_bytes", "baseline_reserved_bytes", "peak_allocated_bytes",
               "peak_reserved_bytes", "incremental_peak_allocated_bytes", "incremental_peak_reserved_bytes")
TIME_KEYS = ("replay_seconds", "observer_callback_host_seconds", "archive_seconds")
FINITE_LOGIT_GUARD = "all returned logits before every argmax; included common driver overhead"


def _cost_module():
    if __package__:
        from . import state_study_observer_cost
    else:
        import state_study_observer_cost
    return state_study_observer_cost


def _sealed(value, field):
    try:
        return value.get(field) == canonical({k: v for k, v in value.items() if k != field})
    except (TypeError, ValueError):
        return False


def _number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _relative(directory, name):
    path = (Path(directory) / name).resolve()
    if not path.is_relative_to(Path(directory).resolve()):
        raise Invalid("Evidence path escapes output directory")
    return path


def validate_modes(unit, pair, modes, plan_hash):
    """Independent scalar/trace checks in addition to the collector's audit."""
    trace = unit["trace"]
    expected_mask = canonical([[j, token] for j, token in enumerate(trace[:-1], 1)])
    if set(modes) != {"on", "off"}:
        raise Invalid("Both measured modes are required")
    for mode, row in modes.items():
        if (not _sealed(row, "mode_hash") or row.get("schema_version") != "state_observer_cost.mode.v1"
                or row.get("plan_hash") != plan_hash or row.get("unit_hash") != unit["unit_hash"]
                or row.get("mode") != mode or row.get("capture_logits") is not False):
            raise Invalid("Measured mode scope/self hash differs")
        replay = row.get("replay", {})
        if (replay.get("decode_forwards") != len(trace)-1 or replay.get("mask_hash") != expected_mask
                or len(replay.get("argmaxes", [])) != len(trace) or replay.get("logits_finiteness_check") != FINITE_LOGIT_GUARD):
            raise Invalid("Complete replay T-1/mask/finiteness proof differs")
        if unit["source_argmax_required"] and replay["argmaxes"] != trace:
            raise Invalid("Actual own/native replay argmax differs from saved trace")
        if row.get("observer_calls") != (48*(len(trace)-1) if mode == "on" else 0):
            raise Invalid("Measured 48-layer callback coverage differs")
        timing = row.get("timing", {})
        if any(not _number(timing.get(k)) or timing[k] < 0 for k in TIME_KEYS):
            raise Invalid("Measured mode time is missing, negative, or nonfinite")
        memory = row.get("memory", {})
        if any(type(memory.get(k)) is not int or memory[k] < 0 for k in MEMORY_KEYS):
            raise Invalid("Measured memory is missing or invalid")
        for kind in ("allocated", "reserved"):
            if memory[f"peak_{kind}_bytes"] - memory[f"baseline_{kind}_bytes"] != memory[f"incremental_peak_{kind}_bytes"]:
                raise Invalid("Measured incremental memory is inconsistent")
        if mode == "on" and not isinstance(row.get("archive_receipt"), dict):
            raise Invalid("Observer-on committed archive receipt is missing")
        if mode == "off" and (row.get("archive_receipt") is not None
                or timing["observer_callback_host_seconds"] != 0 or timing["archive_seconds"] != 0):
            raise Invalid("Observer-off contains observer/archive work")
    if modes["on"]["replay"]["argmaxes"] != modes["off"]["replay"]["argmaxes"]:
        raise Invalid("Observer changed measured argmax")
    delta = {"replay_seconds": modes["on"]["timing"]["replay_seconds"] - modes["off"]["timing"]["replay_seconds"],
             **{k: modes["on"]["memory"][k] - modes["off"]["memory"][k] for k in MEMORY_KEYS}}
    if pair.get("signed_differences") != delta:
        raise Invalid("Signed on-minus-off deltas differ")
    return delta


def _pair_details(directory, unit, plan, snapshot):
    pair = load(Path(directory)/"pairs"/(unit["unit_hash"]+".json"))
    attempt = _relative(directory, pair["attempt_relative_dir"])
    intent = load(attempt/"intent.json")
    if any(intent.get(k) != v for k, v in {"plan_hash": plan["plan_hash"],
            "source_snapshot_hash": snapshot["snapshot_hash"], "unit_hash": unit["unit_hash"], "order": unit["order"]}.items()):
        raise Invalid("Attempt intent differs from source/plan/order")
    warmup = load(attempt/"warmup.json")
    warm_trace = unit["trace"][:8]
    replay = warmup.get("replay", {})
    if (warmup.get("excluded_from_measurement") is not True or warmup.get("trace_length") != len(warm_trace)
            or not _number(warmup.get("elapsed_seconds")) or warmup["elapsed_seconds"] < 0
            or replay.get("decode_forwards") != len(warm_trace)-1
            or replay.get("mask_hash") != canonical([[j, t] for j, t in enumerate(warm_trace[:-1], 1)])
            or len(replay.get("argmaxes", [])) != len(warm_trace)
            or replay.get("logits_finiteness_check") != FINITE_LOGIT_GUARD
            or warmup.get("observer_calls") != 48*(len(warm_trace)-1)
            or (unit["source_argmax_required"] and replay["argmaxes"] != warm_trace)):
        raise Invalid("Unmeasured eight-token observer warmup proof differs")
    modes = {m: load(_relative(directory, pair["modes"][m]["relative_path"])) for m in ("on", "off")}
    validate_modes(unit, pair, modes, plan["plan_hash"])
    details = {"sample_id": unit["sample_id"], "scenario": unit["scenario"], "order": unit["order"],
               "trace_tokens": len(unit["trace"]), "decode_forwards": len(unit["trace"])-1,
               "signed_differences": pair["signed_differences"], "modes": modes}
    auxiliary = {str(p.relative_to(directory)): sha(p) for p in (attempt/"intent.json", attempt/"warmup.json")}
    return details, auxiliary


def _descriptive(pairs, scenarios):
    result = {}
    for scenario in scenarios:
        selected = [p for p in pairs if p["scenario"] == scenario]
        result[scenario] = {"n_pairs": len(selected), "signed_difference_descriptive": {
            k: {"mean": statistics.mean(p["signed_differences"][k] for p in selected),
                "median": statistics.median(p["signed_differences"][k] for p in selected)}
            for k in selected[0]["signed_differences"]}}
    return result


def check_observer_cost(run, plan_path=None):
    """Verify work independently of the signs of time or memory differences."""
    result = {"complete": False, "data_complete": False, "not_required": False, "status": "PENDING",
              "errors": [], "requirements": {}, "no_model_or_bootstrap_execution": True}
    try:
        cost = _cost_module(); plan = cost.load_plan(run, plan_path)
        directory = Path(cost.output_dir(run, plan)).resolve()
        result.update(plan_hash=plan["plan_hash"], output_dir=str(directory))
        snapshot, units, calibrations = cost.source_snapshot(run, plan)
        scenarios = ["A", "B", "C", "D"] + (["M_NT", "M_ST"] if snapshot.get("P4_triggered") is True else [])
        req = result["requirements"]
        req["fixed_four_prompts_and_actual_P4_roles"] = (snapshot.get("sample_ids") == list(IDS)
            and snapshot.get("scenarios") == scenarios
            and [(u["scenario"], u["sample_id"]) for u in units] == [(a, i) for a in scenarios for i in IDS]
            and all(_sealed(u, "unit_hash") and u["order"] == (["on", "off"] if (u["sample_id"]+u["scenario_index"])%2 == 0 else ["off", "on"]) for u in units))
        if not all(req.values()): raise Invalid("Expected engineering scope/roles/order differs")
        req["saved_source_snapshot_matches_actual"] = load(directory/"source_snapshot.json") == snapshot
        receipt = cost.verify_raw_completion(directory, plan, snapshot, units, calibrations)
        req["all_pair_commits_archives_sha_48layer_fullmask"] = True
        pairs = []; auxiliary = {}
        for unit in units:
            cost.audit_pair(directory, unit, calibrations, plan["protocol_hash"])
            details, files = _pair_details(directory, unit, plan, snapshot)
            pairs.append(details); auxiliary.update(files)
        req["both_modes_finite_neutral_and_warmup_excluded"] = True
        result.update(source_snapshot_hash=snapshot["snapshot_hash"], pair_count=len(pairs),
                      scenarios=scenarios, pairs=pairs, auxiliary_evidence_sha256=auxiliary,
                      raw_completion_hash=receipt["raw_completion_hash"])
        result["data_complete"] = all(req.values())
        summary_path = directory/"summaries/observer_cost_summary.json"
        summary = load(summary_path); done = load(directory/"completion.json")
        req["completion_self_hash_and_actual_summary_sha"] = (_sealed(done, "completion_hash")
            and all(done.get(k) == v for k, v in {"schema_version": "state_observer_cost.complete.v1",
                "plan_hash": plan["plan_hash"], "source_snapshot_hash": snapshot["snapshot_hash"],
                "raw_completion_hash": receipt["raw_completion_hash"], "summary_sha256": sha(summary_path), "completed_work": True}.items()))
        req["descriptive_summary_reproduces_actual_pairs"] = all(summary.get(k) == v for k, v in {
            "schema_version": "state_observer_cost.summary.v1", "plan_hash": plan["plan_hash"],
            "source_snapshot_hash": snapshot["snapshot_hash"], "status": "ENGINEERING_SUBSET_ONLY", "sample_ids": list(IDS),
            "scenarios": scenarios, "pair_count": len(pairs), "pairs": pairs,
            "by_scenario": _descriptive(pairs, scenarios), "bootstrap_performed": False}.items())
        result.update(summary=summary, completion_hash=done.get("completion_hash"),
                      work_files_sha256={name: sha(directory/name) for name in
                          ("source_snapshot.json", "raw_completion.json", "completion.json", "summaries/observer_cost_summary.json")})
        if not all(req.values()): raise Invalid("Published cost work fails: "+", ".join(k for k, v in req.items() if not v))
        return {**result, "complete": True, "status": "VERIFIED", "reason": "Measured work and descriptive output verified; signed effects are not a completion gate"}
    except Exception as exc:
        pending = isinstance(exc, (FileNotFoundError, Pending)) or ("cost" in locals() and isinstance(exc, cost.Pending))
        return {**result, "status": "PENDING" if pending else "BLOCKED", "errors": [] if pending else [str(exc)], "reason": str(exc)}


def calibration_cost_accounting(run):
    """Actual recorded components, never a fabricated total calibration wall."""
    root = Path(run)/"formal/calibration"; native = root/"calibration_cost.json"
    updated = root/"updated_calibration_cost_attempts.jsonl"; attempts = []
    if updated.is_file():
        with updated.open("rb") as handle:
            for index, line in enumerate(handle, 1):
                if not line.endswith(b"\n"): raise Invalid(f"Uncommitted calibration JSONL tail at physical line {index}")
                if not line.strip(): raise Invalid(f"Blank calibration JSONL physical line {index}")
                value = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(Invalid("Nonfinite cost JSON")))
                if not isinstance(value, dict): raise Invalid("Calibration cost attempt is not an object")
                attempts.append(value)
    cals = {}
    for arm in ("M_N", "L_N", "M_NT", "M_ST"):
        path = root/(arm+".json")
        if path.is_file():
            value = load(path)
            if not _sealed(value, "calibration_hash"): raise Invalid("Calibration cost metadata hash differs")
            cals[arm] = {"calibration_hash": value["calibration_hash"], "metadata": value.get("metadata", {}), "sha256": sha(path)}
    rollout_values = sorted({record[arm]["rollout_seconds_shared"] for record in attempts for arm in ("M_NT", "M_ST")
                             if isinstance(record.get(arm), dict) and _number(record[arm].get("rollout_seconds_shared"))})
    return {"native_record": load(native) if native.is_file() else None, "updated_attempt_records": attempts,
            "calibration_metadata": cals, "component_file_sha256": {str(p.relative_to(run)): sha(p) for p in (native, updated) if p.is_file()},
            "complete_offline_wall_seconds": None, "shared_student_rollout_values_recorded": rollout_values,
            "shared_student_rollout_seconds_once": rollout_values[0] if len(rollout_values) == 1 else None,
            "repeated_shared_rollout_fields_are_not_summed": True,
            "unknown_components": ["native finalize/serialization after recorded replay timer", "failed work before durable cost record",
                                   "warmup, I/O and unrecorded stage overhead; updated timer scope differs from native"],
            "no_equal_total_budget_claim": True, "no_fitting_replay_performed": True}


def _science_identity(run):
    verified = science.verify_delivery(run)
    if verified.get("verified") is not True or verified.get("delivery_complete") is not True:
        raise Pending("Verified full512/NOT_REQUIRED nine-question delivery required")
    directory = Path(verified["delivery_dir"])
    receipt = load(directory/"delivery_receipt.json"); summary = load(directory/"validation512_summary.json")
    return {"delivery_dir": str(directory), "receipt_sha256": sha(directory/"delivery_receipt.json"),
            "delivery_hash": receipt["delivery_hash"], "run_report_sha256": sha(directory/"RUN_REPORT.md"),
            "summary_sha256": sha(directory/"validation512_summary.json"), "scientific_status": summary["status"]}


def render_appendix(result):
    proof = result["observer_cost"]
    lines = ["# Observer 成本附页：固定四题的工程边际测量", "",
        "范围：C_diag sample_id 128..131；每条既存轨迹完整T−1、48层；A/B/C/D及真实P4触发后的NT/ST自身路径。",
        "这是observer on/off固定子集的边际工程测量，不是原run当时开销、全实验累计成本、拟合observer成本或生产服务加速。",
        "每unit在两mode前进行一次前min(8,T) token的observer-on未计时热身；两mode顺序平衡。同步回放计时含observer隐式同步及benchmark回调时钟/counter的host instrumentation，archive I/O另列。回调host时间不等于纯GPU kernel时间；不新增逐回调同步。",
        "单次配对仅报告实际数值和描述性mean/median，没有bootstrap、置信区间或外推。on−off允许为负，不删除或截零。", "",
        f"验收：`{proof['status']}`；实际配对 `{proof.get('pair_count', 0)}`；plan `{proof.get('plan_hash')}`。",
        f"原九问科学结论保持 **{result['scientific_delivery']['scientific_status']}**；本附页不修改positive gate或P6。", "",
        "|路径/题号|顺序|T−1|on秒|off秒|差秒|on/off回调host秒|on/off归档与验收秒|", "|---|---|---:|---:|---:|---:|---|---|"]
    for p in proof["pairs"]:
        on, off = (p["modes"][m] for m in ("on", "off"))
        lines.append(f"|{p['scenario']}/{p['sample_id']}|{'→'.join(p['order'])}|{p['decode_forwards']}|{on['timing']['replay_seconds']:.6g}|{off['timing']['replay_seconds']:.6g}|{p['signed_differences']['replay_seconds']:.6g}|{on['timing']['observer_callback_host_seconds']:.6g}/{off['timing']['observer_callback_host_seconds']:.6g}|{on['timing']['archive_seconds']:.6g}/{off['timing']['archive_seconds']:.6g}|")
    lines += ["", "|路径/题号/mode|baseline alloc/reserved bytes|peak alloc/reserved bytes|增量peak alloc/reserved bytes|", "|---|---|---|---|"]
    for p in proof["pairs"]:
        for mode in ("on", "off"):
            m = p["modes"][mode]["memory"]
            lines.append(f"|{p['scenario']}/{p['sample_id']}/{mode}|{m['baseline_allocated_bytes']}/{m['baseline_reserved_bytes']}|{m['peak_allocated_bytes']}/{m['peak_reserved_bytes']}|{m['incremental_peak_allocated_bytes']}/{m['incremental_peak_reserved_bytes']}|")
    lines += ["", "各路径有符号差的描述性汇总（四题，秒/bytes；不是推断区间）：", "", "```json",
              json.dumps(proof["summary"]["by_scenario"], ensure_ascii=False, indent=2), "```", "",
              "## 原9.3离线校准记录", "",
              "原始native replay计时在finalize/save前截止，updated replay计时含finalize.save；共享S0 rollout只列一次。记录不能拼成完整作业墙钟。缺失/失败未落盘、热身和I/O等未知，完整离线总秒数保留null；相同拟合token不等于相同总预算。本附页没有重跑拟合。", "",
              "```json", json.dumps(result["calibration_cost"], ensure_ascii=False, indent=2), "```", "",
              "机器可读JSON保留每个mode绝对baseline/peak及有符号差、源快照、commit/归档SHA、未提交attempt清单和证据路径。归档验收核对header/位置数组/计数/身份/实际字节SHA，不重新扫描或计算全部moment数值。旧科学报告及immutable bundles只读保留。", ""]
    return "\n".join(lines)


def _work_identity(proof):
    if proof.get("complete") is not True: raise Pending("Observer work is incomplete")
    return {k: proof[k] for k in ("plan_hash", "source_snapshot_hash", "raw_completion_hash", "completion_hash", "work_files_sha256", "auxiliary_evidence_sha256")}


def build_report(run, plan_path=None, *, work_proof=None):
    proof = work_proof if work_proof is not None else check_observer_cost(run, plan_path)
    _work_identity(proof)
    identity = _science_identity(run)
    result = {"schema_version": "state_observer_cost.report.v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "run_dir": str(Path(run).resolve()), "status": "ENGINEERING_SUBSET_ONLY", "scientific_status_unchanged": identity["scientific_status"],
              "observer_cost": proof, "scientific_delivery": identity, "calibration_cost": calibration_cost_accounting(run),
              "no_P6_authorization_change": True, "no_bootstrap_model_fit_or_generation": True}
    appendix = render_appendix(result)
    original = (Path(identity["delivery_dir"])/"RUN_REPORT.md").read_text()
    return result, original+"\n\n---\n\n"+appendix, appendix


def verify_delivery(run, plan_path=None, *, work_proof=None):
    result = {"verified": False, "delivery_complete": False, "status": "PENDING", "errors": []}
    try:
        cost = _cost_module(); plan = cost.load_plan(run, plan_path)
        delivery = Path(cost.output_dir(run, plan))/"delivery"; path = delivery/"delivery_receipt.json"
        result.update(delivery_dir=str(delivery), receipt_path=str(path), plan_hash=plan["plan_hash"])
        if not path.is_file(): return {**result, "reason": "Observer cost delivery is absent"}
        proof = work_proof if work_proof is not None else check_observer_cost(run, plan_path)
        work = _work_identity(proof); scientific = _science_identity(run); receipt = load(path)
        if (not _sealed(receipt, "delivery_hash") or receipt.get("schema_version") != "state_observer_cost.delivery.v1"
                or receipt.get("work") != work or receipt.get("scientific_delivery") != scientific or receipt.get("plan_hash") != plan["plan_hash"]):
            raise Invalid("Delivery source/work/scientific identity differs")
        files = receipt.get("files", {})
        if set(files) != {"RUN_REPORT.md", "OBSERVER_COST_REPORT.md", "observer_cost_summary.json"}: raise Invalid("Delivery inventory differs")
        bundle_hash = canonical({"work": work, "scientific_delivery": scientific, "files": files})
        if receipt.get("bundle_hash") != bundle_hash: raise Invalid("Delivery bundle identity differs")
        archived = delivery/"receipts"/(receipt["delivery_hash"]+".json")
        if not archived.is_file() or load(archived) != receipt: raise Invalid("Immutable receipt differs")
        for name, digest in files.items():
            if sha(delivery/name) != digest or sha(delivery/"bundles"/bundle_hash/name) != digest: raise Invalid("Actual delivery/bundle SHA differs")
        summary = load(delivery/"observer_cost_summary.json")
        if (summary.get("schema_version") != "state_observer_cost.report.v1" or summary.get("scientific_delivery") != scientific
                or summary.get("run_dir") != str(Path(run).resolve()) or _work_identity(summary.get("observer_cost", {})) != work
                or summary.get("scientific_status_unchanged") != scientific["scientific_status"]
                or summary.get("calibration_cost") != calibration_cost_accounting(run)):
            raise Invalid("Delivered summary/source accounting differs")
        expected_appendix = render_appendix(summary)
        original = (Path(scientific["delivery_dir"])/"RUN_REPORT.md").read_text()
        if ((delivery/"OBSERVER_COST_REPORT.md").read_text() != expected_appendix
                or (delivery/"RUN_REPORT.md").read_text() != original+"\n\n---\n\n"+expected_appendix
                or not all(f"## {i}." in original for i in range(1, 10))):
            raise Invalid("Final report is not the actual nine-question report plus cost appendix")
        return {**result, "verified": True, "delivery_complete": True, "status": "VERIFIED",
                "delivery_hash": receipt["delivery_hash"], "receipt_sha256": sha(path),
                "scientific_status": scientific["scientific_status"], "report_path": str(delivery/"RUN_REPORT.md")}
    except Exception as exc:
        pending = isinstance(exc, (Pending, FileNotFoundError)) or ("cost" in locals() and isinstance(exc, cost.Pending))
        return {**result, "status": "PENDING" if pending else "INVALID", "reason": str(exc), "errors": [] if pending else [str(exc)]}


def ensure_delivery(run, plan_path=None):
    """Only writer; scientific delivery must already exist, no statistics call."""
    previous = verify_delivery(run, plan_path)
    if previous.get("delivery_complete") is True: return {**previous, "reused": True}
    try:
        cost = _cost_module(); plan = cost.load_plan(run, plan_path)
        delivery = Path(cost.output_dir(run, plan))/"delivery"
        result, markdown, appendix = build_report(run, plan_path)
        work = _work_identity(result["observer_cost"]); scientific = result["scientific_delivery"]
        bodies = {"RUN_REPORT.md": markdown, "OBSERVER_COST_REPORT.md": appendix,
                  "observer_cost_summary.json": json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n"}
        digests = {name: hashlib.sha256(body.encode()).hexdigest() for name, body in bodies.items()}
        bundle_hash = canonical({"work": work, "scientific_delivery": scientific, "files": digests})
        bundle = delivery/"bundles"/bundle_hash; bundle.mkdir(parents=True, exist_ok=True)
        for name, body in bodies.items():
            target = bundle/name
            if target.exists():
                if sha(target) != digests[name]: raise Invalid("Immutable observer delivery bytes differ")
            else:
                with target.open("x", encoding="utf-8") as handle:
                    handle.write(body); handle.flush(); os.fsync(handle.fileno())
            science._atomic_text(delivery/name, body)
        value = {"schema_version": "state_observer_cost.delivery.v1", "plan_hash": plan["plan_hash"], "work": work,
                 "scientific_delivery": scientific, "files": digests, "bundle_hash": bundle_hash,
                 "generated_at_utc": result["generated_at_utc"], "scientific_classification_changed": False}
        value["delivery_hash"] = canonical(value)
        archived = delivery/"receipts"/(value["delivery_hash"]+".json")
        if archived.exists():
            if load(archived) != value: raise Invalid("Immutable observer receipt differs")
        else: publish_once(archived, value)
        science._atomic_text(delivery/"delivery_receipt.json", json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
        return {**verify_delivery(run, plan_path, work_proof=result["observer_cost"]), "reused": False}
    except Exception as exc:
        return {"verified": False, "delivery_complete": False, "status": "PENDING", "errors": [str(exc)],
                "reason": "Preserve completed measured data; resume CPU delivery only"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "deliver")); parser.add_argument("--run-dir", required=True); parser.add_argument("--plan")
    args = parser.parse_args(argv)
    result = check_observer_cost(args.run_dir, args.plan) if args.command == "check" else ensure_delivery(args.run_dir, args.plan)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result.get("complete") is True or result.get("delivery_complete") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
