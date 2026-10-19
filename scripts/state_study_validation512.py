#!/usr/bin/env python3
"""Independent all-512 own-state supplement; original run is read-only.

Freeze/status/check import only the standard library and locked CPU analysis
helpers. Torch, Transformers and frozen state modules are imported lazily after
registered GPU-run authorization. No Study, fitting or generation is invoked.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

REPO = Path(__file__).resolve().parents[1]
for _path in (str(REPO), str(REPO / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import freeze_state_study_quality_plan as quality

REPORTS = Path("/scratch/jgu7/moe-state-study/reports")
CONTROL = Path("/raid/jgu7/state-study-validation512")
ARMS = ("M_NT", "M_ST")
IDS = tuple(range(512, 1024))
BINS = ((1,32),(33,128),(129,256),(257,512),(513,1024),(1025,2047))
EPS = {"eps_lda_std":1e-6,"eps_variance":0.0,
       "eps_diagnostic_norm":1e-12,"eps_diagnostic_moment":1e-24}
SPEC = {
    "arms":list(ARMS),"sample_ids":list(IDS),"layer_ids":list(range(48)),"hidden_size":2048,
    "trace_scope":"formal/validate",
    "trace_paths":{arm:f"formal/validate/generations/{arm}.jsonl" for arm in ARMS},
    "calibration_paths":{arm:f"formal/calibration/{arm}.json" for arm in ("M_N","L_N",*ARMS)},
    "position_bins":[list(x) for x in BINS],"family":"lda","all_valid_decode":True,
    "eos_policy":"observe trace[:-1]; explicit empty marker iff T=1; never consume emitted EOS",
    "numerical_protection":EPS,"statistics_dtype":"float64","ddof":0,
    "bootstrap_resamples":10000,"bootstrap_seed":20261019,"weightings":["token","prompt"],
    "reuse_original128":False,"duplicate_first128_replay_cost_reported_separately":True,
    "no_generation_or_fitting":True,"requires_P4_gate":True,
    "quality_final_required_before_collection":False,
    "completion_independent_of_effect_direction_or_CI_estimability":True,
}
SUPPLEMENT_FILES = (
    "scripts/state_study_validation512.py", "scripts/state_study_validation512_report.py",
    "docs/2026-10-19/state_validation512.md", "docs/2026-10-19/state_validation512_reporting.md",
    "docs/2026-10-19/state_validation512_scope_audit.md",
)


class Pending(RuntimeError):
    pass


class Invalid(RuntimeError):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return quality.canonical(value)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    def reject(value):
        raise Invalid("Non-finite JSON: "+value)
    value=json.loads(Path(path).read_bytes(),parse_constant=reject)
    if not isinstance(value,dict):raise Invalid("Expected JSON object: "+str(path))
    return value


def sealed(value, field):
    return {**value,field:canonical(value)}


def verify_seal(value, field):
    if value.get(field)!=canonical({k:v for k,v in value.items() if k!=field}):
        raise Invalid("Self hash differs: "+field)


def plan_path(run, *, reports=REPORTS):
    return Path(reports)/Path(run).name/"validation512/plan.json"


def output_dir(run, plan, *, reports=REPORTS):
    return Path(reports)/Path(run).name/"validation512"/plan["plan_hash"]


def _draft_context(run, *, repo=REPO, reports=REPORTS):
    protocol,source=quality.sources(run,repo)
    path=Path(reports)/Path(run).name/"quality_plan.draft.json"
    if not path.is_file():raise Pending("Published valid quality draft is required")
    draft=quality.load(path)
    quality.verify_self(draft,"DRAFT_NOT_FROZEN");quality.verify_source_archive(draft)
    if (draft.get("source")!=source or draft.get("protocol_hash")!=protocol["protocol_hash"]
        or draft.get("run_dir")!=str(Path(run).resolve())
        or draft.get("formal_selection_output_absent_at_draft") is not True):
        raise Invalid("Original quality draft identity differs")
    return protocol,source,{"path":str(path.resolve()),"sha256":sha(path),"plan_hash":draft["plan_hash"]}


def _supplement_source(repo=REPO):
    hashes={}
    for name in SUPPLEMENT_FILES:
        path=Path(repo)/name
        if not path.is_file():raise Pending("Supplement source not ready: "+name)
        hashes[name]=sha(path)
    digest=canonical(hashes)
    return {"analysis_source_hashes":hashes,"analysis_source_hash":digest,
            "analysis_archive_relative_dir":"source_archives/"+digest}


def freeze_plan(run, target=None, *, repo=REPO, reports=REPORTS):
    run=Path(run).resolve();target=Path(target) if target else plan_path(run,reports=reports)
    target=target.resolve();allowed=(Path(reports)/run.name/"validation512").resolve()
    if (not target.is_relative_to(allowed) or target==allowed or target.is_relative_to(run)
        or target.is_relative_to(Path.home().resolve())):
        raise Invalid("Plan publication must stay in independent scratch reports/<run>/validation512, outside original run/Home")
    if target.exists():return load_plan(run,target,repo=repo,reports=reports)
    protocol,source,draft=_draft_context(run,repo=repo,reports=reports)
    absence=quality.validation_absence(run)
    supplement=_supplement_source(repo)
    archive=quality.publish_analysis_sources(target.parent,supplement,repo=repo)
    value=sealed({"schema_version":"state_study_validation512.plan.v1",
        "status":"FROZEN_PROSPECTIVE_RANGE_SUPPLEMENT","created_at_utc":utc(),
        "run_dir":str(run),"protocol_hash":protocol["protocol_hash"],"source":source,
        "quality_draft":draft,"spec":SPEC,"supplement_source":supplement,
        "supplement_source_archive":archive,"prevalidation_absence":absence,
        "scope":"Complete original section9.4 D_validate512 own-state for both updated arms; no changed method or thresholds",
        "output_layout":"reports/<run>/validation512/<plan_hash>/",
        "P6_authorized":False},"plan_hash")
    quality.validation_absence(run)
    quality.publish_once(target,value)
    return value


def load_plan(run, plan=None, *, repo=REPO, reports=REPORTS):
    """Return a strictly verified dict; absence is Pending, mismatch Invalid."""
    run=Path(run).resolve();path=Path(plan) if plan else plan_path(run,reports=reports)
    if not path.is_file():raise Pending("Prospective validation512 plan has not been sealed")
    value=load(path);verify_seal(value,"plan_hash")
    if (value.get("schema_version")!="state_study_validation512.plan.v1"
        or value.get("status")!="FROZEN_PROSPECTIVE_RANGE_SUPPLEMENT"
        or value.get("run_dir")!=str(run) or value.get("spec")!=SPEC
        or not value.get("prevalidation_absence",{}).get("absence_verified")):
        raise Invalid("Supplement schema/scope/specification/precedence differs")
    protocol,source,draft=_draft_context(run,repo=repo,reports=reports)
    if (value.get("protocol_hash")!=protocol["protocol_hash"] or value.get("source")!=source
        or value.get("quality_draft")!=draft or value.get("supplement_source")!=_supplement_source(repo)):
        raise Invalid("Supplement source/protocol/quality draft differs")
    quality.verify_source_archive({"source":value["supplement_source"],
        "analysis_source_archive":value["supplement_source_archive"]})
    quality.validation_mtime_guard(run,value["created_at_utc"])
    return value


def _error_guard(run, manifest):
    from state_study_continuation import scientific_error,ContinuationRefused
    try:errors=scientific_error(manifest,run)
    except ContinuationRefused as exc:raise Invalid("Malformed scientific manifest: "+str(exc)) from exc
    if errors:raise Invalid("Current scientific error: "+json.dumps(errors,ensure_ascii=False))


def _p4_trigger(run, plan):
    """Never turn missing/PENDING P4 into NOT_REQUIRED."""
    manifest=load(Path(run)/"manifest.json");_error_guard(run,manifest)
    stage=manifest.get("stages",{}).get("P4",{}).get("status")
    if stage not in ("complete","scientifically_not_triggered"):return None
    from summarize_state_study_results import Evidence,load_diagnostic,load_calibrations
    from state_study_positive_evidence import diagnostic_identities,p3_numerical_gate,bootstrap_contract
    evidence=Evidence(run,verify_diagnostic_hashes=True)
    diagnostic=load_diagnostic(evidence,"formal")
    summary=diagnostic.get("summary") or {}
    if summary.get("expected_prompt_ids")!=[str(i) for i in range(128,256)]:
        raise Invalid("Formal P3 expected128 prompt universe differs")
    if (manifest.get("stages",{}).get("P3",{}).get("status")!="complete"
        or not bootstrap_contract(summary) or summary.get("family")!="lda"
        or summary.get("chosen_mismatch_metric")!="E_moment"
        or not all(summary.get("prompt_completeness",{}).get(arm,{}).get("complete_expected_ids") is True for arm in ("A","B","C","D"))):
        raise Invalid("Formal P3 is not complete under the declared bootstrap/family/prompt contract")
    raw={}
    for scope,arms in (("formal/diag_rollouts",("N","L_N")),
                       ("formal/updated_diag_rollouts",ARMS)):
        raw[scope]={arm:read_jsonl(Path(run)/scope/"generations"/(arm+".jsonl"),allow_missing=True)
                    for arm in arms}
    # The locked loader verifies each complete payload's self hash and protocol;
    # archive agreement with a merely asserted hash is not sufficient.
    calibrations=load_calibrations(evidence,plan["protocol_hash"])["arms"]
    proof=diagnostic_identities(evidence,diagnostic,raw,calibrations,"formal",("A","B","C","D"))
    if proof.get("status")!="VERIFIED" or evidence.errors:
        raise Invalid("Formal P3 archive/driver/calibration/coverage identity is not verified")
    if stage=="scientifically_not_triggered":
        if summary.get("gate_recalibration") is not False or summary.get("status") not in ("NEGATIVE","INCONCLUSIVE"):
            raise Invalid("P4/P3 no-trigger contradicts")
        return False
    protocol=load(Path(run)/"protocol.json")
    if not p3_numerical_gate(diagnostic,protocol):raise Invalid("P4 complete lacks the original numerical P3 gate")
    return True


def inspect_status(run, plan=None, *, repo=REPO, reports=REPORTS):
    """Pure CPU prerequisites, independent of completion receipt/CI direction."""
    run=Path(run).resolve()
    try:
        value=load_plan(run,plan,repo=repo,reports=reports)
        directory=output_dir(run,value,reports=reports)
        if (directory/"manifest.json").is_file():
            supplement_manifest=load(directory/"manifest.json")
            if supplement_manifest.get("plan_hash")!=value["plan_hash"]:raise Invalid("Supplement manifest plan differs")
            _error_guard(run,supplement_manifest)
        manifest=load(run/"manifest.json");_error_guard(run,manifest)
        core=(manifest.get("stages",{}).get("P5",{}).get("status")=="complete"
              and manifest.get("stages",{}).get("P6_P7",{}).get("status")=="gated_pending"
              and (run/"RUN_REPORT.md").is_file() and (run/"RUN_REPORT.md").stat().st_size>0)
        result={"plan_hash":value["plan_hash"],"output_dir":str(directory),
            "diagnostics_dir":str(directory/"diagnostics"),"summary_path":str(directory/"summaries/diagnostic_summary.json"),
            "core_complete":core,"expected_prompt_ids":list(IDS),"arms":list(ARMS)}
        triggered=_p4_trigger(run,value)
        if triggered is None:return {**result,"status":"WAITING_P4","reason":"P4 is not decided"}
        if not triggered:return {**result,"status":"NOT_REQUIRED","reason":"Verified original P3 did not trigger P4; not a negative-result inference"}
        if not core:return {**result,"status":"WAITING_CORE","reason":"Original P0-P5 has not completed"}
        return {**result,"status":"READY","reason":"Core complete; collection still requires registered same-PID GPU launch"}
    except Pending as exc:return {"status":"PLAN_PENDING","reason":str(exc)}
    except (Invalid,quality.Invalid,quality.Pending,OSError,ValueError,KeyError) as exc:
        return {"status":"SCIENTIFIC_ERROR" if "scientific error" in str(exc).lower() else "INVALID_PLAN","reason":str(exc)}


def read_jsonl(path, *, allow_missing=False):
    path=Path(path)
    if not path.is_file():
        if allow_missing:return []
        raise Pending("Required raw file is missing: "+str(path))
    rows=[];seen=set()
    with path.open("rb") as handle:
        for raw in handle:
            if not raw.endswith(b"\n"):raise Invalid("Incomplete physical JSONL record: "+str(path))
            row=json.loads(raw,parse_constant=lambda value:(_ for _ in ()).throw(Invalid("Non-finite JSONL: "+value)))
            if not isinstance(row,dict):raise Invalid("JSONL record is not an object")
            pid=row.get("sample_id")
            if isinstance(pid,bool) or not isinstance(pid,int) or pid in seen:raise Invalid("Duplicate/invalid raw sample ID")
            seen.add(pid);rows.append(row)
    return rows


def status(run, plan=None, *, repo=REPO, reports=REPORTS):
    result=inspect_status(run,plan,repo=repo,reports=reports)
    if result["status"]=="NOT_REQUIRED":
        report=importlib.import_module("state_study_validation512_report")
        delivery=report.verify_delivery(run,plan_path=plan)
        return {**result,"delivery_complete":delivery.get("verified") is True and delivery.get("delivery_complete") is True,"delivery_proof":delivery}
    if result["status"]!="READY":return result
    directory=Path(result["output_dir"])
    if (directory/"completion.json").is_file():
        report=importlib.import_module("state_study_validation512_report")
        proof=report.check_validation512(run,plan_path=plan,verify_hashes=True)
        if proof.get("complete") is not True:return {**result,"status":"INVALID_PLAN","completion_proof":proof}
        delivery=report.verify_delivery(run,plan_path=plan,work_proof=proof)
        delivered=delivery.get("verified") is True and delivery.get("delivery_complete") is True
        return {**result,"status":"COMPLETE" if delivered else "RAW_COMPLETE",
            "completion_proof":proof,"delivery_complete":delivered,"delivery_proof":delivery}
    if (directory/"raw_completion.json").is_file():
        try:
            value=load_plan(run,plan,repo=repo,reports=reports)
            snapshot,rows,calibrations=source_snapshot(run,value)
            verify_raw_completion(directory,value,snapshot,rows,calibrations)
            return {**result,"status":"RAW_COMPLETE"}
        except (Invalid,Pending,OSError,ValueError,KeyError) as exc:
            return {**result,"status":"INVALID_PLAN","reason":"Raw completion verification: "+str(exc)}
    return result


def _calibrations(run, protocol):
    """Payload hash, precision/formula, 48-layer shape and original fit masks."""
    from summarize_state_study_results import Evidence,load_calibrations
    from state_study_positive_evidence import calibration_provenance
    evidence=Evidence(run,verify_diagnostic_hashes=True)
    records=load_calibrations(evidence,protocol["protocol_hash"])["arms"]
    if set(records)!={"M_N","L_N",*ARMS} or evidence.errors:
        raise Invalid("Calibration self hash/protocol/formula: "+str(evidence.errors))
    payloads={arm:load(Path(run)/f"formal/calibration/{arm}.json") for arm in records}
    for arm,value in payloads.items():
        if value.get("numerical_protection")!=EPS or value.get("statistics_dtype")!="float64" or value.get("ddof")!=0:
            raise Invalid("Calibration numerical contract differs: "+arm)
        params=value.get("parameters",value.get("params",{}))
        if set(params)!={str(i) for i in range(48)}:raise Invalid("Calibration layer rectangle differs: "+arm)
        if arm in ("L_N",*ARMS):
            if value.get("family")!="lda" or value.get("reduced_branch")!="y4_full_renorm":raise Invalid("Updated LDA branch differs")
            for layer,record in params.items():
                for key in ("mean_native","mean_reduced","std_native","std_reduced"):
                    vector=record.get(key)
                    if (not isinstance(vector,list) or len(vector)!=2048
                        or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in vector)
                        or (key.startswith("std") and any(x<0 for x in vector))):
                        raise Invalid(f"Invalid LDA vector {arm}/{layer}/{key}")
    raw={scope:{arm:read_jsonl(Path(run)/scope/"generations"/(arm+".jsonl"))}
         for scope,arm in (("formal/fit_native","N"),("formal/fit_student","L_N"))}
    proof=calibration_provenance(evidence,raw,payloads)
    if proof.get("status")!="VERIFIED" or evidence.errors:raise Invalid("Original calibration count/mask/source provenance differs: "+str(proof))
    return payloads


def dataset_samples(protocol):
    config=protocol["config"];directory=Path(config["data_dir"])
    manifest=load(directory/"data_manifest.json");path=directory/"train.jsonl"
    if manifest!=protocol.get("dataset") or sha(path)!=manifest["sources"]["train"]["jsonl_sha256"]:
        raise Invalid("Pinned dataset manifest/train bytes differ")
    rows=read_jsonl(path);samples={r["sample_id"]:r for r in rows}
    for sid in IDS:
        row=samples.get(sid,{})
        if (row.get("dataset_revision")!=config["dataset_revision"] or row.get("split")!="train"
            or row.get("question_hash")!=hashlib.sha256(row.get("question","").encode()).hexdigest()):
            raise Invalid("Pinned sample identity differs: "+str(sid))
    return samples


def validate_trace_rows(rows,arm,protocol,calibrations,samples):
    """Actual observer-free complete generation, including the final EOS token."""
    from state_study_positive_evidence import arm_hash
    mapped={r["sample_id"]:r for r in rows}
    if len(mapped)!=len(rows) or set(mapped)!=set(IDS):raise Invalid("Validation trace IDs are not the full512: "+arm)
    config=protocol["config"];expected={"arm":arm,"arm_hash":arm_hash(arm,calibrations),
        "calibration_hash":calibrations[arm]["calibration_hash"],"protocol_hash":protocol["protocol_hash"],
        "parser_version":protocol["parser_version"],"model_revision":config["model_revision"],
        "dataset_revision":config["dataset_revision"],"scope":"formal/validate",
        "max_new_tokens":2048,"observer_enabled":False,"timing_debug_logits":False}
    for sid,row in mapped.items():
        if any(row.get(k)!=v for k,v in expected.items()):
            raise Invalid("Saved free-generation identity differs: "+arm+"/"+str(sid)+" "+str([k for k,v in expected.items() if row.get(k)!=v]))
        sample=samples[sid];trace=row.get("gen_token_ids");eos=row.get("eos_ids")
        if (row.get("question")!=sample["question"] or row.get("question_hash")!=sample["question_hash"]
            or not isinstance(trace,list) or not 1<=len(trace)<=2048
            or any(isinstance(t,bool) or not isinstance(t,int) or t<0 for t in trace)
            or not isinstance(eos,list) or not eos or any(t in eos for t in trace[:-1])
            or row.get("generated_tokens")!=len(trace) or row.get("decode_forward_count")!=len(trace)-1):
            raise Invalid("Saved trace/sample/T-1/EOS contract differs: "+arm+"/"+str(sid))
        stopped=trace[-1] in eos
        if (row.get("eos_position")!=(len(trace)-1 if stopped else None)
            or row.get("stop_reason")!=("eos" if stopped else "max_new_tokens")
            or (not stopped and len(trace)!=2048)):
            raise Invalid("Saved EOS/cap terminal event differs")
        prompt=row.get("prompt_token_ids")
        if (not isinstance(prompt,list) or not prompt or any(isinstance(t,bool) or not isinstance(t,int) or t<0 for t in prompt)
            or row.get("prompt_tokens")!=len(prompt) or not isinstance(row.get("rendered_prompt"),str)):
            raise Invalid("Saved prompt identity is incomplete")
        if any(not isinstance(row.get(k),bool) for k in ("strict_correct","tolerant_correct")):
            raise Invalid("Saved parser correctness is not boolean")
    return mapped


def source_snapshot(run,plan):
    """Deterministic CPU source proof, rows-by-arm-by-int-id, full calibration JSON.

    Missing core work raises Pending; contradictory committed sources raise
    Invalid. A final quality plan is optional for state collection; its absence
    cannot be converted into a positive quality claim by this supplement.
    """
    run=Path(run).resolve();manifest=load(run/"manifest.json");_error_guard(run,manifest)
    from summarize_state_study_results import verify_launch_process
    if verify_launch_process(run,load(run/"launch.json")).get("status")=="VERIFIED_LIVE":
        raise Pending("Original exact core PID is still live")
    stages=manifest.get("stages",{})
    if stages.get("P5",{}).get("status")!="complete" or not (run/"RUN_REPORT.md").is_file():
        raise Pending("Original P5/core report is not complete")
    if _p4_trigger(run,plan) is not True:raise Pending("Original P4 update has not completed")
    protocol=load(run/"protocol.json")
    frozen=load(run/"validation_freeze.json")
    if (frozen.get("protocol_hash")!=plan["protocol_hash"] or frozen.get("updated") is not True
        or frozen.get("validation_sample_ids")!=list(IDS)
        or not set(ARMS).issubset(frozen.get("finalist_arms",[]))
        or frozen.get("quality_noninferiority_margin") is not None):
        raise Invalid("Original prospective validation selection/range differs")
    samples=dataset_samples(protocol);calibrations=_calibrations(run,protocol)
    rows={arm:validate_trace_rows(read_jsonl(run/SPEC["trace_paths"][arm]),arm,protocol,calibrations,samples) for arm in ARMS}
    expected_eos=manifest.get("runtime",{}).get("eos_ids")
    if not expected_eos or any(row.get("eos_ids")!=expected_eos for arm in ARMS for row in rows[arm].values()):
        raise Invalid("Saved validation EOS IDs differ from actual core runtime")
    common=("question_hash","rendered_prompt","prompt_token_ids","prompt_tokens","eos_ids")
    if any(rows[ARMS[0]][sid].get(k)!=rows[ARMS[1]][sid].get(k) for sid in IDS for k in common):
        raise Invalid("Updated arms did not use identical validation prompts/EOS configuration")
    final_path=Path(plan["quality_draft"]["path"]).with_name("quality_plan.json")
    if final_path.is_file():
        final_value=quality.load(final_path);quality.verify_plan(run,final_value)
        if final_value.get("draft_hash")!=plan["quality_draft"]["plan_hash"]:raise Invalid("Final quality plan has another draft")
    from state_study_positive_evidence import arm_hash
    snapshot={"schema_version":"state_study_validation512.sources.v1","plan_hash":plan["plan_hash"],
        "protocol_hash":plan["protocol_hash"],"source_hash":protocol["source_hash"],
        "quality_draft_hash":plan["quality_draft"]["plan_hash"],
        "quality_final_binding":"Independent positive-chain report verifies final; it is not mutable state-collection source identity",
        "validation_freeze_sha256":sha(run/"validation_freeze.json"),
        "dataset_train_sha256":protocol["dataset"]["sources"]["train"]["jsonl_sha256"],
        "calibrations":{arm:{"relative_path":SPEC["calibration_paths"][arm],
            "sha256":sha(run/SPEC["calibration_paths"][arm]),"calibration_hash":value["calibration_hash"]} for arm,value in calibrations.items()},
        "traces":{arm:{"relative_path":SPEC["trace_paths"][arm],"sha256":sha(run/SPEC["trace_paths"][arm]),
            "bytes":(run/SPEC["trace_paths"][arm]).stat().st_size,"sample_ids":list(IDS),
            "driver_hash":arm_hash(arm,calibrations),"calibration_hash":calibrations[arm]["calibration_hash"],
            "trace_hashes":{str(sid):canonical(rows[arm][sid]["gen_token_ids"]) for sid in IDS},
            "decode_counts":{str(sid):len(rows[arm][sid]["gen_token_ids"])-1 for sid in IDS},
            "total_decode_tokens":sum(len(r["gen_token_ids"])-1 for r in rows[arm].values())} for arm in ARMS}}
    return sealed(snapshot,"snapshot_hash"),rows,calibrations


def trace_identity(sid,arm,row,calibrations,protocol_hash):
    from state_study_positive_evidence import arm_hash
    trace=row["gen_token_ids"]
    return {"protocol_hash":protocol_hash,"driver_hash":arm_hash(arm,calibrations),
            "calibration_hash":calibrations[arm]["calibration_hash"],"trace_hash":canonical(trace)}


def audit_archive(directory,sid,arm,genrow,calibrations,protocol_hash):
    """Verify durable per-prompt bytes and actual full/bin positions; no bootstrap.

    Files in deterministic receipt are relative to the supplement output root.
    Missing/uncommitted work is Pending; any contradictory commit is Invalid.
    """
    import numpy as np
    directory=Path(directory);data=directory/"diagnostics";stem=canonical([str(sid),arm])
    identity=trace_identity(sid,arm,genrow,calibrations,protocol_hash)
    trace=genrow["gen_token_ids"];positions=[[j,t] for j,t in enumerate(trace[:-1],1)]
    count=len(positions);mask=canonical(positions);empty=data/(stem+".empty.json")
    archive=data/(stem+".npz");tokens=data/(stem+".tokens.jsonl");commit_path=data/(stem+".complete.json")
    if count==0:
        if not empty.is_file():raise Pending(f"Missing immediate-EOS marker {arm}/{sid}")
        value=load(empty)
        expected={"schema_version":"state_collection.empty.v1","collection_schema_version":"state_collection.v2",
            "prompt_id":str(sid),"path":arm,"count":0,"dtype":"float64","ddof":0,
            "position_bins":[list(x) for x in BINS],"numerical_protection":EPS,"metadata":identity}
        if (any(value.get(k)!=v for k,v in expected.items()) or any(p.exists() for p in (archive,tokens,commit_path))
            or not isinstance(value.get("reason"),str) or not value["reason"]):
            raise Invalid(f"Empty marker identity/coverage conflicts {arm}/{sid}")
        files=[empty];kind="empty";layer_count=0;token_rows=0
    else:
        if not commit_path.is_file():raise Pending(f"Missing archive commit {arm}/{sid}")
        commit=load(commit_path)
        if (empty.exists() or commit.get("schema_version")!="state_collection.commit.v1"
            or commit.get("collection_schema_version")!="state_collection.v2"
            or commit.get("archive")!=archive.name or commit.get("tokens")!=tokens.name
            or not archive.is_file() or not tokens.is_file()
            or commit.get("archive_sha256")!=sha(archive) or commit.get("tokens_sha256")!=sha(tokens)
            or commit.get("token_rows")!=48*count):raise Invalid(f"Archive pair/commit hash/row count differs {arm}/{sid}")
        with tokens.open("rb") as handle:
            lines=sum(1 for line in handle if line.endswith(b"\n"))
        if lines!=48*count:raise Invalid("Scalar token rows differ from actual48*T-1")
        with np.load(archive,allow_pickle=False) as arrays:
            metadata=json.loads(str(arrays["metadata_json"].item()))
            expected={"schema_version":"state_collection.v2","dtype":"float64","ddof":0,
                "moment_representation":"count_mean_centered_m2","position_bins":[list(x) for x in BINS],"numerical_protection":EPS}
            if any(metadata.get(k)!=v for k,v in expected.items()):raise Invalid("NPZ numerical contract differs")
            groups={"all":positions}
            groups.update({f"{lo}-{hi}":[p for p in positions if lo<=p[0]<=hi] for lo,hi in BINS})
            groups={k:v for k,v in groups.items() if v};seen=set()
            for record in metadata.get("records",[]):
                key=record.get("key",[])
                if len(key)!=4 or key[0]!=str(sid) or key[1]!=arm or type(key[2]) is not int or key[2] not in range(48) or key[3] not in groups:
                    raise Invalid("Unexpected NPZ prompt/path/layer/group")
                group=groups[key[3]];position_array=arrays[record["prefix"]+"_positions"]
                if (tuple(key) in seen or record.get("count")!=len(group) or record.get("mask_hash")!=canonical(group)
                    or record.get("driver_hash")!=identity["driver_hash"] or record.get("calibration_hash")!=identity["calibration_hash"]
                    or position_array.dtype!=np.dtype("int64") or position_array.tolist()!=group):raise Invalid("NPZ actual trace/layer/bin identity differs")
                seen.add(tuple(key))
                for suffix in ("native_mean","native_m2","calibrated_mean","calibrated_m2"):
                    array=arrays[record["prefix"]+"_"+suffix]
                    if array.dtype!=np.dtype("float64") or array.shape!=(2048,) or not np.isfinite(array).all():
                        raise Invalid("NPZ moment shape/dtype/finiteness differs")
            if seen!={(str(sid),arm,layer,group) for layer in range(48) for group in groups}:
                raise Invalid("NPZ full48 layer/bin rectangle differs")
        files=[archive,tokens,commit_path];kind="archive";layer_count=48;token_rows=48*count
    return {"sample_id":sid,"path":arm,"kind":kind,"mask_hash":mask,"decode_forwards":count,
        "layer_count":layer_count,"token_rows":token_rows,"identity":identity,
        "files":[{"relative_path":str(p.relative_to(directory)),"sha256":sha(p),"bytes":p.stat().st_size} for p in files]}


def replay_check(sid,arm,row,calibrations,protocol_hash,result):
    trace=row["gen_token_ids"];identity=trace_identity(sid,arm,row,calibrations,protocol_hash)
    expected={"argmaxes":trace,"decode_forwards":len(trace)-1,
              "mask_hash":canonical([[j,t] for j,t in enumerate(trace[:-1],1)])}
    if any(result.get(k)!=v for k,v in expected.items()) or not result.get("logits_finiteness_check"):
        raise Invalid(f"Own replay argmax/mask/finiteness mismatch {arm}/{sid}")
    for key,value in {"sample_id":sid,"path":arm,**identity}.items():
        if key in result and result[key]!=value:raise Invalid("Durable replay check has contradictory identity: "+key)
    return {"sample_id":sid,"path":arm,**expected,
            "logits_finiteness_check":result["logits_finiteness_check"],**identity}


def publish_equal(path,value):
    path=Path(path)
    if path.is_file():
        if load(path)!=value:raise Invalid("Immutable output differs: "+str(path))
    else:quality.publish_once(path,value)


def raw_receipt(directory,plan,snapshot,rows,calibrations):
    directory=Path(directory)
    if load(directory/"source_snapshot.json")!=snapshot:raise Invalid("Saved source snapshot differs")
    checks=[]
    # IDs repeat across arms in this aggregate, so use pair keys explicitly.
    check_path=directory/"replay_checks.jsonl"
    if not check_path.is_file():raise Pending("All1024 replay-check aggregate is not committed")
    with check_path.open("rb") as handle:
        for line in handle:
            if not line.endswith(b"\n"):raise Invalid("Truncated replay checks")
            checks.append(json.loads(line))
    expected=[];archives=[]
    for arm in ARMS:
        for sid in IDS:
            per_prompt=load(directory/"replay_checks"/(canonical([str(sid),arm])+".json"))
            expected.append(replay_check(sid,arm,rows[arm][sid],calibrations,plan["protocol_hash"],per_prompt))
            archives.append(audit_archive(directory,sid,arm,rows[arm][sid],calibrations,plan["protocol_hash"]))
    if checks!=expected:raise Invalid("Replay-check aggregate differs from per-prompt verified checks")
    return sealed({"schema_version":"state_study_validation512.raw.v1","plan_hash":plan["plan_hash"],
        "source_snapshot_hash":snapshot["snapshot_hash"],"expected_prompt_ids":list(IDS),"arms":list(ARMS),
        "archives":archives,"replay_checks_sha256":sha(check_path)},"raw_completion_hash")


def verify_raw_completion(directory,plan,snapshot,rows,calibrations):
    receipt=load(Path(directory)/"raw_completion.json");verify_seal(receipt,"raw_completion_hash")
    if receipt!=raw_receipt(directory,plan,snapshot,rows,calibrations):raise Invalid("Raw completion no longer matches actual committed bytes")
    return receipt


def registered_launch(run,plan,*,reports=REPORTS,proc=Path("/proc"),environment=None,resume=None):
    """Require the exclusive shared ops handoff, actual allocation and same PID.

    A local flock is supplementary only. Remote predecessor death is established
    by current positive Slurm terminal evidence, never missing local /proc.
    This pure CPU guard runs before importing any model/GPU module.
    """
    if resume is None:resume=importlib.import_module("launch_state_study_resume")
    environment=os.environ if environment is None else environment
    run=Path(run).resolve();ops=Path(reports)/run.name/"validation512/ops"
    path=ops/"launch.json"
    if not path.is_file():raise Invalid("Unregistered supplement run: exclusive ops handoff launch is required")
    launch=load(path);core_path=run/"launch.json";core=load(core_path)
    expected_command=[sys.executable,"-u",str(REPO/"scripts/state_study_validation512.py"),"run","--run-dir",str(run)]
    protocol=load(run/"protocol.json")
    expected={"schema_version":"state_validation512_launch.v1","task":"validation512","phase":"validation512",
        "run_dir":str(run),"plan_hash":plan["plan_hash"],"supplement_plan_hash":plan["plan_hash"],
        "protocol_hash":plan["protocol_hash"],"source_hash":protocol["source_hash"],"pid":os.getpid(),
        "hostname":socket.gethostname(),"command":expected_command,"core_launch_sha256":sha(core_path),
        "metadata_written_before_exec":True,"exec_preserves_pid":True}
    if any(launch.get(k)!=v for k,v in expected.items()):raise Invalid("Registered supplement launch/source/command differs")
    observed=resume._proc_stat((Path(proc)/"self/stat").read_text())
    boot=(Path(proc)/"sys/kernel/random/boot_id").read_text().strip()
    cgroup=(Path(proc)/"self/cgroup").read_text()
    actual_command=(Path(proc)/"self/cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
    if (str(observed["start_ticks"])!=str(launch.get("proc_start_ticks")) or boot!=launch.get("boot_id")
        or cgroup!=launch.get("cgroup") or actual_command!=expected_command
        or str(launch.get("slurm_job_id"))!=environment.get("SLURM_JOB_ID")):
        raise Invalid("Registered supplement exact PID/boot/ticks/cgroup/argv/allocation differs")
    previous=str(launch.get("previous_slurm_job_id",""));core_job=str(core.get("slurm_job_id",""))
    for job,proof_key in ((previous,"predecessor_terminal_proof"),(core_job,"core_terminal_proof")):
        proof=launch.get(proof_key,{})
        if not job.isdigit() or proof.get("status")!="VERIFIED_TERMINAL" or str(proof.get("job_id"))!=job:
            raise Invalid("Registered terminal proof does not bind actual predecessor/core")
        if resume.predecessor_terminal(job).get("status")!="VERIFIED_TERMINAL":
            raise Invalid("Predecessor/core Slurm state is currently unknown or active")
    if resume.verify_launch_process(run,core).get("status")=="VERIFIED_LIVE":raise Invalid("Original exact core PID is live")
    if not resume.completed(run,load(run/"manifest.json")):raise Pending("Original core is incomplete")
    _error_guard(run,load(run/"manifest.json"));resume.validate_sources(run)
    allocation=resume.current_allocation(previous,environment=environment,proc_root=Path(proc))
    resume.validate_allocation_resources(allocation)
    if allocation.get("cgroup")!=cgroup or str(allocation.get("job_id"))!=str(launch["slurm_job_id"]):
        raise Invalid("Actual allocation proof differs from registered launch")
    reference=launch.get("shared_handoff_claim",{});claim_path=Path(reference.get("path","/missing"))
    expected_claim_path=ops/"claims"/(launch.get("previous_launch_sha256","")+".json")
    if claim_path.resolve()!=expected_claim_path.resolve() or not claim_path.is_file() or sha(claim_path)!=reference.get("sha256"):
        raise Invalid("Exclusive shared handoff claim is missing/changed/outside exact ops path")
    claim=load(claim_path)
    expected_claim={"schema_version":"state_validation512_handoff_claim.v1","task":"validation512",
        "run_dir":str(run),"plan_hash":plan["plan_hash"],"protocol_hash":plan["protocol_hash"],
        "owner_uid":os.getuid(),"pid":os.getpid(),"hostname":socket.gethostname(),"boot_id":boot,
        "proc_start_ticks":observed["start_ticks"],"slurm_job_id":allocation["job_id"],
        "previous_launch_sha256":launch["previous_launch_sha256"]}
    if any(claim.get(k)!=v for k,v in expected_claim.items()):raise Invalid("Shared handoff claim identity differs")
    history=Path(launch.get("previous_launch_history","/missing"))
    if (not history.resolve().is_relative_to((ops/"launch_history").resolve())
        or not history.is_file() or sha(history)!=launch["previous_launch_sha256"]):
        raise Invalid("Exclusive predecessor history bytes no longer match claim")
    return {"launch_path":str(path),"launch_sha256":sha(path),"claim_sha256":sha(claim_path),
        "allocation":allocation,"core_terminal_proof":launch["core_terminal_proof"],"pid":os.getpid()}


def local_filesystem(path,*,query=subprocess.run):
    """Positive kernel mount evidence before placing any active lock/control."""
    probe=Path(path).resolve()
    while not probe.exists():
        if probe.parent==probe:raise Invalid("No existing parent for local control mount proof")
        probe=probe.parent
    command=["findmnt","--json","--target",str(probe),"--output","TARGET,SOURCE,FSTYPE"]
    response=query(command,capture_output=True,text=True,timeout=10,check=False)
    if response.returncode!=0:raise Invalid("Cannot verify local control filesystem; no network lock fallback")
    try:
        mounts=json.loads(response.stdout)["filesystems"]
        if len(mounts)!=1:raise ValueError("ambiguous mount")
        mount=mounts[0]
    except (ValueError,KeyError,TypeError) as exc:raise Invalid("Malformed control filesystem proof") from exc
    if (mount.get("fstype") not in {"xfs","ext4","ext3","ext2","btrfs","zfs"}
        or (mount.get("fstype")!="zfs" and not str(mount.get("source","")).startswith("/dev/"))):
        raise Invalid("Control filesystem is not positively verified local disk: "+str(mount))
    return {"command":command,"mount":mount,"stdout":response.stdout,"verified_local_disk":True}


def local_lock(run,plan,*,control=CONTROL,fs_probe=local_filesystem):
    proof=fs_probe(control)
    if proof.get("verified_local_disk") is not True:raise Invalid("Local control filesystem proof is missing")
    directory=Path(control)/canonical([str(Path(run).resolve()),plan["plan_hash"]]);directory.mkdir(parents=True,exist_ok=True)
    handle=(directory/"active.lock").open("a+")
    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:handle.close();raise Invalid("A local supplement process already owns its lock")
    publish_equal(directory/(str(os.getpid())+"."+uuid.uuid4().hex+".json"),
        {"pid":os.getpid(),"hostname":socket.gethostname(),"created_at_utc":utc(),"run_dir":str(run),"plan_hash":plan["plan_hash"],
         "local_filesystem":proof})
    return handle


def write_manifest(directory,plan,stage,status_value,**details):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/"manifest.json";value=load(path) if path.is_file() else {
        "schema_version":"state_study_validation512.manifest.v1","plan_hash":plan["plan_hash"],"stages":{}}
    if value.get("plan_hash")!=plan["plan_hash"]:raise Invalid("Supplement manifest has another plan")
    value["stages"][stage]={"status":status_value,"at_utc":utc(),**details}
    value["status"]=status_value
    temporary=path.with_name("."+path.name+"."+uuid.uuid4().hex+".pending")
    try:
        with temporary.open("x") as handle:
            json.dump(value,handle,ensure_ascii=False,allow_nan=False,indent=2);handle.write("\n");handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:temporary.unlink(missing_ok=True)


def publish_jsonl(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=b"".join((json.dumps(row,sort_keys=True,ensure_ascii=False,allow_nan=False)+"\n").encode() for row in rows)
    if path.exists():
        if path.read_bytes()!=data:raise Invalid("Immutable replay-check aggregate differs")
        return
    temporary=path.with_name("."+path.name+"."+uuid.uuid4().hex+".pending")
    try:
        with temporary.open("xb") as handle:handle.write(data);handle.flush();os.fsync(handle.fileno())
        os.link(temporary,path)
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:temporary.unlink(missing_ok=True)


def _load_driver(protocol):
    """The sole model-loading entrypoint; called only after registered guards."""
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from moe_research.state_driver import StudyDriver
    config=protocol["config"]
    if (not torch.cuda.is_available() or torch.cuda.device_count()!=1 or "B200" not in torch.cuda.get_device_name(0)):
        raise Invalid("Actual run requires exactly one visible B200")
    if config.get("precision")!="bfloat16" or config.get("attention_backend")!="eager":raise Invalid("Frozen BF16/eager runtime differs")
    source=Path(config["model_path"]);provenance=load(source/"study_download_provenance.json")
    if provenance!=protocol["model"] or provenance.get("status")!="complete":raise Invalid("Actual pinned model provenance differs")
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(source,dtype=torch.bfloat16,
        attn_implementation=config["attention_backend"],device_map={"":"cuda:0"},local_files_only=True)
    model.eval();return StudyDriver(model,tokenizer)


def isolate_uncommitted(directory,stem):
    """Retain interrupted unpublished pairs away from frozen loader globbing."""
    directory=Path(directory)
    if (directory/(stem+".complete.json")).exists():return []
    remnants=[directory/(stem+suffix) for suffix in (".npz",".tokens.jsonl") if (directory/(stem+suffix)).exists()]
    if not remnants:return []
    quarantine=directory/"_uncommitted"/uuid.uuid4().hex;quarantine.mkdir(parents=True,exist_ok=False)
    names=[]
    for path in remnants:
        target=quarantine/path.name;os.rename(path,target);names.append(str(target))
    publish_equal(quarantine/"retained.json",{"reason":"No durable complete marker; preserve interrupted bytes before real replay",
        "files":[{"path":str(Path(p)),"sha256":sha(p)} for p in names],"at_utc":utc()})
    return names


def collect(run,plan,directory,snapshot,rows,calibrations,*,driver_factory=None):
    """Replay saved traces once per unfinished prompt; never generate or fit."""
    import torch
    from moe_research.state_calibration import FrozenStateCalibration
    from moe_research.state_collection import StateCollector
    from moe_research.state_adapter import shadow_branches
    run=Path(run);directory=Path(directory);protocol=load(run/"protocol.json")
    samples=dataset_samples(protocol);frozen={a:FrozenStateCalibration.from_dict(v) for a,v in calibrations.items()}
    checks=[];remaining=[]
    publish_equal(directory/"source_snapshot.json",snapshot)
    for arm in ARMS:
        for sid in IDS:
            check_path=directory/"replay_checks"/(canonical([str(sid),arm])+".json")
            try:
                audit_archive(directory,sid,arm,rows[arm][sid],calibrations,plan["protocol_hash"])
            except Pending:
                isolate_uncommitted(directory/"diagnostics",canonical([str(sid),arm]))
                remaining.append((sid,arm,False));continue
            if not check_path.is_file():
                # A legacy/interrupted commit is real state work, but its
                # argmax proof cannot be inferred from the expected trace.
                remaining.append((sid,arm,True));continue
            check=load(check_path);replay_check(sid,arm,rows[arm][sid],calibrations,plan["protocol_hash"],check)
    if not remaining:
        checks=[load(directory/"replay_checks"/(canonical([str(sid),arm])+".json")) for arm in ARMS for sid in IDS]
        publish_jsonl(directory/"replay_checks.jsonl",checks)
        receipt=raw_receipt(directory,plan,snapshot,rows,calibrations);publish_equal(directory/"raw_completion.json",receipt)
        return receipt
    driver=None;attempt=uuid.uuid4().hex;start=time.perf_counter()
    try:
        driver=(driver_factory or _load_driver)(protocol)
        publish_equal(directory/"cost_attempts"/(attempt+".model_load.json"),{
            "schema_version":"state_study_validation512.cost.v1","plan_hash":plan["plan_hash"],"operation":"model_load",
            "elapsed_seconds":time.perf_counter()-start,"started_at_utc":utc(),"pid":os.getpid()})
        if sorted(driver.eos)!=load(run/"manifest.json").get("runtime",{}).get("eos_ids"):
            raise Invalid("Supplement actual model EOS differs from core runtime")
        for sid,arm,proof_only in remaining:
            prompt_start=time.perf_counter();row=rows[arm][sid];trace=row["gen_token_ids"]
            text,encoded=driver.prompt(samples[sid])
            if text!=row["rendered_prompt"] or encoded[0].tolist()!=row["prompt_token_ids"]:
                raise Invalid("Saved generation and actual replay prompt differ")
            collector=StateCollector(directory/"diagnostics");calibration=frozen[arm]
            identity=trace_identity(sid,arm,row,calibrations,plan["protocol_hash"])
            def observer(block,h,output,phase,layer,step,route):
                if phase!="decode":return
                branches=shadow_branches(block,h);z=calibration.apply(layer,branches["y4_full"])
                if not torch.equal(z,output):raise Invalid(f"Actual driver/shadow mismatch {arm}/{sid}/{layer}/{step}")
                collector.observe(prompt_id=sid,path=arm,layer=layer,decode_forward_index=step,
                    token_id=trace[step-1],y8=branches["y8"],u4=branches["u4"],z4=z,
                    residual=driver.residuals[layer],driver_hash=identity["driver_hash"],calibration_hash=identity["calibration_hash"])
            if driver_factory is None:torch.cuda.synchronize()
            replay_start=time.perf_counter()
            result,_=driver.replay(samples[sid],trace,arm,frozen,None if proof_only else observer)
            if driver_factory is None:torch.cuda.synchronize()
            replay_seconds=time.perf_counter()-replay_start
            check=replay_check(sid,arm,row,calibrations,plan["protocol_hash"],result)
            # Check is durable before the collector publishes its commit-last
            # pair; a killed process cannot leave a finished, unchecked path.
            publish_equal(directory/"replay_checks"/(canonical([str(sid),arm])+".json"),check)
            if not proof_only:collector.finish_prompt(sid,arm,allow_empty=len(trace)==1,empty_metadata=identity)
            audit_archive(directory,sid,arm,row,calibrations,plan["protocol_hash"])
            publish_equal(directory/"cost_attempts"/(attempt+f".{arm}.{sid}.json"),{
                "schema_version":"state_study_validation512.cost.v1","plan_hash":plan["plan_hash"],
                "operation":"repair_missing_argmax_proof" if proof_only else "own_trace_replay",
                "sample_id":sid,"arm":arm,"decode_forwards":len(trace)-1,"replay_seconds_including_observer":replay_seconds,
                "prompt_wall_seconds_including_archive":time.perf_counter()-prompt_start,
                "duplicates_original128_scope":sid<640,"original_archive_not_reused":True,"at_utc":utc()})
            write_manifest(directory,plan,"collection","running",last_sample_id=sid,last_arm=arm)
            del collector
        checks=[load(directory/"replay_checks"/(canonical([str(sid),arm])+".json")) for arm in ARMS for sid in IDS]
        publish_jsonl(directory/"replay_checks.jsonl",checks)
        receipt=raw_receipt(directory,plan,snapshot,rows,calibrations);publish_equal(directory/"raw_completion.json",receipt)
        write_manifest(directory,plan,"collection","complete",raw_completion_hash=receipt["raw_completion_hash"],paths=1024)
        return receipt
    finally:
        if driver is not None:
            driver.close();del driver
        import gc
        gc.collect()
        if driver_factory is None:torch.cuda.empty_cache()


def analyze(run,plan,directory,*,analysis_fn=None):
    """Pure CPU fixed 10k analysis; complete work even if intervals undefined."""
    directory=Path(directory)
    if not (directory/"raw_completion.json").is_file():raise Pending("All1024 durable raw paths must complete before analysis")
    snapshot,rows,calibrations=source_snapshot(run,plan)
    receipt=verify_raw_completion(directory,plan,snapshot,rows,calibrations)
    completion_path=directory/"completion.json"
    if completion_path.is_file():
        value=load(completion_path);verify_seal(value,"completion_hash")
        if (value.get("raw_completion_hash")!=receipt["raw_completion_hash"]
            or value.get("summary_sha256")!=sha(directory/"summaries/diagnostic_summary.json")):
            raise Invalid("Existing completion source/summary differs")
        return value
    if analysis_fn is None:
        from moe_research.state_study_analysis import analyze_diagnostics
        analysis_fn=analyze_diagnostics
    config=load(Path(run)/"protocol.json")["config"];start=time.perf_counter();attempt=uuid.uuid4().hex
    write_manifest(directory,plan,"analysis","running",bootstrap_unit="prompt",bootstrap_resamples=10000)
    analysis_fn(directory/"diagnostics",directory/"summaries",iters=10000,family="lda",seed=20261019,
        native_match_tolerance=config["native_match_tolerance"],state_effect_tolerance=config["state_effect_tolerance"],
        expected_prompt_ids=list(IDS))
    summary=directory/"summaries/diagnostic_summary.json"
    if not summary.is_file():raise Invalid("Analyzer returned without publishing summary")
    publish_equal(directory/"cost_attempts"/(attempt+".analysis.json"),{"operation":"analysis_cpu",
        "plan_hash":plan["plan_hash"],"elapsed_seconds":time.perf_counter()-start,"iters":10000,
        "bootstrap_unit":"prompt","at_utc":utc(),"no_calibration_refitting":True})
    completion=sealed({"schema_version":"state_study_validation512.complete.v1","plan_hash":plan["plan_hash"],
        "source_snapshot_hash":snapshot["snapshot_hash"],"raw_completion_hash":receipt["raw_completion_hash"],
        "summary_sha256":sha(summary),"expected_prompt_ids":list(IDS),"arms":list(ARMS),"completed_work":True},"completion_hash")
    publish_equal(completion_path,completion);write_manifest(directory,plan,"analysis","complete",completion_hash=completion["completion_hash"])
    return completion


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=("freeze","check","status","run","analyze"))
    parser.add_argument("--run-dir",required=True);parser.add_argument("--plan")
    args=parser.parse_args(argv);run=Path(args.run_dir).resolve();directory=None;plan=None;lock=None
    try:
        if args.command=="freeze":result=freeze_plan(run,args.plan)
        elif args.command in ("status","check"):
            result=status(run,args.plan)
            print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
            return 0 if result["status"] in ("COMPLETE","NOT_REQUIRED","READY","RAW_COMPLETE") else 2 if result["status"] in ("PLAN_PENDING","WAITING_P4","WAITING_CORE") else 3
        else:
            plan=load_plan(run,args.plan);state=inspect_status(run,args.plan)
            if state["status"]=="NOT_REQUIRED":
                report=importlib.import_module("state_study_validation512_report")
                delivery=report.ensure_delivery(run,plan_path=args.plan)
                if delivery.get("verified") is not True or delivery.get("delivery_complete") is not True:raise Pending("NOT_REQUIRED delivery pending: "+str(delivery))
                print(json.dumps({**state,"delivery_complete":True,"delivery":delivery}),flush=True);return 0
            if state["status"]!="READY":raise Pending("Supplement prerequisites: "+state["status"])
            directory=output_dir(run,plan)
            if (directory/"manifest.json").is_file() and load(directory/"manifest.json").get("status") in ("FAILED_IMPLEMENTATION","error","failed"):
                raise Invalid("Explicit supplement scientific error requires review; no automatic retry")
            if args.command=="run":registered_launch(run,plan)
            lock=local_lock(run,plan)
            if args.command=="run":
                snapshot,rows,calibrations=source_snapshot(run,plan)
                if (directory/"raw_completion.json").is_file():verify_raw_completion(directory,plan,snapshot,rows,calibrations)
                else:collect(run,plan,directory,snapshot,rows,calibrations)
            result=analyze(run,plan,directory)
            report=importlib.import_module("state_study_validation512_report")
            try:delivery=report.ensure_delivery(run,plan_path=args.plan)
            except OSError as exc:raise Pending("State complete; delivery I/O pending: "+str(exc)) from exc
            if delivery.get("verified") is not True or delivery.get("delivery_complete") is not True:raise Pending("State complete; delivery pending: "+str(delivery))
            result={"status":"COMPLETE","completion":result,"delivery_complete":True,"delivery":delivery}
        print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True);return 0
    except (Pending,quality.Pending) as exc:
        print(json.dumps({"status":"PENDING","reason":str(exc)}),flush=True);return 2
    except Exception as exc:
        # Only a started, registered collection/analysis owns its separate
        # manifest. A rejected manual invocation cannot poison valid work.
        if directory is not None and plan is not None and lock is not None:
            try:write_manifest(directory,plan,"failure","FAILED_IMPLEMENTATION",error=repr(exc))
            except OSError:pass
        print(json.dumps({"status":"REFUSED","reason":repr(exc)}),file=sys.stderr,flush=True);return 3
    finally:
        if lock is not None:lock.close()


if __name__=="__main__":raise SystemExit(main())
