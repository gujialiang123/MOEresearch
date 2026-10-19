#!/usr/bin/env python3
"""Prospectively seal an independent quality-superiority analysis supplement.

No model, frozen experiment module, parser update, or bootstrap is executed.
The draft fixes mathematical rules before formal selection results; the final
seal binds the complete six-arm D_select and must precede all D_validate data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import uuid


REPO = Path(__file__).resolve().parents[1]
BASELINES = ("N","F","R8","R16","M_N","L_N")
REFERENCES = ("F","R8","R16")
SELECT_IDS = tuple(range(256,512))
VALIDATE_IDS = tuple(range(512,1024))
IDENTITY = ("question_hash","protocol_hash","parser_version","model_revision",
            "model_hash","dtype","max_new_tokens","rendered_prompt","prompt_token_ids")
ANALYSIS_FILES = ("scripts/freeze_state_study_quality_plan.py",
                  "scripts/state_study_positive_gate.py","scripts/state_study_positive_evidence.py",
                  "scripts/summarize_state_study_results.py","docs/2026-10-19/state_quality_plan.md")
ANALYSIS_SPEC = {
    "quality_route":"strict_quality_superiority_no_noninferiority_margin",
    "quality_scores":["strict_correct","tolerant_correct"],
    "quality_controls":["M_NT","strongest_existing_baseline"],
    "quality_method":"simultaneous_bonferroni_clopper_pearson_difference.v1",
    "quality_family_alpha":0.05,"quality_family_comparisons":4,
    "quality_per_tail_alpha":0.00625,"quality_condition":"all four exact lower bounds > 0",
    "cost_endpoint":"relative_complete_generated_token_reduction",
    "cost_formula":"1 - sum(T_M_ST) / sum(T_control)",
    "cost_controls":["M_NT","strongest_existing_baseline"],
    "cost_method":"paired_prompt_ratio_of_sums_percentile_bootstrap.v1",
    "cost_family_alpha":0.05,"cost_family_comparisons":2,"cost_per_tail_alpha":0.0125,
    "bootstrap_resamples":10000,"bootstrap_seed":20261019,
    "undefined_draw_policy":"withhold interval and pass; never discard zero-denominator/nonfinite draws",
    "minimum_useful_relative_token_reduction":0.05,
    "cost_condition":"both simultaneous interval lower bounds >= 0.05",
    "cost_interval_coverage":"nominal simultaneous percentile-bootstrap approximation, not finite-sample exact coverage",
    "downstream_gate_role":"D_validate fixed512 only; D_select complete8 arms supplies selection and exploratory descriptions",
    "state_update_condition":"E_moment(M_ST own)-E_moment(M_NT own) saved 95% CI upper < 0 on C_diag and fixed validation state subset",
    "state_full_trace_required":True,"state_weighting":"token; prompt equal-weight sensitivity remains reported",
    "administrative_cap_policy":"2048 observed request costs remain cap-censored; if >1% selection cap trigger, complete fixed128 cap4096 finalists; any negative relative-token benefit or negative strict/tolerant accuracy difference against NT/strongest withholds positive; never pool caps",
    "confirmation":"POSITIVE_EXPLORATORY only; never unseal P6 or assert historical independence/power automatically",
    "selection_rule":"greatest min(strict,tolerant), then lower mean generated tokens, then fixed baseline ID order; references fixed F/R8/R16",
    "initial_calibration":"L_N","primary_method":"M_ST","same_text_control":"M_NT",
    "noninferiority_margin":None,
}


class Pending(RuntimeError):
    pass


class Invalid(RuntimeError):
    pass


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
        separators=(",",":"),allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    value = json.loads(Path(path).read_bytes())
    if not isinstance(value,dict):
        raise Invalid(f"JSON object required: {path}")
    return value


def sources(run,repo=REPO):
    protocol = load(Path(run)/"protocol.json")
    frozen = load(Path(run)/"source_hashes.json")
    actual_paths = [*Path(repo).glob("moe_research/*.py"),Path(repo)/"scripts/_harness.py",
                    Path(repo)/"scripts/run_state_calibration_study.py",Path(repo)/"configs/state_calibration_study.yaml"]
    actual = {str(path.relative_to(repo)):sha(path) for path in actual_paths}
    if len(frozen)!=18 or actual!=frozen or canonical(frozen)!=protocol.get("source_hash"):
        raise Invalid("Frozen 18-source inventory/hash differs")
    if canonical({k:v for k,v in protocol.items() if k!="protocol_hash"})!=protocol.get("protocol_hash"):
        raise Invalid("Protocol self hash differs")
    analysis = {}
    for name in ANALYSIS_FILES:
        path=Path(repo)/name
        if not path.is_file():
            raise Pending(f"Independent analysis source not yet ready: {name}")
        analysis[name]=sha(path)
    return protocol,{"frozen_source_hash":canonical(frozen),"analysis_source_hashes":analysis,
                     "analysis_source_hash":canonical(analysis),
                     "analysis_archive_relative_dir":"quality_analysis_sources/"+canonical(analysis)}


def validation_absence(run):
    """No validation output is permitted before a new final seal."""
    found=[]
    for name in ("formal/validate","formal/validation_state"):
        directory=Path(run)/name
        if directory.exists():
            found.extend(str(path.relative_to(run)) for path in directory.rglob("*") if path.is_file())
    if found:
        raise Invalid("Validation evidence already exists; cannot prospectively freeze: "+", ".join(found[:8]))
    return {"checked_at_utc":datetime.now(timezone.utc).isoformat(),
            "formal_validate_files":[],"formal_validation_state_files":[],"absence_verified":True}


def formal_selection_absence(run):
    if any(path.stat().st_size for path in (Path(run)/"formal/select/generations").glob("*.jsonl")):
        raise Invalid("Draft mathematical rules must be declared before formal selection output; no late draft")


def validation_mtime_guard(run,created_at):
    """Additional fail-closed check, never proof of a file's first write time."""
    created=datetime.fromisoformat(created_at.replace("Z","+00:00"))
    if created.tzinfo is None:raise Invalid("Final seal timestamp must include a timezone")
    delta=created-datetime(1970,1,1,tzinfo=timezone.utc)
    cutoff=((delta.days*86400+delta.seconds)*1000000+delta.microseconds)*1000
    checked=0
    for name in ("formal/validate","formal/validation_state"):
        for path in (Path(run)/name).rglob("*"):
            if path.is_file():
                checked+=1
                if path.stat().st_mtime_ns<cutoff:
                    raise Invalid("Validation file mtime is earlier than final seal: "+str(path.relative_to(run)))
    return {"checked_files":checked,"status":"NO_EARLIER_CURRENT_MTIME",
            "limitation":"Current mtime is auxiliary; append changes it and it cannot prove first-write order or replace coordinated absence checks."}


def read_rows(path):
    path=Path(path)
    if not path.is_file():
        raise Pending(f"Missing complete D_select arm: {path.name}")
    rows={}
    with path.open("rb") as handle:
        for line,raw in enumerate(handle,1):
            if not raw.endswith(b"\n"):
                raise Pending(f"D_select physical record is still being written: {path}:{line}")
            try:row=json.loads(raw)
            except (ValueError,UnicodeError) as exc:raise Invalid(f"Invalid JSONL {path}:{line}") from exc
            sid=row.get("sample_id") if isinstance(row,dict) else None
            if isinstance(sid,bool) or not isinstance(sid,int) or sid in rows:
                raise Invalid(f"Invalid/duplicate sample ID {path}:{line}")
            rows[sid]=row
    if set(rows)!=set(SELECT_IDS):
        if set(rows)-set(SELECT_IDS):raise Invalid(f"Unexpected selection IDs: {path}")
        raise Pending(f"D_select {path.stem} has {len(rows)}/256 fixed IDs")
    return rows


def selection_snapshot(run,protocol):
    """Bind all six baseline request outcomes, prompts, parser and source IDs."""
    run=Path(run);config=protocol["config"];rows={};records={};ranking=[];calibrations={}
    for arm,family in (("M_N","scalar"),("L_N","lda")):
        path=run/f"formal/calibration/{arm}.json"
        if not path.is_file():raise Pending("P2 formal native calibration is not published: "+arm)
        value=load(path)
        if value.get("family")!=family or value.get("metadata",{}).get("protocol_hash")!=protocol["protocol_hash"] or value.get("calibration_hash")!=canonical({k:v for k,v in value.items() if k!="calibration_hash"}):
            raise Invalid("P2 native calibration identity differs: "+arm)
        calibrations[arm]={"relative_path":str(path.relative_to(run)),"sha256":sha(path),"calibration_hash":value["calibration_hash"]}
    for arm in BASELINES:
        path=run/f"formal/select/generations/{arm}.jsonl"
        before=path.stat() if path.is_file() else None
        data=read_rows(path)
        for sid,row in data.items():
            expected={"arm":arm,"scope":"formal/select","protocol_hash":protocol["protocol_hash"],
                      "parser_version":protocol["parser_version"],"model_revision":config["model_revision"],
                      "dataset_revision":config["dataset_revision"],"max_new_tokens":2048}
            cal_hash=calibrations.get(arm,{}).get("calibration_hash")
            expected["calibration_hash"]=cal_hash
            expected["arm_hash"]=canonical({"arm":arm,"k_compute":8 if arm=="N" else 4,
                "k_reference":{"N":8,"F":4,"R8":8,"R16":16,"M_N":8,"L_N":4}[arm],
                "calibration_hash":cal_hash,"prefill":"native","seed":20261019})
            if any(row.get(k)!=value for k,value in expected.items()):raise Invalid(f"Selection request identity differs: {arm}/{sid}")
            if not isinstance(row.get("question"),str) or hashlib.sha256(row["question"].encode()).hexdigest()!=row.get("question_hash"):
                raise Invalid(f"Selection question hash differs: {arm}/{sid}")
            if not isinstance(row.get("strict_correct"),bool) or not isinstance(row.get("tolerant_correct"),bool):raise Invalid(f"Missing strict/tolerant score: {arm}/{sid}")
            n=row.get("generated_tokens");tokens=row.get("gen_token_ids")
            if isinstance(n,bool) or not isinstance(n,int) or not 1<=n<=2048 or not isinstance(tokens,list) or len(tokens)!=n:
                raise Invalid(f"Selection length identity differs: {arm}/{sid}")
            if row.get("stop_reason") not in ("eos","max_new_tokens") or row.get("observer_enabled") is not False or row.get("timing_debug_logits") is not False:
                raise Invalid(f"Incomplete/debug selection request: {arm}/{sid}")
            if row.get("stop_reason")=="eos" and (not tokens or tokens[-1] not in row.get("eos_ids",[])):
                raise Invalid(f"Selection EOS identity differs: {arm}/{sid}")
        if rows and any(data[sid].get(field)!=rows["N"][sid].get(field) for sid in SELECT_IDS for field in IDENTITY):
            raise Invalid(f"Selection paired prompt identity differs: {arm}")
        after=path.stat()
        if before is None or (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise Pending(f"Selection arm changed during snapshot: {arm}")
        strict=sum(row["strict_correct"] for row in data.values())/256
        tolerant=sum(row["tolerant_correct"] for row in data.values())/256
        length=sum(row["generated_tokens"] for row in data.values())/256
        records[arm]={"relative_path":str(path.relative_to(run)),"sha256":sha(path),"bytes":after.st_size,
                      "sample_ids":list(SELECT_IDS),"strict_accuracy":strict,"tolerant_accuracy":tolerant,
                      "mean_generated_tokens":length,"arm_hash":next(iter(data.values())).get("arm_hash"),
                      "outcomes_hash":canonical([{k:data[sid].get(k) for k in
                          ("sample_id","question_hash","strict_correct","tolerant_correct","generated_tokens","stop_reason","arm_hash")} for sid in SELECT_IDS])}
        rows[arm]=data
        if arm!="N":ranking.append((-min(strict,tolerant),length,BASELINES.index(arm),arm))
    reference=min(((-min(records[a]["strict_accuracy"],records[a]["tolerant_accuracy"]),records[a]["mean_generated_tokens"],REFERENCES.index(a),a) for a in REFERENCES))[-1]
    return {"scope":"formal/select","sample_ids":list(SELECT_IDS),"arms":records,
            "native_calibration_inputs":calibrations,
            "selected_reference":reference,"strongest_existing_baseline":min(ranking)[-1],
            "snapshot_hash":canonical(records),"all_six_arms_complete":True}


def seal(value):
    return {**value,"plan_hash":canonical(value)}


def supplement(protocol):
    return {"type":"prospective_analysis_supplement_and_correction_before_D_validate",
        "basis":"Original user plan section11.4 permits an explicit quality-superiority constraint; implementation NI-only text narrowed that requirement.",
        "original_protocol_constraint":protocol["config"].get("quality_constraint_for_internal_validation"),
        "original_NI_only_condition_satisfied":False,"original_NI_margin_remains_None":True,
        "corrected_scope":"Quality-superiority decision and executable POSITIVE_EXPLORATORY report chain only.",
        "additional_sensitivity_requirement":"The 4096 no-adverse-point-direction veto is an explicitly new prospective supplement, not an original-plan per-score veto.",
        "unchanged":"Frozen measurement/model/dtype/arms/sample ranges/main state thresholds and original validation_freeze.json.",
        "original_protocol_hash":protocol["protocol_hash"],"original_source_hash":protocol["source_hash"],
        "not_an_original_protocol_condition":True,"no_P6_authorization":True}


def make_draft(run,repo=REPO):
    protocol,source=sources(run,repo);absence=validation_absence(run)
    formal_selection_absence(run)
    return seal({"schema_version":"state_study_quality_plan.v1","status":"DRAFT_NOT_FROZEN",
        "created_at_utc":datetime.now(timezone.utc).isoformat(),"run_dir":str(Path(run).resolve()),
        "protocol_hash":protocol["protocol_hash"],"source":source,"analysis_spec":ANALYSIS_SPEC,
        "analysis_supplement":supplement(protocol),
        "prevalidation_absence":absence,"formal_selection_output_absent_at_draft":True,
        "note":"Prospective analysis supplement; not a frozen final comparison, no observed outcome claim."})


def verify_self(plan,status):
    if plan.get("schema_version")!="state_study_quality_plan.v1" or plan.get("status")!=status:
        raise Invalid("Quality plan schema/status differs")
    if plan.get("plan_hash")!=canonical({k:v for k,v in plan.items() if k!="plan_hash"}):raise Invalid("Quality plan self hash differs")
    if plan.get("analysis_spec")!=ANALYSIS_SPEC:raise Invalid("Prospective formulas/thresholds differ from current declared specification")


def verify_source_archive(plan):
    proof=plan.get("analysis_source_archive",{});directory=Path(proof.get("directory","/nonexistent"))
    manifest=directory/"MANIFEST.json"
    if not manifest.is_file() or sha(manifest)!=proof.get("manifest_sha256"):
        raise Invalid("Immutable actual analysis source archive is missing or differs")
    expected=plan["source"]["analysis_source_hashes"]
    if load(manifest).get("files")!=expected or any(not (directory/name).is_file() or sha(directory/name)!=digest for name,digest in expected.items()):
        raise Invalid("Archived analysis source bytes differ")


def prepare_plan(run,draft,repo=REPO):
    protocol,source=sources(run,repo)
    snapshot=selection_snapshot(run,protocol)  # incomplete P2 remains PENDING
    absence=validation_absence(run)
    verify_self(draft,"DRAFT_NOT_FROZEN")
    verify_source_archive(draft)
    if draft.get("protocol_hash")!=protocol["protocol_hash"] or draft.get("source")!=source or not draft.get("formal_selection_output_absent_at_draft"):
        raise Invalid("Draft protocol/source/precedence identity differs")
    return seal({"schema_version":"state_study_quality_plan.v1","status":"FROZEN_PROSPECTIVE",
        "created_at_utc":datetime.now(timezone.utc).isoformat(),"run_dir":str(Path(run).resolve()),
        "protocol_hash":protocol["protocol_hash"],"source":source,"analysis_spec":ANALYSIS_SPEC,
        "analysis_supplement":supplement(protocol),
        "draft_hash":draft["plan_hash"],"draft_created_at_utc":draft["created_at_utc"],
        "selection":snapshot,"prevalidation_absence":absence,
        "validation_sample_ids":list(VALIDATE_IDS),"validation_state_sample_ids":list(range(512,640)),
        "preserves_original_validation_freeze":True,"original_noninferiority_margin":None,
        "note":"Selection-bound prospective quality-superiority supplement; no P6 authorization."})


def verify_plan(run,plan,repo=REPO):
    """Read-only verification after validation exists; never backdates a seal."""
    verify_self(plan,"FROZEN_PROSPECTIVE");protocol,source=sources(run,repo)
    verify_source_archive(plan)
    if plan.get("protocol_hash")!=protocol["protocol_hash"] or plan.get("source")!=source or plan.get("run_dir")!=str(Path(run).resolve()):
        raise Invalid("Frozen quality plan protocol/source/run identity differs")
    if plan.get("selection")!=selection_snapshot(run,protocol):raise Invalid("Frozen D_select bytes/selection changed")
    if not plan.get("prevalidation_absence",{}).get("absence_verified") or not plan.get("draft_hash"):
        raise Invalid("Prospective absence/draft proof is missing")
    if plan.get("analysis_supplement")!=supplement(protocol):raise Invalid("Quality supplement scope/original-condition boundary differs")
    timing=validation_mtime_guard(run,plan["created_at_utc"])
    freeze_path=Path(run)/"validation_freeze.json"
    if freeze_path.is_file():
        frozen=load(freeze_path)
        for key in ("selected_reference","strongest_existing_baseline"):
            if frozen.get(key)!=plan["selection"][key]:raise Invalid("Runner/final quality selection differs: "+key)
        if frozen.get("quality_noninferiority_margin") is not None:raise Invalid("This superiority plan does not authorize a new NI margin")
    return {"status":"VERIFIED_PROSPECTIVE_PLAN","plan_hash":plan["plan_hash"],
            "source":source,"auxiliary_validation_mtime_check":timing,"selected_reference":plan["selection"]["selected_reference"],
            "strongest_existing_baseline":plan["selection"]["strongest_existing_baseline"]}


def publish_once(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name("."+path.name+"."+uuid.uuid4().hex+".pending")
    try:
        with temporary.open("x",encoding="utf-8") as handle:
            json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write("\n");handle.flush();os.fsync(handle.fileno())
        os.link(temporary,path)
        descriptor=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    finally:temporary.unlink(missing_ok=True)


def publish_analysis_sources(report_dir,source,repo=REPO):
    """Keep actual small analysis sources separately from the model-code tar."""
    destination=Path(report_dir)/source["analysis_archive_relative_dir"]
    manifest=destination/"MANIFEST.json"
    if destination.exists():
        if not manifest.is_file() or load(manifest).get("files")!=source["analysis_source_hashes"]:
            raise Invalid("Existing analysis source archive is incomplete or differs; never overwrite")
        if any(not (destination/name).is_file() or sha(destination/name)!=digest for name,digest in source["analysis_source_hashes"].items()):
            raise Invalid("Immutable analysis source archive hash differs")
        return {"directory":str(destination.resolve()),"manifest_sha256":sha(manifest)}
    destination.mkdir(parents=True,exist_ok=False)
    for name,digest in source["analysis_source_hashes"].items():
        raw=(Path(repo)/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=digest:raise Invalid("Analysis source changed before archival: "+name)
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open("xb") as handle:handle.write(raw);handle.flush();os.fsync(handle.fileno())
    publish_once(manifest,{"schema_version":"state_quality_analysis_sources.v1",
        "files":source["analysis_source_hashes"],"analysis_source_hash":source["analysis_source_hash"],
        "scope":"Independent analysis source, separate from frozen18 model measurement code archive",
        "measurement_code_archive":"original run/source_code.tar.gz (not duplicated here)"})
    return {"directory":str(destination.resolve()),"manifest_sha256":sha(manifest)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",required=True);parser.add_argument("--output")
    parser.add_argument("--draft");parser.add_argument("--draft-only",action="store_true")
    parser.add_argument("--check-only",action="store_true")
    args=parser.parse_args(argv);run=Path(args.run_dir).resolve()
    reports=Path("/scratch/jgu7/moe-state-study/reports")/run.name
    output=Path(args.output).resolve() if args.output else reports/("quality_plan.draft.json" if args.draft_only else "quality_plan.json")
    draft_path=Path(args.draft).resolve() if args.draft else reports/"quality_plan.draft.json"
    if not run.is_dir() or not run.is_relative_to(Path("/scratch/jgu7/moe-state-study/runs")):
        parser.error("Only an existing scratch study run is allowed")
    if output.is_relative_to(run) or output.is_relative_to(Path.home().resolve()):parser.error("Quality supplements must be outside the read-only run and outside Home")
    try:
        if args.draft_only:
            value=make_draft(run)
        elif output.exists():
            value=load(output);proof=verify_plan(run,value)
            print(json.dumps({**proof,"path":str(output),"already_frozen":True}));return 0
        else:
            protocol,_=sources(run)
            selection_snapshot(run,protocol)
            if not draft_path.is_file():raise Pending("Prospective draft has not been published")
            value=prepare_plan(run,load(draft_path))
        if not args.check_only:
            archive=publish_analysis_sources(output.parent,value["source"])
            value=seal({**{k:v for k,v in value.items() if k!="plan_hash"},"analysis_source_archive":archive})
            validation_absence(run)
            if args.draft_only:formal_selection_absence(run)
            publish_once(output,value)
        print(json.dumps({"status":"READY_NOT_WRITTEN" if args.check_only else value["status"],"path":str(output),"plan_hash":value["plan_hash"]}));return 0
    except Pending as exc:
        print(json.dumps({"status":"PENDING","reason":str(exc),"output_created":False}));return 2
    except (Invalid,OSError,ValueError,KeyError) as exc:
        print(json.dumps({"status":"REFUSED","reason":str(exc),"output_created":False}),file=sys.stderr);return 3


if __name__=="__main__":raise SystemExit(main())
