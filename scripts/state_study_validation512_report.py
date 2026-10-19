#!/usr/bin/env python3
"""Independent full-D_validate own-state acceptance and scientific reporting.

The frozen runner and original five analysis sources remain unchanged.  This
module reuses their pure predicates and readers, and replaces only the primary
validation-state endpoint with verified 512-prompt evidence.  It never executes
a model, refits calibration, or recomputes a state bootstrap.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

if __package__:
    from . import summarize_state_study_results as legacy_report
    from . import state_study_positive_evidence as legacy_gate
    from .freeze_state_study_quality_plan import Invalid, Pending, canonical, load, publish_once, sha, verify_plan
else:
    import summarize_state_study_results as legacy_report
    import state_study_positive_evidence as legacy_gate
    from freeze_state_study_quality_plan import Invalid, Pending, canonical, load, publish_once, sha, verify_plan

IDS = tuple(range(512,1024))
ARMS = ("M_NT","M_ST")
OWN = "M_ST_minus_M_NT_own_trajectories"
PREDICATE_MAPPING = {
    "P0_protocol_pass":"UNCHANGED: original preflight status and protocol identity",
    "no_record_identity_or_format_errors":"UNCHANGED: original readers and generation records",
    "runner_validation_freeze_matches_supplement":"UNCHANGED: legacy validation_contract",
    "all_eight_formal_selection_arms_complete":"UNCHANGED: all eight original D_select arms",
    "frozen_validation_arms_complete":"UNCHANGED: frozen finalists, all512 free-generation requests",
    "P3_numerical_state_source_gate":"UNCHANGED: legacy p3_numerical_gate",
    "updated_Cdiag_own_state_improvement":"UNCHANGED: legacy own_state_improves on C_diag128",
    "Cdiag_declared_prompt_completeness":"UNCHANGED: fixed128 C_diag and six diagnostic paths",
    "Cdiag_actual_full_trace_masks":"UNCHANGED: original masks and actual C_diag generation traces",
    "Cdiag_bottom_archive_hashes":"UNCHANGED: original committed NPZ/token hashes",
    "Cdiag_frozen_saved_state_bootstrap_contract":"UNCHANGED: legacy bootstrap_contract",
    "same_formula_mask_budget_source_provenance":"UNCHANGED: legacy calibration_provenance",
    "all_formal_generation_dataset_arm_identity":"UNCHANGED: legacy generation_identities on real unchanged base/raw; includes legacy128 generation artifacts",
    "Cdiag_bottom_driver_calibration_position_identity":"UNCHANGED: legacy diagnostic_identities for C_diag",
    "legacy_validation128_archive_identity":"RETAINED archival identity audit; its CI significance is not a primary endpoint",
    "updated_validation_own_state_improvement":"REPLACED: full512 token-weighted own contrast finite saved95% CI upper<0",
    "validation_declared_prompt_completeness":"REPLACED: actual expected IDs512:1024 for both updated arms",
    "validation_actual_full_trace_masks":"REPLACED: exact full512 actual formal/validate trace[:-1] masks and48 layers",
    "validation_bottom_archive_hashes":"REPLACED: full512 committed NPZ/token SHA and physical rows",
    "validation_frozen_saved_state_bootstrap_contract":"REPLACED: full512 expected IDs,10000,seed20261019,paired,token+prompt; no refit",
    "validation_bottom_driver_calibration_position_identity":"REPLACED: full512 actual generation/calibrator/driver/position identity",
    "triggered_fixed128_cap4096_complete":"UNCHANGED: original finalists/fixed128/cap4096 sensitivity",
    "cap4096_no_adverse_point_direction":"UNCHANGED: original prospective supplementary direction rule",
    "four_simultaneous_quality_lower_bounds_positive_and_two_ratio_lower_bounds_at_least_5pct":"UNCHANGED: legacy downstream_statistics, original quality plan hash and same-input cache",
}


def read_base_evidence(run_dir,verify_hashes=True):
    """Use frozen readers without executing the old128 positive classifier."""
    ev=legacy_report.Evidence(run_dir,verify_hashes)
    protocol=ev.json("protocol.json") or {};manifest=ev.json("manifest.json") or {}
    phash=protocol.get("protocol_hash");freeze=ev.json("validation_freeze.json") or {}
    preflight=ev.json("preflight.json") or ev.json("preflight/preflight.json") or {}
    if protocol and canonical({k:v for k,v in protocol.items() if k!="protocol_hash"})!=phash:
        ev.errors.append("Protocol self hash differs")
    if any(x and x.get("protocol_hash")!=phash for x in (manifest,freeze,preflight)):
        ev.errors.append("Manifest/freeze/preflight protocol identity differs")
    groups={};raw={}
    scopes=(("smoke/select",256,260,legacy_gate.METHODS[:6]),("pilot/select",256,288,legacy_gate.METHODS[:6]),
        ("formal/select",256,512,legacy_gate.METHODS[:6]),("formal/validate",512,1024,freeze.get("finalist_arms",())),
        ("formal/select_cap4096",256,384,freeze.get("finalist_arms",())),("formal/fit_native",0,128,("N",)),
        ("formal/fit_student",0,128,("L_N",)),("formal/diag_rollouts",128,256,("N","L_N")),
        ("formal/updated_diag_rollouts",128,256,ARMS),("formal/validation_state/diag_rollouts",512,640,("N","L_N")),
        ("formal/validation_state/updated_diag_rollouts",512,640,ARMS))
    for scope,lo,hi,arms in scopes:
        groups[scope],raw[scope]=legacy_report.load_generation(ev,scope,list(range(lo,hi)),phash,arms)
    diagnostics={scope:legacy_report.load_diagnostic(ev,scope) for scope in ("smoke","pilot","formal","formal/validation_state")}
    legacy_report.reconcile_state_masks(diagnostics["formal"],raw,list(range(128,256)),"formal/diag_rollouts","formal/updated_diag_rollouts")
    legacy_report.reconcile_state_masks(diagnostics["formal/validation_state"],raw,list(range(512,640)),"formal/validation_state/diag_rollouts","formal/validation_state/updated_diag_rollouts")
    calibration=legacy_report.load_calibrations(ev,phash)
    launch=ev.json("launch.json") or {}
    base={"protocol":protocol,"manifest":manifest,"preflight":preflight,"validation_freeze":freeze,
          "calibration":calibration,"generations":groups,"diagnostics":diagnostics,
          "run_dir":str(ev.run),"generated_at_utc":datetime.now(timezone.utc).isoformat(),
          "process_verification":legacy_report.verify_launch_process(ev.run,launch)}
    return base,raw,ev


def full512_analysis_contract(diagnostic):
    """Completion is distinct from evidence of improvement."""
    summary=diagnostic.get("summary") or {};checks={}
    checks["expected_512_ids"]=summary.get("expected_prompt_ids")==sorted(str(i) for i in IDS)
    checks["full512_prompt_completeness"]=all(summary.get("prompt_completeness",{}).get(a,{}).get("complete_expected_ids") is True for a in ARMS)
    checks["bootstrap_contract"]=legacy_gate.bootstrap_contract(summary)
    checks["lda_E_moment"]=summary.get("family")=="lda" and summary.get("chosen_mismatch_metric")=="E_moment"
    comparisons={}
    for weight in ("token","prompt"):
        comparison=summary.get("analyses",{}).get("all",{}).get(weight,{}).get("comparisons",{}).get(OWN,{})
        comparisons[weight]=comparison
        structured=(comparison.get("prompt_count")==512
            and comparison.get("left")=="M_ST" and comparison.get("right")=="M_NT"
            and comparison.get("contrast")=="left_minus_right"
            and comparison.get("same_calibration_required") is False and comparison.get("same_mask_required") is False)
        # An undefined interval remains an honest scientific result if the saved
        # analysis ran; it cannot establish improvement or an equivalence claim.
        metric=comparison.get("metrics",{}).get("E_moment")
        metric_present=(legacy_gate.good_interval(metric) and metric["ci_low"]<=metric["ci_high"]) if isinstance(metric,dict) else False
        if isinstance(metric,dict) and metric.get("status")=="undefined_bootstrap_draw":
            draws=metric.get("finite_draws")
            metric_present=(all(k in metric for k in ("estimate","ci_low","ci_high","status","finite_draws"))
                and metric.get("ci_low") is None and metric.get("ci_high") is None
                and (metric.get("estimate") is None or legacy_gate.finite(metric.get("estimate")))
                and isinstance(draws,int) and not isinstance(draws,bool) and 0<=draws<=10000)
        coverage=summary.get("coverage",{}).get("all",{})
        an_empty_path=any(coverage.get(a,{}).get("zero_observation_prompt_ids")==sorted(str(i) for i in IDS) for a in ARMS)
        explicitly_undefined=(comparison.get("status")=="NOT_ESTIMABLE" and comparison.get("reason")=="one_or_both_paths_have_no_observations" and an_empty_path)
        no_observations=(summary.get("analyses",{}).get("all",{}).get("status")=="NO_OBSERVATIONS"
                         and summary.get("analyses",{}).get("all",{}).get("reason")=="no_valid_decode_positions_in_predeclared_group"
                         and all(coverage.get(a,{}).get("zero_observation_prompt_ids")==sorted(str(i) for i in IDS) for a in ARMS))
        checks[weight+"_saved_analysis_result"]=(structured and metric_present and not an_empty_path) or explicitly_undefined or no_observations
    token=comparisons["token"].get("metrics",{}).get("E_moment")
    improved=(all(checks.values()) and comparisons["token"].get("status")=="ok" and legacy_gate.good_interval(token) and token["ci_high"]<0)
    estimable=all(comparisons[w].get("status")=="ok" and legacy_gate.good_interval(comparisons[w].get("metrics",{}).get("E_moment")) for w in ("token","prompt"))
    return {"checks":checks,"analysis_complete":all(checks.values()),"analysis_estimable":estimable,"own_state_improves":improved,
        "token":token,"prompt":comparisons["prompt"].get("metrics",{}).get("E_moment"),
        "prompt_direction_is_not_an_additional_veto":True,
        "top_level_ABCD_missing_status_is_not_a_veto":True}


def verify_full512_archives(ev,diagnostic,rows,calibrations,protocol_hash):
    """Verify committed bytes and position arrays; do not recompute moments."""
    proof={"status":"INCOMPLETE","errors":[],"paths_verified":[],"metadata_only_no_moment_recomputation":True}
    if not diagnostic.get("archive_hashes_verified") or not diagnostic.get("current_inventory_matches"):return proof
    if any(set(rows.get(a,{}))!=set(IDS) for a in ARMS):return proof
    import numpy as np
    directory=ev.run/"diagnostics";seen=set();layers=set(range(48))
    for archive_path in sorted(directory.glob("*.npz")):
        try:
            commit=ev.json(archive_path.with_suffix(".complete.json"))
            if not commit or commit.get("schema_version")!="state_collection.commit.v1" or commit.get("collection_schema_version")!="state_collection.v2" or commit.get("archive")!=archive_path.name:
                raise ValueError("commit schema/archive differs")
            token_path=directory/commit.get("tokens","")
            if token_path.parent!=directory or token_path.name!=commit.get("tokens"):
                raise ValueError("commit token basename differs")
            inventory=ev.inventory.get(str(token_path.relative_to(ev.run)),{})
            if inventory.get("physical_lines")!=commit.get("token_rows"):raise ValueError("token physical lines differ")
            with np.load(archive_path,allow_pickle=False) as archive:
                meta=json.loads(str(archive["metadata_json"]))
                if any(meta.get(k)!=v for k,v in {"schema_version":"state_collection.v2","dtype":"float64","ddof":0,"moment_representation":"count_mean_centered_m2"}.items()):
                    raise ValueError("numerical metadata differs")
                entries=meta.get("records",[]);all_entries=[e for e in entries if e["key"][3]=="all"]
                if not all_entries:raise ValueError("all-position records missing")
                pid,arm=all_entries[0]["key"][:2];sid=int(pid)
                if arm not in ARMS or sid not in IDS:raise ValueError("unexpected prompt/path")
                if (sid,arm) in seen or archive_path.stem!=canonical([str(pid),arm]):raise ValueError("duplicate or wrong archive filename")
                trace=rows[arm][sid]["gen_token_ids"];positions=[[j,t] for j,t in enumerate(trace[:-1],1)]
                if not positions:raise ValueError("empty trace incorrectly archived as NPZ")
                if len(all_entries)!=48 or {int(e["key"][2]) for e in all_entries}!=layers:raise ValueError("48-layer rectangle differs")
                if commit.get("token_rows")!=48*len(positions):raise ValueError("commit T-1 count differs")
                if meta.get("numerical_protection")!=calibrations[arm].get("numerical_protection"):raise ValueError("numerical protection differs")
                groups=set();expected_bins=meta.get("position_bins")
                if expected_bins!=[[1,32],[33,128],[129,256],[257,512],[513,1024],[1025,2047]]:raise ValueError("position bins differ")
                for entry in entries:
                    epid,path,layer,group=entry["key"];groups.add((int(layer),group))
                    if str(epid)!=str(pid) or path!=arm or int(layer) not in layers:raise ValueError("mixed record identity")
                    if entry.get("driver_hash")!=legacy_gate.arm_hash(arm,calibrations) or entry.get("calibration_hash")!=calibrations[arm]["calibration_hash"]:
                        raise ValueError("driver/calibration differs")
                    selected=positions
                    if group!="all":
                        if group not in {str(lo)+"-"+str(hi) for lo,hi in expected_bins}:raise ValueError("unexpected group")
                        lo,hi=map(int,group.split("-"));selected=[p for p in positions if lo<=p[0]<=hi]
                    if entry.get("count")!=len(selected) or entry.get("mask_hash")!=canonical(selected):raise ValueError("actual trace mask/count differs")
                    actual=archive[entry["prefix"]+"_positions"]
                    if actual.dtype!=np.int64 or actual.shape!=(len(selected),2) or actual.tolist()!=selected:raise ValueError("actual position array differs")
                expected_groups={(layer,"all") for layer in layers}|{(layer,str(lo)+"-"+str(hi)) for layer in layers for lo,hi in expected_bins if any(lo<=p[0]<=hi for p in positions)}
                if len(entries)!=len(groups) or groups!=expected_groups:raise ValueError("position-bin/layer rectangle differs")
                seen.add((sid,arm))
        except (OSError,ValueError,KeyError,TypeError) as exc:proof["errors"].append(f"{archive_path.name}: {exc}")
    for marker_path in sorted(directory.glob("*.empty.json")):
        try:
            marker=ev.json(marker_path) or {};sid=int(marker.get("prompt_id"));arm=marker.get("path")
            if arm not in ARMS or sid not in IDS or (sid,arm) in seen:raise ValueError("unexpected/duplicate empty identity")
            trace=rows[arm][sid]["gen_token_ids"]
            identity={"protocol_hash":protocol_hash,"driver_hash":legacy_gate.arm_hash(arm,calibrations),
                "calibration_hash":calibrations[arm]["calibration_hash"],"trace_hash":canonical(trace)}
            if len(trace)!=1 or any(marker.get(k)!=v for k,v in {"schema_version":"state_collection.empty.v1","collection_schema_version":"state_collection.v2","count":0,"dtype":"float64","ddof":0,"metadata":identity}.items()):
                raise ValueError("empty marker identity differs")
            if not isinstance(marker.get("reason"),str) or not marker["reason"] or marker.get("numerical_protection")!=calibrations[arm].get("numerical_protection") or marker.get("position_bins")!=[[1,32],[33,128],[129,256],[257,512],[513,1024],[1025,2047]]:
                raise ValueError("empty numerical contract/reason differs")
            if marker_path.name!=canonical([str(sid),arm])+".empty.json":raise ValueError("empty marker filename differs")
            seen.add((sid,arm))
        except (OSError,ValueError,KeyError,TypeError) as exc:proof["errors"].append(f"{marker_path.name}: {exc}")
    for arm in ARMS:
        if {sid for sid,path in seen if path==arm}==set(IDS):proof["paths_verified"].append(arm)
    proof["verified_prompt_paths"]=len(seen)
    proof["status"]="VERIFIED" if not proof["errors"] and set(proof["paths_verified"])==set(ARMS) else "UNVERIFIED"
    return proof


def evaluate_validation512_result(base,raw,ev,state512,quality_plan_path,cache_dir=None,allow_statistics=True):
    """Reuse every nonvalidation predicate without changing base/raw fields."""
    result={"status":"INCONCLUSIVE","passed":False,"p6_authorized":False,"requirements":{},
        "predicate_mapping":PREDICATE_MAPPING,"validation512":state512,
        "reason":"Full512 evidence or unchanged prerequisites incomplete"}
    if (state512.get("state_status")=="BLOCKED" or ev.errors
        or base["preflight"].get("status") in ("FAILED_IMPLEMENTATION","FAILED","FAIL")
        or base["manifest"].get("status") in ("error","FAILED_IMPLEMENTATION")):
        return {**result,"status":"BLOCKED","reason":"Evidence identity/integrity error"}
    if state512.get("state_status")=="NOT_REQUIRED":
        # Preserve the original narrow negative branch. NOT_REQUIRED by itself
        # is neither a negative scientific result nor a completed512 dataset.
        negative=(all(d.get("status")=="NEGATIVE" and d.get("archive_hashes_verified")
            and all(d.get("full_trace_masks_reconciled",{}).get(p) for p in ("A","B","C","D"))
            for d in (base["diagnostics"]["formal"],base["diagnostics"]["formal/validation_state"]))
            and base["generations"]["formal/validate"].get("complete") is True
            and base["preflight"].get("protocol_hash")==base["protocol"].get("protocol_hash")
            and base["preflight"].get("status") in ("PASS","PASSED"))
        return {**result,"status":"NEGATIVE" if negative else "INCONCLUSIVE","state512_not_required":True,
            "reason":"Original verified formal and validation negative state branch passes; no updated full512 completion claim" if negative else "Original formal P3 did not trigger an update; no full512 completion or negative-result claim"}
    try:plan=load(quality_plan_path);verification=verify_plan(ev.run,plan)
    except (Pending,FileNotFoundError) as exc:return {**result,"quality_plan_status":"UNVERIFIED_OR_PENDING","reason":str(exc)}
    except (Invalid,OSError,ValueError,KeyError) as exc:return {**result,"status":"BLOCKED","quality_plan_status":"UNVERIFIED","reason":str(exc)}
    result.update(quality_plan_status=verification["status"],plan_hash=plan["plan_hash"])
    req=result["requirements"];protocol=base["protocol"];groups=base["generations"];cal=base["calibration"]["arms"]
    formal=base["diagnostics"]["formal"];old=base["diagnostics"]["formal/validation_state"]
    req["P0_protocol_pass"]=base["preflight"].get("status") in ("PASS","PASSED") and base["preflight"].get("protocol_hash")==protocol.get("protocol_hash")
    req["no_record_identity_or_format_errors"]=not ev.errors
    req["runner_validation_freeze_matches_supplement"]=legacy_gate.validation_contract(base,plan)
    req["all_eight_formal_selection_arms_complete"]=all(groups["formal/select"]["arms"].get(a,{}).get("complete") for a in legacy_gate.METHODS)
    finalists=base["validation_freeze"].get("finalist_arms",[])
    req["frozen_validation_arms_complete"]=bool(finalists) and all(groups["formal/validate"]["arms"].get(a,{}).get("complete") for a in finalists)
    req["P3_numerical_state_source_gate"]=legacy_gate.p3_numerical_gate(formal,protocol)
    req["updated_Cdiag_own_state_improvement"]=legacy_gate.own_state_improves(formal)
    summary=formal.get("summary") or {};paths=("A","B","C","D",*ARMS)
    req["Cdiag_declared_prompt_completeness"]=(summary.get("expected_prompt_ids")==sorted(str(i) for i in range(128,256)) and all(summary.get("prompt_completeness",{}).get(a,{}).get("complete_expected_ids") is True for a in paths))
    req["Cdiag_actual_full_trace_masks"]=all(formal.get("full_trace_masks_reconciled",{}).get(a) for a in paths)
    req["Cdiag_bottom_archive_hashes"]=formal.get("archive_hashes_verified") is True
    req["Cdiag_frozen_saved_state_bootstrap_contract"]=legacy_gate.bootstrap_contract(summary)
    fit=legacy_gate.calibration_provenance(ev,raw,cal);result["calibration_provenance"]=fit
    req["same_formula_mask_budget_source_provenance"]=fit["status"]=="VERIFIED"
    req["validation512_plan_and_evidence_verified"]=state512.get("state_status")=="VERIFIED"
    req["validation512_full_coverage_and_analysis"]=state512.get("coverage_complete") is True and state512.get("analysis_complete") is True
    req["updated_validation512_own_state_improvement"]=state512.get("own_state_improves") is True
    result["legacy128_scope"]={"ids":list(range(512,640)),"ci_is_not_a_primary_gate":True,
        "saved_token_comparison":legacy_gate.state_contrast(old,OWN)}
    if not all(req.values()):return result
    try:req["all_formal_generation_dataset_arm_identity"]=legacy_gate.generation_identities(base,raw,cal)
    except (OSError,ValueError,KeyError,TypeError):req["all_formal_generation_dataset_arm_identity"]=False
    if not all(req.values()):return result
    identity=legacy_gate.diagnostic_identities(ev,formal,raw,cal,"formal",paths)
    req["Cdiag_bottom_driver_calibration_position_identity"]=identity["status"]=="VERIFIED"
    result["Cdiag_archive_identity"]=identity
    legacy_identity=legacy_gate.diagnostic_identities(ev,old,raw,cal,"formal/validation_state",ARMS)
    req["legacy_validation128_archive_identity"]=legacy_identity["status"]=="VERIFIED"
    result["legacy128_archive_identity"]=legacy_identity
    if not all(req.values()):return result
    controls=("M_NT",plan["selection"]["strongest_existing_baseline"])
    trigger=any((x.get("cap_rate") or 0)>.01 for x in groups["formal/select"]["arms"].values())
    result["cap4096_triggered"]=trigger
    if trigger:
        follow=groups["formal/select_cap4096"]
        req["triggered_fixed128_cap4096_complete"]=all(follow["arms"].get(a,{}).get("complete") for a in finalists)
        if not req["triggered_fixed128_cap4096_complete"]:return result
        directions={}
        for control in controls:
            pair=legacy_gate.paired_rows(raw,"formal/select_cap4096","M_ST",control,range(256,384))
            if not pair:return result
            left,right=pair;directions[control]={"relative_token_reduction":1-sum(left[i]["generated_tokens"] for i in left)/sum(right[i]["generated_tokens"] for i in right),
                **{k:sum(left[i][k]-right[i][k] for i in left)/len(left) for k in ("strict_correct","tolerant_correct")}}
        result["cap4096_direction_sensitivity"]=directions
        req["cap4096_no_adverse_point_direction"]=all(all(v>=0 for v in row.values()) for row in directions.values())
        if not req["cap4096_no_adverse_point_direction"]:return result
    if not allow_statistics:return {**result,"reason":"All structural predicates pass; new downstream statistics disabled by caller"}
    try:downstream=legacy_gate.downstream_statistics(raw,"formal/validate",controls,list(IDS),plan,cache_dir)
    except (Invalid,OSError,ValueError,ImportError,KeyError) as exc:return {**result,"downstream_error":str(exc),"reason":"Downstream exact/ratio unavailable: "+str(exc)}
    result["downstream"]=downstream
    req["four_simultaneous_quality_lower_bounds_positive_and_two_ratio_lower_bounds_at_least_5pct"]=downstream.get("passed") is True
    if all(req.values()):result.update(status="POSITIVE_EXPLORATORY",passed=True,reason="Original nonvalidation chain and full512 own-state endpoint pass; original128 significance is not a gate; no independent-confirmation claim")
    return result


def _supplement_module():
    if __package__:
        from . import state_study_validation512
    else:
        import state_study_validation512
    return state_study_validation512


def _self_hash(value,key):
    return value.get(key)==canonical({k:v for k,v in value.items() if k!=key})


def _not_required(base,raw,ev):
    """An actual formal scientific stop, never absence of update artifacts."""
    formal=base["diagnostics"]["formal"];summary=formal.get("summary") or {}
    stages=base["manifest"].get("stages",{})
    checks={"P3_completed":stages.get("P3",{}).get("status")=="complete",
        "P4_explicitly_not_triggered":stages.get("P4",{}).get("status")=="scientifically_not_triggered",
        "saved_formal_gate_not_triggered":summary.get("gate_recalibration") is False and summary.get("status") in ("NEGATIVE","INCONCLUSIVE"),
        "formal_expected_ids":summary.get("expected_prompt_ids")==sorted(str(i) for i in range(128,256)),
        "formal_archive_hashes":formal.get("archive_hashes_verified") is True,
        "formal_full_masks":all(formal.get("full_trace_masks_reconciled",{}).get(p) for p in ("A","B","C","D")),
        "formal_bootstrap_contract":legacy_gate.bootstrap_contract(summary),
        "P0_passed":base["preflight"].get("status") in ("PASS","PASSED") and base["preflight"].get("protocol_hash")==base["protocol"].get("protocol_hash")}
    if not all(checks.values()):return False,checks
    identity=legacy_gate.diagnostic_identities(ev,formal,raw,base["calibration"]["arms"],"formal",("A","B","C","D"))
    checks["formal_driver_calibration_position_identity"]=identity["status"]=="VERIFIED"
    return all(checks.values()),checks


def check_validation512(run_dir,plan_path=None,verify_hashes=True,*,context=None):
    """CPU/I/O acceptance only; never runs either state or cost bootstrap.

    ``complete`` records finished, valid work, including legitimately undefined
    intervals. ``positive_state_gate`` is separate. The ops terminal status
    must use complete/not_required, never the sign of an observed estimate.
    """
    run=Path(run_dir).resolve();report_root=Path('/scratch/jgu7/moe-state-study/reports')/run.name
    path=Path(plan_path) if plan_path else report_root/"validation512/plan.json"
    result={"state_status":"PLAN_PENDING","complete":False,"coverage_complete":False,"analysis_complete":False,
        "analysis_estimable":False,"positive_state_gate":False,"own_state_improves":False,
        "not_required":False,"plan_path":str(path),"errors":[],"requirements":{},
        "checked_at_utc":datetime.now(timezone.utc).isoformat(),"no_model_or_bootstrap_execution":True}
    if not path.is_file():return {**result,"reason":"Prospective full512 plan not published; no completion claim"}
    try:
        supplement=_supplement_module();plan=supplement.load_plan(run,path)
        directory=Path(supplement.output_dir(run,plan)).resolve()
        if plan.get("spec",{}).get("sample_ids")!=list(IDS) or plan["spec"].get("arms")!=list(ARMS):
            raise Invalid("Supplement roles/IDs differ from full512 contract")
    except Exception as exc:return {**result,"state_status":"BLOCKED","errors":["Plan/source verification: "+str(exc)]}
    result.update(plan_hash=plan["plan_hash"],output_dir=str(directory))
    base,raw,ev=context if context is not None else read_base_evidence(run,verify_hashes)
    if ev.errors or base["manifest"].get("status") in ("error","FAILED_IMPLEMENTATION") or base["preflight"].get("status") in ("FAILED_IMPLEMENTATION","FAILED","FAIL"):
        return {**result,"state_status":"BLOCKED","errors":list(ev.errors) or ["Original scientific implementation/runtime error"]}
    no_update,not_required_checks=_not_required(base,raw,ev)
    if no_update:return {**result,"state_status":"NOT_REQUIRED","not_required":True,"not_required_checks":not_required_checks,
        "reason":"Original formal P3/P4 explicitly did not trigger an update; full512 updated-state work is not applicable, not completed"}
    if base["manifest"].get("stages",{}).get("P4",{}).get("status")!="complete":
        return {**result,"state_status":"WAITING_P4","reason":"Updated model branch has not completed; absence is not NOT_REQUIRED"}
    try:snapshot,rows,calibrations=supplement.source_snapshot(run,plan)
    except (Pending,supplement.Pending,FileNotFoundError) as exc:return {**result,"state_status":"WAITING_CORE","reason":str(exc)}
    except Exception as exc:return {**result,"state_status":"BLOCKED","errors":["Actual generation/calibration source verification: "+str(exc)]}
    result["source_snapshot_hash"]=snapshot["snapshot_hash"]
    sev=legacy_report.Evidence(directory,verify_hashes)
    saved_snapshot=sev.json("source_snapshot.json")
    if saved_snapshot is None:return {**result,"state_status":"BLOCKED" if sev.errors else "INCOMPLETE","reason":"No valid committed full512 source snapshot","errors":sev.errors}
    req=result["requirements"]
    req["source_snapshot_matches_actual_inputs"]=saved_snapshot==snapshot
    raw_receipt=sev.json("raw_completion.json");completion=sev.json("completion.json")
    if raw_receipt is None or completion is None:
        return {**result,"state_status":"BLOCKED" if sev.errors else "INCOMPLETE","reason":"Full512 raw/completion receipts not both published","errors":sev.errors}
    req["raw_receipt_self_hash"]=_self_hash(raw_receipt,"raw_completion_hash")
    req["completion_self_hash"]=_self_hash(completion,"completion_hash")
    common={"plan_hash":plan["plan_hash"],"source_snapshot_hash":snapshot["snapshot_hash"],"expected_prompt_ids":list(IDS),"arms":list(ARMS)}
    req["raw_receipt_roles_and_source"]=all(raw_receipt.get(k)==v for k,v in {**common,"schema_version":"state_study_validation512.raw.v1"}.items())
    req["completion_roles_and_source"]=all(completion.get(k)==v for k,v in {**common,"schema_version":"state_study_validation512.complete.v1","completed_work":True,"raw_completion_hash":raw_receipt.get("raw_completion_hash")}.items())
    diagnostic=legacy_report.load_diagnostic(sev,"")
    req["saved_summary_hash_bound_to_completion"]=(directory/"summaries/diagnostic_summary.json").is_file() and completion.get("summary_sha256")==sha(directory/"summaries/diagnostic_summary.json")
    req["committed_diagnostic_inventory_and_hashes"]=diagnostic.get("archive_hashes_verified") is True
    replay_path=directory/"replay_checks.jsonl"
    replay_rows=sev.jsonl(replay_path);seen=set();replay_valid=True
    for row in replay_rows:
        sid,arm=row.get("sample_id"),row.get("path")
        if isinstance(sid,bool) or sid not in IDS or arm not in ARMS or (sid,arm) in seen:
            replay_valid=False;continue
        seen.add((sid,arm));trace=rows[arm][sid]["gen_token_ids"]
        expected={"protocol_hash":base["protocol"]["protocol_hash"],"driver_hash":legacy_gate.arm_hash(arm,calibrations),
            "calibration_hash":calibrations[arm]["calibration_hash"],"trace_hash":canonical(trace),
            "argmaxes":trace,"decode_forwards":len(trace)-1,"mask_hash":canonical([[j,t] for j,t in enumerate(trace[:-1],1)])}
        replay_valid=replay_valid and all(row.get(k)==v for k,v in expected.items())
    req["all1024_replay_checks_match_actual_own_traces"]=(replay_valid and seen=={(sid,a) for sid in IDS for a in ARMS})
    req["replay_checks_hash_bound_to_raw_receipt"]=replay_path.is_file() and raw_receipt.get("replay_checks_sha256")==sha(replay_path)
    # The supplementation module's deterministic receipts bind all bytes in
    # raw_completion to the actual archive pairs. Independent position checks
    # below additionally compare them to the real formal/validate traces.
    try:
        current_receipt=supplement.verify_raw_completion(directory,plan,snapshot,rows,calibrations)
        req["raw_receipt_archives_match_actual_commits"]=raw_receipt==current_receipt
    except Exception as exc:
        req["raw_receipt_archives_match_actual_commits"]=False;result["errors"].append("Committed archive receipt verification: "+str(exc))
    archives=verify_full512_archives(sev,diagnostic,rows,calibrations,base["protocol"]["protocol_hash"])
    result["archive_identity"]=archives;req["all512_each_actual_trace_driver_calibration_48layer_positions"]=archives["status"]=="VERIFIED"
    legacy_report.reconcile_state_masks(diagnostic,{"formal/validate":{a:list(rows[a].values()) for a in ARMS}},list(IDS),"unused","formal/validate")
    req["full512_summary_masks_match_actual_traces"]=all(diagnostic.get("full_trace_masks_reconciled",{}).get(a) for a in ARMS)
    analysis=full512_analysis_contract(diagnostic);result["analysis"]=analysis
    req.update({"saved_analysis_"+k:v for k,v in analysis["checks"].items()})
    result["errors"].extend(sev.errors);result["errors"].extend(archives["errors"])
    result["errors"].extend("Published full512 evidence fails: "+key for key,value in req.items() if not value)
    structural_keys=[k for k in req if not k.startswith("saved_analysis_")]
    result["coverage_complete"]=all(req[k] for k in structural_keys) and not result["errors"]
    result["analysis_complete"]=analysis["analysis_complete"]
    result["analysis_estimable"]=analysis["analysis_estimable"]
    result["complete"]=all(req.values()) and not result["errors"]
    result["own_state_improves"]=result["complete"] and analysis["own_state_improves"]
    result["positive_state_gate"]=result["own_state_improves"]
    result["state_status"]="VERIFIED" if result["complete"] else "BLOCKED" if result["errors"] else "INCOMPLETE"
    result["summary_path"]=str(directory/"summaries/diagnostic_summary.json")
    result["evidence_inventory"]=sev.inventory
    result["reason"]="Complete full512 work; effect direction and estimability are separately reported" if result["complete"] else "Incomplete or invalid full512 receipts/archives/analysis"
    return result


def render_report(base,result):
    state=result["validation512"];protocol=base["protocol"];run=Path(base["run_dir"])
    groups=base["generations"];freeze=base["validation_freeze"];formal=base["diagnostics"]["formal"]
    old=base["diagnostics"]["formal/validation_state"];analysis=state.get("analysis",{})
    historical=Path('/scratch/jgu7/moe-state-study/reports')/run.name/'RUN_REPORT.md'
    lines=["# 状态来源校准：全512验证状态补充与最终交付", "",
        f"读取时间：`{result['generated_at_utc']}`；原始 run：`{run}`。",
        f"最终科学判断：**{result['status']}**。{result['reason']}。",
        f"全512工作状态：`{state.get('state_status')}`；complete=`{state.get('complete',False)}`；analysis_estimable=`{state.get('analysis_estimable',False)}`；token主改善条件=`{state.get('positive_state_gate',False)}`。",
        "完成采集/分析不等于结果正向；512条自由生成也不证明512题状态已采集。立即EOS的零观察保留在全集，未定义区间不伪造成零或导致追结果式重跑。",
        f"原128子集九问报告保留在 {legacy_report.link(historical)}。它的状态范围与结论保持历史身份；本交付的验证状态主端点为两个更新模型各512题，旧128区间的方向/显著性不作否决。",
        "原18份模型测量源码、原5份质量分析源码/方案及原quality plan档案均不修改；本补充不解封P6，不声称历史独立或NI。", ""]
    for i,question in enumerate(legacy_report.QUESTIONS,1):
        lines += [f"## {i}. {question}",""]
        if i==1:
            config=protocol.get("config",{});runtime=base["manifest"].get("runtime",{})
            lines += [f"模型 `{config.get('model_id','PENDING')}`，revision `{config.get('model_revision','PENDING')}`，dtype `{config.get('precision','PENDING')}`，backend `{config.get('attention_backend','PENDING')}`，batch `{config.get('batch_size','PENDING')}`。",
                f"原protocol `{protocol.get('protocol_hash','PENDING')}`；原测量源码集合 `{protocol.get('source_hash','PENDING')}`；本补充plan `{state.get('plan_hash','PENDING')}`。",
                f"实际GPU `{runtime.get('gpu','未记录')}`；进程核验 `{base['process_verification'].get('status')}`，只对应读取时刻。"]
        elif i==2:
            lines += [f"P0保存状态 `{base['preflight'].get('status','PENDING')}`。本补充独立核验source snapshot、生成trace、参数与driver、commit/文件SHA、48层、T−1全位置数组及empty身份；不重新计算moment数组，不把元数据核验说成数值重算。",
                f"全512逐项要求：`{json.dumps(state.get('requirements',{}),ensure_ascii=False,sort_keys=True)}`。",
                f"身份/格式错误：`{json.dumps(result.get('errors',[]),ensure_ascii=False)}`。"]
        elif i==3:
            lines += legacy_report.generation_table(groups["formal/select"],legacy_gate.METHODS[:6])
            lines += [f"D_select reference=`{freeze.get('selected_reference','PENDING')}`，strongest=`{freeze.get('strongest_existing_baseline','PENDING')}`；选择仅按原冻结规则，不用D_validate改选。"]
        elif i==4:
            lines += [f"原正式C_diag状态 `{formal.get('status','PENDING')}`；P3数值门槛结果 `{result.get('requirements',{}).get('P3_numerical_state_source_gate','PENDING')}`。",
                "|C_diag全轨迹端点|E_moment保存95%区间|", "|---|---|"]
            for path in ("A","C","D"):
                lines.append(f"|{path}|{legacy_report.interval_text(legacy_report.state_metric(formal,path=path))}|")
            for contrast in ("B_minus_A","C_minus_D"):
                lines.append(f"|{contrast}|{legacy_report.interval_text(legacy_report.state_metric(formal,contrast=contrast))}|")
            lines += ["完整后缀与按位置段结果仍在原诊断摘要；共同位置补充不能替代全轨迹，C_diag的128题不能冒充D_validate512。"]
        elif i==5:
            lines += ["D_validate自由生成与状态证据分别列出：", "", "|对象|范围|实际证据|", "|---|---|---|"]
            for arm in ARMS:
                n=groups["formal/validate"]["arms"].get(arm,{}).get("n",0)
                lines.append(f"|{arm}自由生成|512:1024，共512题|{n}/512请求；独立于状态覆盖|")
            lines += [f"|两个更新模型自身状态|各512题/48层/全部有效decode|{state.get('archive_identity',{}).get('verified_prompt_paths',0)}/1024已核验prompt/path；完整性{state.get('coverage_complete',False)}|",
                f"全512 token主差 E(M_ST own)−E(M_NT own)：{legacy_report.interval_text(analysis.get('token'))}。",
                f"全512 prompt等权敏感性：{legacy_report.interval_text(analysis.get('prompt'))}；不临时增加方向否决。",
                f"原128子集保存差：{legacy_report.interval_text(legacy_gate.state_contrast(old,OWN).get('metrics',{}).get('E_moment'))}。该区间仍披露，但不参与新主改善条件。",
                "只有全512完整并且保存token主95%区间上界<0才建立此状态改善条件；不显著、不可估或反向都不会被改写成改善。"]
        elif i==6:
            cals=base["calibration"]["arms"]
            lines += [f"同文本/公式/mask/计数/source链：`{result.get('requirements',{}).get('same_formula_mask_budget_source_provenance','PENDING')}`。所有拟合仍只来自C_fit。",
                "|校准臂|拟合有效decode token|source driver|参数hash|", "|---|---:|---|---|"]
            for arm in ("M_N","L_N",*ARMS):
                cal=cals.get(arm,{});meta=cal.get("metadata",{})
                lines.append(f"|{arm}|{meta.get('fit_decode_token_count','PENDING')}|{meta.get('source_driver',meta.get('source','PENDING'))}|{cal.get('calibration_hash','PENDING')}|")
            lines += ["相同拟合token数不等于同总离线预算。全512补充直接回放已保存正式自由生成trace，不生成新文本、不重估参数；全部512回放（含旧128重复诊断）的额外时间和shadow成本须单列。",
                f"补充成本证据目录：`{state.get('output_dir','PENDING')}/cost_attempts/`。原生成计时保持无observer作用域。"]
        elif i==7:
            lines += legacy_report.generation_table(groups["formal/validate"],freeze.get("finalist_arms",()))
            lines += legacy_report.cost_table(groups["formal/validate"])
            lines += legacy_report.comparison_lines(groups["formal/validate"])
            lines += legacy_report.positive_gate_tables(result)
            lines += ["行政cap成本仍有删失，EOS恰在cap位置归自然结束；不外推cap之后未知自然长度。Strict/tolerant并列，真实秒数来自相同后端原请求，不能推出部署系统加速。",
                f"4096触发与同向敏感性：`{result.get('cap4096_triggered','PENDING')}` / `{json.dumps(result.get('cap4096_direction_sensitivity',{}),ensure_ascii=False)}`；仍沿用原前瞻质量补充规则。"]
        elif i==8:
            lines += [f"**{result['status']}**：{result['reason']}。",
                "正向链逐项重用原质量计划的P0、P3、C_diag、同文本拟合、8臂选择、冻结复核、4096、四项同时精确质量下界和两项≥5% paired ratio成本下界。仅验证状态主范围/端点换成严格全512；不是旧128 positive AND 新512，也不修改旧档案。",
                "质量仍为原前瞻补充的明确优越性，原NI-only条件和None未被偷偷满足。质量下界为保守exact-binomial联合界；成本覆盖为近似percentile-bootstrap，不是有限样本exact。未计算或未通过任何条件都不宣称正向。",
                f"所有门槛：`{json.dumps(result.get('requirements',{}),ensure_ascii=False,sort_keys=True)}`。",
                "NOT_REQUIRED仅表示正式P3未触发更新，不表示全512状态完成或科学负结果。P6/P7与历史独立性、功效及重拟合不确定性仍须另审计。"]
        else:
            lines += [f"先依据 `{state.get('state_status')}` 补齐未完成工作；数据/分析完整但效应不支持时报告真实负向或不确定结果，不重放以追求有利区间。",
                "本报告的精确CPU只读验收命令（不运行模型或bootstrap）：", "", "```sh",
                f"PYTHONDONTWRITEBYTECODE=1 /scratch/jgu7/envs/moe-state/bin/python scripts/state_study_validation512_report.py --run-dir {run} --plan {state['plan_path']} --check-only",
                "```", "生成最终报告时显式允许下游统计，仍只在全部先决证据完成后复用或建立原同输入缓存；本脚本不会发起模型或Slurm作业。"]
        lines.append("")
    lines += ["## 原门槛到全512门槛映射", "", "|原predicate|本补充处理|", "|---|---|"]
    lines.extend(f"|`{key}`|{value}|" for key,value in PREDICATE_MAPPING.items())
    lines += ["", "底层证据清单、SHA与receipt关系保留在 validation512_summary.json；原128报告和原run只读保留。", ""]
    return "\n".join(lines)


def build_report(run_dir,plan_path=None,quality_plan_path=None,*,verify_hashes=True,allow_statistics=False,cache_dir=None):
    context=read_base_evidence(run_dir,verify_hashes);base,raw,ev=context
    state=check_validation512(run_dir,plan_path,verify_hashes,context=context)
    root=Path('/scratch/jgu7/moe-state-study/reports')/Path(run_dir).resolve().name
    quality=Path(quality_plan_path) if quality_plan_path else root/'quality_plan.json'
    # Same original mathematical plan and same actual paired inputs preserve
    # the original cache key. Full512 changes a state endpoint, not these data.
    cache=Path(cache_dir) if cache_dir else root
    result=evaluate_validation512_result(base,raw,ev,state,quality,cache,allow_statistics)
    result.update(schema_version="state_study_validation512.report.v1",generated_at_utc=datetime.now(timezone.utc).isoformat(),
        run_dir=str(Path(run_dir).resolve()),errors=list(ev.errors)+list(state.get("errors",[])),
        analysis_contract={"original_18_and_5_modified":False,"original128_CI_is_not_a_primary_veto":True,
            "full512_source":"original formal/validate two updated-arm traces, not newly generated text",
            "new_model_or_state_bootstrap_execution":False,"downstream_statistics_explicitly_allowed":allow_statistics,
            "quality_cost_cache_uses_original_plan_and_inputs":True,"new_primary_weighting":"token; prompt equal-weight sensitivity disclosed",
            "completion_is_independent_of_effect_direction_or_estimability":True,"no_P6_authorization":True},
        base_evidence_inventory=ev.inventory)
    return result,render_report(base,result)


def _quality_identity(run):
    path=Path('/scratch/jgu7/moe-state-study/reports')/Path(run).name/'quality_plan.json'
    return {"path":str(path),"exists":path.is_file(),"sha256":sha(path) if path.is_file() else None}


def _work_identity(run,directory,proof):
    if proof.get('complete') is True:
        return {"kind":"FULL512_COMPLETE","completion_sha256":sha(Path(directory)/'completion.json'),
            "raw_completion_sha256":sha(Path(directory)/'raw_completion.json'),
            "source_snapshot_hash":proof.get('source_snapshot_hash')}
    if proof.get('not_required') is True:
        manifest=load(Path(run)/'manifest.json')
        if manifest.get('stages',{}).get('P5',{}).get('status')!='complete' or not (Path(run)/'RUN_REPORT.md').is_file():
            raise Pending('NOT_REQUIRED branch still needs the original P5 and report to finish')
        paths=('formal/summaries/diagnostic_summary.json','formal/validation_state/summaries/diagnostic_summary.json','validation_freeze.json')
        return {"kind":"NOT_REQUIRED","note":"No updated512 work claimed",
            "core_negative_branch_evidence":{name:sha(Path(run)/name) for name in paths}}
    raise Pending('Full512 work is neither complete nor explicitly not required')


def _atomic_text(path,body):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.next')
    try:
        with temporary.open('x',encoding='utf-8') as handle:
            handle.write(body);handle.flush();os.fsync(handle.fileno())
        temporary.replace(path)
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:temporary.unlink(missing_ok=True)


def verify_delivery(run,plan_path=None,*,work_proof=None):
    """Verify actual nine-question report bytes after work, never bootstrap."""
    result={"verified":False,"delivery_complete":False,"status":"MISSING","errors":[],"no_model_or_bootstrap_execution":True}
    try:
        supplement=_supplement_module();plan=supplement.load_plan(Path(run).resolve(),plan_path)
        directory=Path(supplement.output_dir(run,plan)).resolve();delivery=directory/'delivery'
        receipt_path=delivery/'delivery_receipt.json'
        result.update(delivery_dir=str(delivery),receipt_path=str(receipt_path),plan_hash=plan['plan_hash'])
        if not receipt_path.is_file():return {**result,"reason":"Final full512/NOT_REQUIRED delivery receipt is absent"}
        proof=work_proof if work_proof is not None else check_validation512(run,plan_path,True)
        if not (proof.get('complete') is True or proof.get('not_required') is True):
            return {**result,"status":"PENDING","reason":"Work has not passed full512/NOT_REQUIRED acceptance","work_status":proof.get('state_status')}
        receipt=load(receipt_path)
        if not _self_hash(receipt,'delivery_hash') or receipt.get('schema_version')!='state_study_validation512.delivery.v1':
            raise Invalid('Delivery receipt self hash/schema differs')
        if receipt.get('plan_hash')!=plan['plan_hash'] or receipt.get('work')!=_work_identity(run,directory,proof) or receipt.get('quality_identity')!=_quality_identity(run):
            raise Invalid('Delivery plan/current work/quality identity differs')
        expected_names={'RUN_REPORT.md','VALIDATION512_REPORT.md','validation512_summary.json'}
        files=receipt.get('files',{})
        if set(files)!=expected_names:raise Invalid('Delivery file inventory differs')
        expected_bundle=canonical({'plan_hash':plan['plan_hash'],'work':receipt['work'],'quality_identity':receipt['quality_identity'],'files':files})
        if receipt.get('bundle_hash')!=expected_bundle:raise Invalid('Delivery bundle identity differs')
        archive=delivery/'receipts'/(receipt['delivery_hash']+'.json')
        if not archive.is_file() or load(archive)!=receipt:raise Invalid('Immutable delivery receipt archive differs')
        bundle=delivery/'bundles'/receipt['bundle_hash']
        if not bundle.is_dir():raise Invalid('Immutable delivery bundle is missing')
        for name,digest in files.items():
            if not (delivery/name).is_file() or sha(delivery/name)!=digest or not (bundle/name).is_file() or sha(bundle/name)!=digest:
                raise Invalid('Actual delivery/bundle file SHA differs: '+name)
        markdown=(delivery/'RUN_REPORT.md').read_text()
        if (delivery/'RUN_REPORT.md').read_bytes()!=(delivery/'VALIDATION512_REPORT.md').read_bytes() or not all('## '+str(i)+'.' in markdown for i in range(1,10)):
            raise Invalid('Delivery does not contain identical full nine-question reports')
        summary=load(delivery/'validation512_summary.json')
        if summary.get('schema_version')!='state_study_validation512.report.v1' or summary.get('run_dir')!=str(Path(run).resolve()) or summary.get('validation512',{}).get('plan_hash')!=plan['plan_hash']:
            raise Invalid('Delivered summary run/plan/schema differs')
        if summary.get('status')=='POSITIVE_EXPLORATORY' and (not proof.get('positive_state_gate') or summary.get('passed') is not True or not receipt['quality_identity']['exists']):
            raise Invalid('Delivered positive label lacks full512 improvement/final-quality proof')
        return {**result,"verified":True,"delivery_complete":True,"status":"VERIFIED","scientific_status":summary.get('status'),
            "receipt_sha256":sha(receipt_path),"delivery_hash":receipt['delivery_hash'],"report_path":str(delivery/'RUN_REPORT.md'),
            "reason":"Finished work and actual complete nine-question report/summary hashes verified"}
    except (Pending,FileNotFoundError) as exc:return {**result,"status":"PENDING","reason":str(exc)}
    except Exception as exc:return {**result,"status":"INVALID","errors":[str(exc)],"reason":"Delivery/source identity verification failed"}


def ensure_delivery(run,plan_path=None):
    """Sole CPU delivery writer; cache reuse avoids repeated real statistics."""
    previous=verify_delivery(run,plan_path)
    if previous.get('delivery_complete') is True:return {**previous,"reused":True}
    try:
        supplement=_supplement_module();plan=supplement.load_plan(Path(run).resolve(),plan_path)
        directory=Path(supplement.output_dir(run,plan)).resolve();delivery=directory/'delivery'
        result,markdown=build_report(run,plan_path,allow_statistics=True)
        proof=result['validation512']
        if not (proof.get('complete') is True or proof.get('not_required') is True):
            return {"verified":False,"delivery_complete":False,"status":"PENDING","reason":"Full512 work/NOT_REQUIRED is not verified","work_status":proof.get('state_status')}
        if result.get('downstream_error'):
            return {"verified":False,"delivery_complete":False,"status":"PENDING","reason":"Required downstream report analysis could not be persisted: "+result['downstream_error']}
        work=_work_identity(run,directory,proof);quality=_quality_identity(run)
        bodies={'RUN_REPORT.md':markdown,'VALIDATION512_REPORT.md':markdown,'validation512_summary.json':json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'}
        digests={name:hashlib.sha256(body.encode()).hexdigest() for name,body in bodies.items()}
        bundle_hash=canonical({'plan_hash':plan['plan_hash'],'work':work,'quality_identity':quality,'files':digests})
        bundle=delivery/'bundles'/bundle_hash;bundle.mkdir(parents=True,exist_ok=True)
        for name,body in bodies.items():
            target=bundle/name
            if target.exists():
                if sha(target)!=digests[name]:raise Invalid('Immutable delivery bundle differs: '+name)
            else:
                with target.open('x',encoding='utf-8') as handle:handle.write(body);handle.flush();os.fsync(handle.fileno())
            _atomic_text(delivery/name,body)
        value={'schema_version':'state_study_validation512.delivery.v1','plan_hash':plan['plan_hash'],'work':work,
            'quality_identity':quality,'files':digests,'bundle_hash':bundle_hash,'scientific_status':result['status'],
            'generated_at_utc':result['generated_at_utc'],'original128_report_preserved':True,'no_P6_authorization':True}
        value['delivery_hash']=canonical(value)
        archive=delivery/'receipts'/(value['delivery_hash']+'.json')
        if archive.is_file():
            if load(archive)!=value:raise Invalid('Immutable delivery receipt differs')
        else:publish_once(archive,value)
        _atomic_text(delivery/'delivery_receipt.json',json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        verification=verify_delivery(run,plan_path,work_proof=proof)
        return {**verification,'reused':False}
    except Exception as exc:
        return {'verified':False,'delivery_complete':False,'status':'PENDING','reason':'CPU delivery incomplete; preserve raw/analysis and resume reporting only','errors':[str(exc)]}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True);parser.add_argument('--plan')
    parser.add_argument('--quality-plan');parser.add_argument('--output-dir')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--allow-downstream-statistics',action='store_true',help='Allow original quality/ratio calculation/cache only after all full512 prerequisites pass')
    args=parser.parse_args(argv);run=Path(args.run_dir).resolve()
    if args.check_only:
        proof=check_validation512(run,args.plan,True)
        print(json.dumps(proof,ensure_ascii=False,allow_nan=False))
        return 0 if proof.get('complete') or proof.get('not_required') else 3 if proof.get('state_status')=='BLOCKED' else 2
    if not args.output_dir:parser.error('--output-dir is required for derived reports')
    output=Path(args.output_dir).resolve()
    if output.is_relative_to(run) or output.is_relative_to(Path.home().resolve()):parser.error('Derived report must be outside original run and Home')
    result,markdown=build_report(run,args.plan,args.quality_plan,allow_statistics=args.allow_downstream_statistics)
    output.mkdir(parents=True,exist_ok=True)
    for name,body in (("RUN_REPORT.md",markdown),("VALIDATION512_REPORT.md",markdown),("validation512_summary.json",json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")):
        temporary=output/('.'+name+'.next');temporary.write_text(body);temporary.replace(output/name)
    print(json.dumps({'status':result['status'],'state_status':result['validation512']['state_status'],'report':str(output/'RUN_REPORT.md')}))
    return 0


if __name__=='__main__':raise SystemExit(main())
