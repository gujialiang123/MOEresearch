"""Independent evidence chain for a prospectively amended exploratory result.

This module reads saved identities/metadata, never executes a model or refits a
calibrator. Existing state-bootstrap intervals are checked, never recomputed.
New downstream ratio bootstraps run once only after all prerequisites hold.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

if __package__:
    from .freeze_state_study_quality_plan import Invalid, Pending, canonical, load, publish_once, sha, verify_plan
else:
    from freeze_state_study_quality_plan import Invalid, Pending, canonical, load, publish_once, sha, verify_plan

METHODS=("N","F","R8","R16","M_N","L_N","M_NT","M_ST")
FIELDS=("question_hash","protocol_hash","parser_version","model_revision","model_hash",
        "dtype","max_new_tokens","rendered_prompt","prompt_token_ids")


def finite(x):
    return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)


def arm_hash(arm,calibrations):
    ref={"N":8,"F":4,"R8":8,"R16":16,"M_N":8,"L_N":4,"M_NT":4,"M_ST":4}[arm]
    return canonical({"arm":arm,"k_compute":8 if arm=="N" else 4,"k_reference":ref,
        "calibration_hash":calibrations.get(arm,{}).get("calibration_hash"),"prefill":"native","seed":20261019})


def row_map(raw,scope,arm,ids):
    rows=raw.get(scope,{}).get(arm,[])
    result={row.get("sample_id"):row for row in rows}
    if len(result)!=len(rows) or set(result)!=set(ids):return None
    return result


def paired_rows(raw,scope,left,right,ids):
    l,r=row_map(raw,scope,left,ids),row_map(raw,scope,right,ids)
    if not l or not r:return None
    if any(l[i].get(k)!=r[i].get(k) for i in ids for k in FIELDS):return None
    return l,r


def good_interval(value):
    return isinstance(value,dict) and value.get("status")=="ok" and all(finite(value.get(k)) for k in ("estimate","ci_low","ci_high"))


def state_contrast(diagnostic,name):
    return (diagnostic.get("summary") or {}).get("analyses",{}).get("all",{}).get("token",{}).get("comparisons",{}).get(name,{})


def p3_numerical_gate(diagnostic,protocol):
    summary=diagnostic.get("summary") or {};block=summary.get("analyses",{}).get("all",{}).get("token",{})
    paths=block.get("paths",{});a=paths.get("A",{}).get("intervals",{}).get("E_moment")
    c=paths.get("C",{}).get("intervals",{}).get("E_moment");d=paths.get("D",{}).get("intervals",{}).get("E_moment")
    cd=state_contrast(diagnostic,"C_minus_D");ba=state_contrast(diagnostic,"B_minus_A")
    delta=cd.get("metrics",{}).get("E_moment");config=protocol.get("config",{})
    native=config.get("native_match_tolerance");state=config.get("state_effect_tolerance")
    decision=summary.get("decision",{});policy=summary.get("decision_protocol",{})
    return (summary.get("status")=="STATE_MISMATCH_SUPPORTED" and summary.get("gate_recalibration") is True
        and summary.get("chosen_mismatch_metric")=="E_moment" and summary.get("family")=="lda"
        and all(decision.get(key) is True for key in ("native_matching_established","student_deployment_mismatch_established","state_effect_exceeds_practical_bound"))
        and policy.get("native_match_tolerance")==native and policy.get("state_effect_tolerance")==state
        and finite(native) and finite(state) and all(good_interval(x) for x in (a,c,d,delta))
        and cd.get("status")=="ok" and ba.get("status")=="ok"
        and good_interval(ba.get("metrics",{}).get("E_moment"))
        and a["ci_high"]<=native and c["ci_low"]>native and delta["ci_low"]>state)


def bootstrap_contract(summary):
    bootstrap=summary.get("bootstrap",{})
    return all(bootstrap.get(k)==v for k,v in {"unit":"prompt","paired":True,"resamples":10000,
        "seed":20261019,"confidence_level":.95,"calibration_parameters_refitted":False,
        "includes_empty_and_inactive_prompt_ids":True}.items())


def own_state_improves(diagnostic):
    comparison=state_contrast(diagnostic,"M_ST_minus_M_NT_own_trajectories")
    value=comparison.get("metrics",{}).get("E_moment")
    return (comparison.get("status")=="ok" and comparison.get("prompt_count")==128
            and comparison.get("same_calibration_required") is False
            and comparison.get("same_mask_required") is False
            and good_interval(value) and value["ci_high"]<0)


def generation_identities(report,raw,calibrations):
    """Bind every formal row to its frozen dataset prompt, arm and true cap."""
    protocol=report["protocol"];config=protocol.get("config",{});path=Path(config["data_dir"])/"train.jsonl"
    expected_sha=protocol.get("dataset",{}).get("sources",{}).get("train",{}).get("jsonl_sha256")
    if not expected_sha or sha(path)!=expected_sha:return False
    questions={}
    with path.open("rb") as handle:
        for line in handle:
            row=json.loads(line)
            if row.get("sample_id") in range(1024):questions[row["sample_id"]]=row
    eos=set(report.get("manifest",{}).get("runtime",{}).get("eos_ids",[]))
    if not eos:return False
    scopes=[("formal/select",METHODS,range(256,512),2048),
        ("formal/validate",report["validation_freeze"].get("finalist_arms",[]),range(512,1024),2048),
        ("formal/fit_native",("N",),range(128),2048),
        ("formal/fit_student",("L_N",),range(128),2048),
        ("formal/diag_rollouts",("N","L_N"),range(128,256),2048),
        ("formal/updated_diag_rollouts",("M_NT","M_ST"),range(128,256),2048),
        ("formal/validation_state/updated_diag_rollouts",("M_NT","M_ST"),range(512,640),2048)]
    if any((data.get("cap_rate") or 0)>.01 for data in report["generations"]["formal/select"]["arms"].values()):
        scopes.append(("formal/select_cap4096",report["validation_freeze"]["finalist_arms"],range(256,384),4096))
    for scope,arms,ids,cap in scopes:
        for arm in arms:
            rows=row_map(raw,scope,arm,ids)
            if not rows:return False
            for sid,row in rows.items():
                sample=questions.get(sid,{})
                if any(row.get(k)!=v for k,v in {"scope":scope,"arm":arm,"arm_hash":arm_hash(arm,calibrations),
                    "calibration_hash":calibrations.get(arm,{}).get("calibration_hash"),"max_new_tokens":cap,
                    "protocol_hash":protocol["protocol_hash"],"parser_version":protocol["parser_version"],
                    "model_revision":config["model_revision"],"dataset_revision":config["dataset_revision"],
                    "question":sample.get("question"),"question_hash":sample.get("question_hash")}.items()):return False
                if row.get("observer_enabled") is not False or row.get("timing_debug_logits") is not False or set(row.get("eos_ids",[]))!=eos:return False
                tokens=row.get("gen_token_ids",[]);n=len(tokens)
                if not 1<=n<=cap or row.get("generated_tokens")!=n or row.get("decode_forward_count")!=n-1:return False
                if any(token in eos for token in tokens[:-1]):return False
                if row.get("stop_reason")=="eos":
                    if tokens[-1] not in eos or row.get("eos_position")!=n-1:return False
                elif row.get("stop_reason")=="max_new_tokens":
                    if n!=cap or tokens[-1] in eos or row.get("eos_position") is not None:return False
                else:return False
                if not all(isinstance(row.get(k),bool) for k in ("strict_correct","tolerant_correct")):return False
    return True


def calibration_provenance(evidence,raw,calibrations):
    """Count/mask/source checks; this does not re-estimate any calibration."""
    proof={"status":"INCOMPLETE","checks":{},"errors":[]};checks=proof["checks"]
    if not all(arm in calibrations for arm in ("M_N","L_N","M_NT","M_ST")):return proof
    ids=list(range(128));native=row_map(raw,"formal/fit_native","N",ids)
    student=row_map(raw,"formal/fit_student","L_N",ids)
    if not native or not student:return proof
    native_mask=evidence.json("formal/calibration/fit_position_masks.json")
    student_mask=evidence.json("formal/calibration/student_position_masks.json")
    if not native_mask or not student_mask:return proof
    for label,rows,saved in (("native",native,native_mask),("student",student,student_mask)):
        positions={str(i):[[j,t] for j,t in enumerate(rows[i]["gen_token_ids"][:-1],1)] for i in ids}
        checks[label+"_actual_trace_mask"]=(saved.get("positions")==positions and saved.get("mask_hash")==canonical(positions))
    formula=("family","reduced_branch","application_phase","application_dtype","statistics_dtype","ddof","operation_order","numerical_protection")
    ln=calibrations["L_N"]
    checks["same_lda_formula_and_precision"]=all(calibrations[a].get(k)==ln.get(k) for a in ("M_NT","M_ST") for k in formula) and ln.get("family")=="lda"
    for arm in ("M_N","L_N","M_NT","M_ST"):
        cal=calibrations[arm];meta=cal["metadata"];is_updated=arm in ("M_NT","M_ST")
        rows=student if is_updated else native;saved=student_mask if is_updated else native_mask
        lengths={str(i):len(rows[i]["gen_token_ids"])-1 for i in ids};total=sum(lengths.values())
        counts=meta.get("counts_per_layer",{});per_prompt=meta.get("counts_per_prompt_layer",{})
        checks[arm+"_count_rectangle"]=(total>0 and set(counts)=={str(i) for i in range(48)} and all(v==total for v in counts.values())
            and set(per_prompt)=={i for i,n in lengths.items() if n>0}
            and all(set(per_prompt[i])=={str(j) for j in range(48)} and all(v==lengths[i] for v in per_prompt[i].values()) for i in per_prompt))
        checks[arm+"_source_mask_counts"]=(meta.get("prompt_ids")==ids and meta.get("max_new_tokens")==2048
            and meta.get("fit_mask_hash")==saved.get("mask_hash") and meta.get("fit_decode_token_count")==total
            and meta.get("rollout_token_count")==sum(len(row["gen_token_ids"]) for row in rows.values()))
        if is_updated:
            driver="N" if arm=="M_NT" else "L_N"
            checks[arm+"_driver_identity"]=(meta.get("source")==arm and meta.get("source_driver")==driver
                and meta.get("source_driver_hash")==arm_hash(driver,calibrations)
                and meta.get("source_student_calibration_hash")==ln["calibration_hash"])
        else:
            checks[arm+"_native_source"]=(meta.get("source")=="N on native C_fit text" and meta.get("scope")=="formal"
                and cal.get("family")==('scalar' if arm=="M_N" else 'lda'))
    proof["status"]="VERIFIED" if all(checks.values()) else "UNVERIFIED"
    return proof


def expected_diagnostics(raw,calibrations,scope):
    original=scope+"/diag_rollouts";updated=scope+"/updated_diag_rollouts"
    ids=list(range(128,256)) if scope=="formal" else list(range(512,640));result={}
    for path,driver,text_arm in (("A","N","N"),("B","L_N","N"),("C","L_N","L_N"),("D","N","L_N")):
        rows=row_map(raw,original,text_arm,ids)
        if rows and "L_N" in calibrations:
            result[path]={"rows":rows,"driver_hash":arm_hash(driver,calibrations),"calibration_hash":calibrations["L_N"]["calibration_hash"]}
    for arm in ("M_NT","M_ST"):
        rows=row_map(raw,updated,arm,ids)
        if rows and arm in calibrations:
            result[arm]={"rows":rows,"driver_hash":arm_hash(arm,calibrations),"calibration_hash":calibrations[arm]["calibration_hash"]}
    return result


def diagnostic_identities(evidence,diagnostic,raw,calibrations,scope,required_paths):
    """Read only NPZ metadata/position arrays, not moment arrays or a bootstrap."""
    proof={"status":"INCOMPLETE","metadata_only_no_moment_recomputation":True,"errors":[],"paths_verified":[]}
    if not diagnostic.get("archive_hashes_verified") or not diagnostic.get("current_inventory_matches"):return proof
    expected=expected_diagnostics(raw,calibrations,scope)
    if any(path not in expected for path in required_paths):return proof
    try:import numpy as np
    except ImportError:
        proof["errors"].append("NumPy unavailable for safe allow_pickle=False metadata reads");return proof
    directory=evidence.run/scope/"diagnostics";seen=set();expected_layers={str(i) for i in range(48)}
    for archive_path in sorted(directory.glob("*.npz")):
        commit=evidence.json(archive_path.with_suffix(".complete.json"))
        if not commit:proof["errors"].append("Missing committed archive");continue
        token_path=directory/commit.get("tokens","")
        inv=evidence.inventory.get(str(token_path.relative_to(evidence.run)),{})
        if inv.get("physical_lines")!=commit.get("token_rows"):proof["errors"].append("Token physical-line count differs from commit")
        try:
            with np.load(archive_path,allow_pickle=False) as archive:
                metadata=json.loads(str(archive["metadata_json"]))
                if any(metadata.get(k)!=v for k,v in {"schema_version":"state_collection.v2","dtype":"float64","ddof":0,"moment_representation":"count_mean_centered_m2"}.items()):raise ValueError("numerical contract differs")
                if metadata.get("numerical_protection")!=calibrations["L_N"].get("numerical_protection"):raise ValueError("diagnostic numerical protection differs")
                entries=metadata.get("records",[]);all_entries=[x for x in entries if x["key"][3]=="all"]
                if not all_entries:raise ValueError("archive lacks all-position records")
                pid,path=all_entries[0]["key"][:2]
                if path not in expected:raise ValueError("archive path lacks expected complete generation source")
                target=expected[path];row=target["rows"].get(int(pid))
                if row is None:raise ValueError("archive prompt not in expected ID set")
                trace=row["gen_token_ids"];positions=[[j,t] for j,t in enumerate(trace[:-1],1)]
                if not positions or archive_path.stem!=canonical([str(pid),path]):raise ValueError("archive filename/empty trace differs")
                if {str(x["key"][2]) for x in all_entries}!=expected_layers or len(all_entries)!=48:raise ValueError("archive all-position layer rectangle differs")
                if sum(x["count"] for x in all_entries)!=commit.get("token_rows"):raise ValueError("archive/commit layer-token count differs")
                for entry in entries:
                    epid,epath,layer,group=entry["key"]
                    if str(epid)!=str(pid) or epath!=path or not 0<=int(layer)<48:raise ValueError("mixed prompt/path/layer identity")
                    if entry.get("driver_hash")!=target["driver_hash"] or entry.get("calibration_hash")!=target["calibration_hash"]:raise ValueError("driver/calibration hash differs")
                    selected=positions
                    if group.startswith("common"):
                        native=row_map(raw,scope+"/diag_rollouts","N",target["rows"])
                        student=row_map(raw,scope+"/diag_rollouts","L_N",target["rows"])
                        if not native or not student:raise ValueError("common mask generation source absent")
                        limit=min(len(native[int(pid)]["gen_token_ids"]),len(student[int(pid)]["gen_token_ids"]))-1
                        selected=[x for x in selected if x[0]<=limit]
                        group=group.partition(":")[2] or "all"
                    if group!="all":
                        parts=group.split("-")
                        if len(parts)!=2 or not all(x.isdigit() for x in parts):raise ValueError("unknown position group")
                        lo,hi=map(int,parts);selected=[x for x in selected if lo<=x[0]<=hi]
                    if entry.get("count")!=len(selected) or entry.get("mask_hash")!=canonical(selected):raise ValueError("actual trace count/mask differs")
                    actual=archive[entry["prefix"]+"_positions"]
                    if actual.dtype!=np.int64 or actual.shape!=(len(selected),2) or actual.tolist()!=selected:raise ValueError("stored position array differs from actual trace")
                if (str(pid),path) in seen:raise ValueError("duplicate prompt/path archive")
                seen.add((str(pid),path))
        except (OSError,ValueError,KeyError,TypeError) as exc:proof["errors"].append(f"{archive_path.name}: {exc}")
    for marker_path in sorted(directory.glob("*.empty.json")):
        marker=evidence.json(marker_path)
        if not marker:continue
        pid,path=str(marker.get("prompt_id")),marker.get("path")
        if path not in expected:proof["errors"].append("empty marker path unexpected");continue
        target=expected[path];row=target["rows"].get(int(pid)) if pid.isdigit() else None
        identity={"protocol_hash":(diagnostic.get("summary") or {}).get("protocol_hash",None),"driver_hash":target["driver_hash"],"calibration_hash":target["calibration_hash"]}
        identity["protocol_hash"]=next(iter(target["rows"].values())).get("protocol_hash")
        if row:identity["trace_hash"]=canonical(row["gen_token_ids"])
        if (not row or len(row["gen_token_ids"])!=1 or marker.get("schema_version")!="state_collection.empty.v1"
            or marker.get("collection_schema_version")!="state_collection.v2" or marker.get("dtype")!="float64" or marker.get("ddof")!=0
            or marker.get("count")!=0 or not marker.get("reason") or marker.get("metadata")!=identity
            or marker_path.stem.removesuffix(".empty")!=canonical([pid,path]) or (pid,path) in seen):
            proof["errors"].append("empty marker source/identity differs")
        seen.add((pid,path))
    for path in required_paths:
        ids={str(i) for i in expected[path]["rows"]}
        if {pid for pid,actual_path in seen if actual_path==path}==ids:proof["paths_verified"].append(path)
    proof["status"]="VERIFIED" if not proof["errors"] and set(proof["paths_verified"])==set(required_paths) else "UNVERIFIED"
    return proof


def validation_contract(report,plan):
    freeze=report.get("validation_freeze",{});selection=plan["selection"]
    expected={"protocol_hash":plan["protocol_hash"],"selected_reference":selection["selected_reference"],
        "strongest_existing_baseline":selection["strongest_existing_baseline"],"initial_calibration":"L_N",
        "selection_sample_ids":list(range(256,512)),"validation_sample_ids":list(range(512,1024)),
        "minimum_useful_relative_token_reduction":.05,"quality_noninferiority_margin":None,
        "updated":True,"max_method_updates":1,"followup_trigger":.01,
        "followup_prompt_ids":list(range(256,384)),"followup_cap":4096}
    return all(freeze.get(k)==v for k,v in expected.items()) and {"N","L_N","M_NT","M_ST",selection["selected_reference"],selection["strongest_existing_baseline"]}==set(freeze.get("finalist_arms",[]))


def downstream_statistics(raw,scope,controls,ids,plan,cache_dir=None):
    if __package__:
        from .state_study_positive_gate import quality_superiority,paired_relative_token_reduction
    else:
        from state_study_positive_gate import quality_superiority,paired_relative_token_reduction
    pairs={control:paired_rows(raw,scope,"M_ST",control,ids) for control in controls}
    if not all(pairs.values()):return {"status":"INCOMPLETE_PAIRS","passed":False}
    data_hash=canonical({control:[[pairs[control][0][i][k],pairs[control][1][i][k]] for i in ids for k in ("strict_correct","tolerant_correct","generated_tokens")] for control in controls})
    key=canonical({"plan_hash":plan["plan_hash"],"scope":scope,"ids":list(ids),"data_hash":data_hash})
    cache=Path(cache_dir)/("positive_downstream."+key+".json") if cache_dir else None
    if cache and cache.is_file():
        value=load(cache)
        if value.get("cache_key")!=key or value.get("result_hash")!=canonical({k:v for k,v in value.items() if k!="result_hash"}):raise Invalid("Positive downstream cache identity/hash differs")
        return {**value,"cache_reused":True}
    quality={};cost={}
    for control,(left,right) in pairs.items():
        for score in ("strict_correct","tolerant_correct"):
            quality[control+":"+score]=quality_superiority([left[i][score] for i in ids],[right[i][score] for i in ids],family_alpha=.05,family_comparisons=4)
        cost[control]=paired_relative_token_reduction([left[i]["generated_tokens"] for i in ids],[right[i]["generated_tokens"] for i in ids],iters=10000,seed=20261019,family_alpha=.05,family_comparisons=2)
    passed=all(x.get("status")=="SUPERIORITY_ESTABLISHED" and finite(x.get("exact_lower_bound")) and x["exact_lower_bound"]>0 for x in quality.values()) and all(x.get("status")=="OK" and finite(x.get("lower_bound")) and x["lower_bound"]>=.05 for x in cost.values())
    value={"cache_key":key,"data_hash":data_hash,"plan_hash":plan["plan_hash"],"scope":scope,"n":len(ids),"quality":quality,"cost":cost,"passed":passed,"status":"PASS" if passed else "INCONCLUSIVE"}
    value["result_hash"]=canonical(value)
    if cache:publish_once(cache,value)
    return {**value,"cache_reused":False}


def evaluate_positive_evidence(report,raw,evidence,plan_path,cache_dir=None):
    result={"status":"INCONCLUSIVE","passed":False,"p6_authorized":False,"requirements":{},"reason":"Prospective plan/evidence incomplete","plan_path":str(plan_path)}
    if not Path(plan_path).is_file():return {**result,"quality_plan_status":"PENDING_NOT_FROZEN"}
    try:
        plan=load(plan_path);verification=verify_plan(evidence.run,plan)
    except (Invalid,Pending,OSError,ValueError,KeyError) as exc:return {**result,"quality_plan_status":"UNVERIFIED","reason":str(exc)}
    result.update(quality_plan_status=verification["status"],plan_hash=plan["plan_hash"],analysis_supplement=plan["analysis_supplement"])
    requirements=result["requirements"];protocol=report["protocol"];calibrations=report["calibration"]["arms"];groups=report["generations"]
    formal=report["diagnostics"]["formal"];validation=report["diagnostics"]["formal/validation_state"]
    requirements["P0_protocol_pass"]=(report["preflight"].get("status") in ("PASS","PASSED") and report["preflight"].get("protocol_hash")==protocol.get("protocol_hash"))
    requirements["no_record_identity_or_format_errors"]=not evidence.errors
    requirements["runner_validation_freeze_matches_supplement"]=validation_contract(report,plan)
    requirements["all_eight_formal_selection_arms_complete"]=all(groups["formal/select"]["arms"].get(a,{}).get("complete") for a in METHODS)
    finalists=report["validation_freeze"].get("finalist_arms",[])
    requirements["frozen_validation_arms_complete"]=bool(finalists) and all(groups["formal/validate"]["arms"].get(a,{}).get("complete") for a in finalists)
    requirements["P3_numerical_state_source_gate"]=p3_numerical_gate(formal,protocol)
    requirements["updated_Cdiag_own_state_improvement"]=own_state_improves(formal)
    requirements["updated_validation_own_state_improvement"]=own_state_improves(validation)
    for label,diagnostic,paths,ids in (("Cdiag",formal,("A","B","C","D","M_NT","M_ST"),range(128,256)),("validation",validation,("M_NT","M_ST"),range(512,640))):
        summary=diagnostic.get("summary") or {};completeness=summary.get("prompt_completeness",{})
        requirements[label+"_declared_prompt_completeness"]=(summary.get("expected_prompt_ids")==sorted(str(i) for i in ids) and all(completeness.get(p,{}).get("complete_expected_ids") is True for p in paths))
        requirements[label+"_actual_full_trace_masks"]=all(diagnostic.get("full_trace_masks_reconciled",{}).get(p) for p in paths)
        requirements[label+"_bottom_archive_hashes"]=diagnostic.get("archive_hashes_verified") is True
        requirements[label+"_frozen_saved_state_bootstrap_contract"]=bootstrap_contract(summary)
    fit=calibration_provenance(evidence,raw,calibrations);result["calibration_provenance"]=fit;requirements["same_formula_mask_budget_source_provenance"]=fit["status"]=="VERIFIED"
    # Do not load NPZ or bootstrap while earlier numerical/generation evidence
    # is absent; no completed state-bootstrap computation is repeated here.
    if not all(requirements.values()):return result
    try:requirements["all_formal_generation_dataset_arm_identity"]=generation_identities(report,raw,calibrations)
    except (OSError,ValueError,KeyError,TypeError):requirements["all_formal_generation_dataset_arm_identity"]=False
    if not all(requirements.values()):return result
    for label,diagnostic,scope,paths in (("Cdiag",formal,"formal",("A","B","C","D","M_NT","M_ST")),("validation",validation,"formal/validation_state",("M_NT","M_ST"))):
        identity=diagnostic_identities(evidence,diagnostic,raw,calibrations,scope,paths)
        result[label+"_archive_identity"]=identity;requirements[label+"_bottom_driver_calibration_position_identity"]=identity["status"]=="VERIFIED"
    if not all(requirements.values()):return result
    controls=("M_NT",plan["selection"]["strongest_existing_baseline"])
    trigger=any((data.get("cap_rate") or 0)>.01 for data in groups["formal/select"]["arms"].values())
    result["cap4096_triggered"]=trigger
    if trigger:
        follow=groups["formal/select_cap4096"];complete=all(follow["arms"].get(a,{}).get("complete") for a in finalists)
        requirements["triggered_fixed128_cap4096_complete"]=complete
        if not complete:return result
        directions={}
        for control in controls:
            pair=paired_rows(raw,"formal/select_cap4096","M_ST",control,range(256,384))
            if not pair:return result
            left,right=pair
            reduction=1-sum(left[i]["generated_tokens"] for i in left)/sum(right[i]["generated_tokens"] for i in right)
            diffs={k:sum(left[i][k]-right[i][k] for i in left)/len(left) for k in ("strict_correct","tolerant_correct")}
            directions[control]={"relative_token_reduction":reduction,**diffs}
        result["cap4096_direction_sensitivity"]=directions
        requirements["cap4096_no_adverse_point_direction"]=all(all(v>=0 for v in row.values()) for row in directions.values())
        if not requirements["cap4096_no_adverse_point_direction"]:return result
    try:downstream=downstream_statistics(raw,"formal/validate",controls,list(range(512,1024)),plan,cache_dir)
    except (Invalid,OSError,ValueError,ImportError,KeyError) as exc:return {**result,"reason":"Downstream exact/ratio analysis unavailable: "+str(exc)}
    result["downstream"]=downstream;requirements["four_simultaneous_quality_lower_bounds_positive_and_two_ratio_lower_bounds_at_least_5pct"]=downstream.get("passed") is True
    if all(requirements.values()):result.update(status="POSITIVE_EXPLORATORY",passed=True,reason="Complete verified state/source/update/cost and explicit superiority chain under a prospective analysis supplement; no NI/independent-confirmation claim")
    return result
