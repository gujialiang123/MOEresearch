"""CPU fixtures for the isolated full512 supplement; no model or bootstrap."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/"scripts"))
sys.path.insert(0,str(REPO))
import state_study_validation512 as study


@pytest.fixture(autouse=True)
def forbid_cuda(monkeypatch):
    import torch
    for name in ("synchronize","empty_cache","is_available","device_count","get_device_name","current_device","init","manual_seed_all"):
        monkeypatch.setattr(torch.cuda,name,lambda *a,**k:pytest.fail("CPU verification attempted a CUDA call"))


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False)+"\n")


def cals():
    return {arm:{"calibration_hash":study.canonical(arm),"numerical_protection":study.EPS} for arm in ("M_N","L_N",*study.ARMS)}


def trace_row(sid=512,arm="M_NT",trace=None):
    return {"sample_id":sid,"arm":arm,"gen_token_ids":[3,9] if trace is None else trace}


def saved_check(row,calibrations):
    trace=row["gen_token_ids"]
    return study.replay_check(row["sample_id"],row["arm"],row,calibrations,"proto",{
        "argmaxes":trace,"decode_forwards":len(trace)-1,"mask_hash":study.canonical([[j,t] for j,t in enumerate(trace[:-1],1)]),
        "logits_finiteness_check":"Actual fake-driver finite returned logits"})


def write_actual_collector(directory,row,calibrations):
    import torch
    from moe_research.state_collection import StateCollector
    collector=StateCollector(Path(directory)/"diagnostics")
    sid,arm=row["sample_id"],row["arm"]
    identity=study.trace_identity(sid,arm,row,calibrations,"proto")
    tensor=torch.linspace(.5,1.5,2048,dtype=torch.float32).reshape(1,1,2048)
    for step,token in enumerate(row["gen_token_ids"][:-1],1):
        for layer in range(48):
            collector.observe(prompt_id=sid,path=arm,layer=layer,decode_forward_index=step,token_id=token,
                y8=tensor,u4=tensor*.6,z4=tensor*.95,residual=tensor*2,
                driver_hash=identity["driver_hash"],calibration_hash=identity["calibration_hash"])
    collector.finish_prompt(sid,arm,allow_empty=len(row["gen_token_ids"])==1,empty_metadata=identity)
    return collector


def test_import_and_help_are_cpu_only():
    code="import sys;sys.path.insert(0,'scripts');import state_study_validation512;assert 'torch' not in sys.modules;assert 'transformers' not in sys.modules"
    response=subprocess.run([sys.executable,"-c",code],cwd=REPO,capture_output=True,text=True)
    assert response.returncode==0,response.stderr


def test_frozen_collector_nonempty_roundtrip_across_bin_boundary(tmp_path):
    import numpy as np
    row=trace_row(trace=list(range(100,133))+[9]);calibrations=cals()
    write_actual_collector(tmp_path,row,calibrations)
    receipt=study.audit_archive(tmp_path,512,"M_NT",row,calibrations,"proto")
    assert receipt["layer_count"]==48 and receipt["token_rows"]==48*33
    assert receipt["decode_forwards"]==33 and receipt["kind"]=="archive"
    archive=next((tmp_path/"diagnostics").glob("*.npz"))
    with np.load(archive,allow_pickle=False) as arrays:
        records=json.loads(str(arrays["metadata_json"].item()))["records"]
        assert {r["key"][3] for r in records}=={"all","1-32","33-128"}
    # Independent report reader, separate from the supplement receipt logic.
    import state_study_validation512_report as report
    from summarize_state_study_results import Evidence,load_diagnostic
    ev=Evidence(tmp_path,True)
    diagnostic={"archive_hashes_verified":True,"current_inventory_matches":True,"summary":None}
    # Its checker expects the full universe; reduce only the expected fixture
    # universe, preserving the real writer/reader/layer/bin implementation.
    old=report.IDS
    try:
        report.IDS=(512,)
        row2=trace_row(512,"M_ST",[9]);write_actual_collector(tmp_path,row2,calibrations)
        for token_path in (tmp_path/"diagnostics").glob("*.tokens.jsonl"):ev.record(token_path)
        proof=report.verify_full512_archives(ev,diagnostic,{"M_NT":{512:row},"M_ST":{512:row2}},calibrations,"proto")
        assert proof["status"]=="VERIFIED",proof
    finally:report.IDS=old


def test_empty_marker_identity_and_conflict(tmp_path):
    row=trace_row(trace=[9]);calibrations=cals();write_actual_collector(tmp_path,row,calibrations)
    receipt=study.audit_archive(tmp_path,512,"M_NT",row,calibrations,"proto")
    assert receipt["kind"]=="empty" and receipt["layer_count"]==0
    assert receipt["decode_forwards"]==0 and receipt["token_rows"]==0
    path=tmp_path/receipt["files"][0]["relative_path"];marker=study.load(path)
    marker["metadata"]["trace_hash"]="wrong";put(path,marker)
    with pytest.raises(study.Invalid,match="Empty marker"):study.audit_archive(tmp_path,512,"M_NT",row,calibrations,"proto")


def test_committed_pair_bytes_and_driver_are_checked(tmp_path):
    row=trace_row();calibrations=cals();write_actual_collector(tmp_path,row,calibrations)
    study.audit_archive(tmp_path,512,"M_NT",row,calibrations,"proto")
    with pytest.raises(study.Invalid,match="trace/layer/bin"):
        study.audit_archive(tmp_path,512,"M_NT",row,{**calibrations,"M_NT":{"calibration_hash":"changed"}},"proto")
    tokens=next((tmp_path/"diagnostics").glob("*.tokens.jsonl"))
    with tokens.open("ab") as handle:handle.write(b"{}\n")
    with pytest.raises(study.Invalid,match="pair/commit"):
        study.audit_archive(tmp_path,512,"M_NT",row,calibrations,"proto")


def test_uncommitted_real_pair_is_retained_then_frozen_loader_resumes(tmp_path):
    from moe_research.state_collection import StateCollector
    row=trace_row();calibrations=cals();write_actual_collector(tmp_path,row,calibrations)
    data=tmp_path/"diagnostics";stem=study.canonical(["512","M_NT"])
    committed=data/(stem+".complete.json");committed.unlink()
    old_hashes={p.name:study.sha(p) for p in data.glob("*") if p.is_file()}
    with pytest.raises(study.Pending):study.audit_archive(tmp_path,512,"M_NT",row,calibrations,"proto")
    retained=study.isolate_uncommitted(data,stem)
    assert {Path(p).name:study.sha(p) for p in retained}==old_hashes
    assert not list(data.glob("*.npz"))
    write_actual_collector(tmp_path,row,calibrations)
    assert len(StateCollector.load(data).finished)==1
    assert study.audit_archive(tmp_path,512,"M_NT",row,calibrations,"proto")["token_rows"]==48


@pytest.mark.parametrize("mutation",["missing","duplicate","eos_prefix","cap_short","observer","armhash","prompt"])
def test_actual_full512_trace_validation_rejects_identity_errors(mutation):
    calibrations=cals();protocol={"protocol_hash":"proto","parser_version":"parser",
        "config":{"model_revision":"revision","dataset_revision":"datarev"}}
    from state_study_positive_evidence import arm_hash
    samples={i:{"question":str(i),"question_hash":hashlib.sha256(str(i).encode()).hexdigest()} for i in study.IDS}
    rows=[]
    for sid in study.IDS:
        rows.append({**trace_row(sid,trace=[3,9]),"arm_hash":arm_hash("M_NT",calibrations),"calibration_hash":calibrations["M_NT"]["calibration_hash"],
            "protocol_hash":"proto","parser_version":"parser","model_revision":"revision","dataset_revision":"datarev","scope":"formal/validate",
            "max_new_tokens":2048,"observer_enabled":False,"timing_debug_logits":False,**samples[sid],
            "eos_ids":[9],"generated_tokens":2,"decode_forward_count":1,"eos_position":1,"stop_reason":"eos",
            "prompt_token_ids":[1,2],"prompt_tokens":2,"rendered_prompt":"prompt","strict_correct":False,"tolerant_correct":False})
    assert len(study.validate_trace_rows(rows,"M_NT",protocol,calibrations,samples))==512
    if mutation=="missing":rows.pop()
    elif mutation=="duplicate":rows[-1]=rows[0]
    elif mutation=="eos_prefix":rows[0]["gen_token_ids"]=[9,9]
    elif mutation=="cap_short":rows[0].update(gen_token_ids=[3,4],eos_position=None,stop_reason="max_new_tokens")
    elif mutation=="observer":rows[0]["observer_enabled"]=True
    elif mutation=="armhash":rows[0]["arm_hash"]="wrong"
    elif mutation=="prompt":rows[0]["question_hash"]="wrong"
    with pytest.raises(study.Invalid):study.validate_trace_rows(rows,"M_NT",protocol,calibrations,samples)


def test_jsonl_uses_physical_lf_and_rejects_partial_duplicate_nonfinite(tmp_path):
    path=tmp_path/"data.jsonl";put(path,{"sample_id":512,"question":"A\u2028B"})
    assert study.read_jsonl(path)[0]["question"]=="A\u2028B"
    path.write_bytes(b'{"sample_id":512}')
    with pytest.raises(study.Invalid,match="Incomplete"):study.read_jsonl(path)
    path.write_bytes(b'{"sample_id":512}\n{"sample_id":512}\n')
    with pytest.raises(study.Invalid,match="Duplicate"):study.read_jsonl(path)
    path.write_bytes(b'{"sample_id":512,"x":NaN}\n')
    with pytest.raises(study.Invalid,match="Non-finite"):study.read_jsonl(path)


def test_replay_check_never_synthesizes_argmax_proof():
    row=trace_row();good=saved_check(row,cals())
    assert good["argmaxes"]==[3,9]
    for bad in ({**good,"argmaxes":[4,9]},{**good,"mask_hash":"wrong"},{**good,"path":"M_ST"},
                {**good,"logits_finiteness_check":""}):
        with pytest.raises(study.Invalid):study.replay_check(512,"M_NT",row,cals(),"proto",bad)


def test_missing_committed_replay_proof_requires_real_replay(tmp_path,monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda,"synchronize",lambda *a:pytest.fail("fake CPU path touched CUDA"))
    monkeypatch.setattr(torch.cuda,"empty_cache",lambda *a:pytest.fail("fake CPU path touched CUDA"))
    calibrations=cals();rows={a:{512:trace_row(512,a,[9])} for a in study.ARMS}
    for arm in study.ARMS:write_actual_collector(tmp_path,rows[arm][512],calibrations)
    # Both archive commits exist but actual replay proofs are absent.
    monkeypatch.setattr(study,"IDS",(512,))
    monkeypatch.setattr(study,"dataset_samples",lambda protocol:{512:{"sample_id":512}})
    from moe_research import state_calibration
    monkeypatch.setattr(state_calibration.FrozenStateCalibration,"from_dict",staticmethod(lambda value:object()))
    put(tmp_path/"run/protocol.json",{"config":{}});put(tmp_path/"run/manifest.json",{"runtime":{"eos_ids":[9]}})
    for arm in study.ARMS:rows[arm][512].update(rendered_prompt="saved",prompt_token_ids=[1])
    calls=[]
    class Driver:
        eos={9}
        def prompt(self,sample):return "saved",torch.tensor([[1]])
        def replay(self,sample,trace,arm,frozen,observer):
            calls.append((arm,observer));return saved_check(rows[arm][512],calibrations),[]
        def close(self):pass
    plan={"plan_hash":"plan","protocol_hash":"proto"};snapshot={"snapshot_hash":"snapshot"}
    result=study.collect(tmp_path/"run",plan,tmp_path,snapshot,rows,calibrations,driver_factory=lambda protocol:Driver())
    assert calls==[("M_NT",None),("M_ST",None)]
    assert len(result["archives"])==2
    # A second attempt validates both commits and proofs without loading driver.
    study.collect(tmp_path/"run",plan,tmp_path,snapshot,rows,calibrations,
                  driver_factory=lambda protocol:pytest.fail("completed paths must not load model"))


def test_analyze_fixed_full512_contract_and_resume_does_not_rerun(tmp_path,monkeypatch):
    plan={"plan_hash":"plan","protocol_hash":"proto"};snapshot={"snapshot_hash":"snapshot"}
    raw={"raw_completion_hash":"raw"};calls=[]
    monkeypatch.setattr(study,"source_snapshot",lambda run,plan:(snapshot,{},{}))
    monkeypatch.setattr(study,"verify_raw_completion",lambda *args:raw)
    put(tmp_path/"output/raw_completion.json",raw)
    put(tmp_path/"run/protocol.json",{"config":{"native_match_tolerance":.05,"state_effect_tolerance":.01}})
    def fake_analysis(source,out,**kwargs):
        calls.append(kwargs);put(Path(out)/"diagnostic_summary.json",{"status":"INCONCLUSIVE","reason":"undefined draws"})
    completion=study.analyze(tmp_path/"run",plan,tmp_path/"output",analysis_fn=fake_analysis)
    assert calls[0]["expected_prompt_ids"]==list(range(512,1024))
    assert calls[0]["iters"]==10000 and calls[0]["seed"]==20261019 and calls[0]["family"]=="lda"
    assert completion["completed_work"] is True
    assert study.analyze(tmp_path/"run",plan,tmp_path/"output",analysis_fn=lambda *a,**k:pytest.fail("repeat bootstrap"))==completion


def test_unregistered_gpu_run_rejected_before_model_import(tmp_path):
    with pytest.raises(study.Invalid,match="Unregistered"):
        study.registered_launch(tmp_path,{"plan_hash":"p"},reports=tmp_path/"reports",resume=object())


def test_plan_freeze_exclusive_source_and_late_validation(tmp_path,monkeypatch):
    repo=tmp_path/"repo";run=tmp_path/"run";run.mkdir();reports=tmp_path/"reports"
    for name in study.SUPPLEMENT_FILES:
        path=repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(name)
    protocol={"protocol_hash":"p"};source={"source_hash":"s"};draft={"plan_hash":"draft"}
    monkeypatch.setattr(study,"_draft_context",lambda *a,**k:(protocol,source,draft))
    plan=study.freeze_plan(run,repo=repo,reports=reports)
    assert study.load_plan(run,repo=repo,reports=reports)==plan
    archive=Path(plan["supplement_source_archive"]["directory"])
    assert all((archive/name).read_bytes()==(repo/name).read_bytes() for name in study.SUPPLEMENT_FILES)
    path=repo/study.SUPPLEMENT_FILES[0];path.write_text("changed")
    with pytest.raises(study.Invalid,match="source/protocol"):study.load_plan(run,repo=repo,reports=reports)
    fresh=tmp_path/"late";put(fresh/"formal/validate/generations/M_NT.jsonl",{"sample_id":512})
    with pytest.raises(study.quality.Invalid):study.freeze_plan(fresh,repo=repo,reports=reports)


def test_missing_plan_never_not_required(tmp_path):
    assert study.status(tmp_path,reports=tmp_path/"reports")["status"]=="PLAN_PENDING"


def test_explicit_scientific_error_stops():
    with pytest.raises(study.Invalid,match="scientific error"):
        study._error_guard("/does/not/exist",{"status":"FAILED_IMPLEMENTATION","stages":{}})


@pytest.mark.parametrize("mutation",[None,"boot","pid","argv","core_active","claim","resources"])
def test_registered_same_pid_claim_and_current_slurm_guards(tmp_path,mutation):
    from state_study_continuation import ContinuationRefused
    run=tmp_path/"run";reports=tmp_path/"reports";ops=reports/run.name/"validation512/ops"
    proc=tmp_path/"proc";pid=os.getpid();host=socket.gethostname();plan={"plan_hash":"plan","protocol_hash":"proto"}
    command=[sys.executable,"-u",str(study.REPO/"scripts/state_study_validation512.py"),"run","--run-dir",str(run.resolve())]
    put(run/"launch.json",{"slurm_job_id":"10"});put(run/"protocol.json",{"source_hash":"source"});put(run/"manifest.json",{"stages":{}})
    history=ops/"launch_history/old.json";put(history,{"old":"launch"});previous_hash=study.sha(history)
    claim_path=ops/"claims"/(previous_hash+".json")
    claim={"schema_version":"state_validation512_handoff_claim.v1","task":"validation512","run_dir":str(run.resolve()),
        "plan_hash":"plan","protocol_hash":"proto","owner_uid":os.getuid(),"pid":pid,"hostname":host,
        "boot_id":"boot","proc_start_ticks":7,"slurm_job_id":"11","previous_launch_sha256":previous_hash}
    put(claim_path,claim)
    launch={"schema_version":"state_validation512_launch.v1","task":"validation512","phase":"validation512",
        "run_dir":str(run.resolve()),"plan_hash":"plan","supplement_plan_hash":"plan","protocol_hash":"proto","source_hash":"source",
        "pid":pid,"hostname":host,"boot_id":"boot","proc_start_ticks":7,"command":command,
        "core_launch_sha256":study.sha(run/"launch.json"),"metadata_written_before_exec":True,"exec_preserves_pid":True,
        "cgroup":"/job_11\n","slurm_job_id":"11","previous_slurm_job_id":"10","previous_launch_sha256":previous_hash,
        "previous_launch_history":str(history),"shared_handoff_claim":{"path":str(claim_path),"sha256":study.sha(claim_path)},
        "core_terminal_proof":{"status":"VERIFIED_TERMINAL","job_id":"10"},"predecessor_terminal_proof":{"status":"VERIFIED_TERMINAL","job_id":"10"}}
    if mutation=="boot":launch["boot_id"]="other"
    if mutation=="pid":launch["pid"]=pid+1
    if mutation=="argv":command=command+["--plan","unregistered"]
    if mutation=="claim":put(claim_path,{**claim,"pid":pid+1});launch["shared_handoff_claim"]["sha256"]=study.sha(claim_path)
    put(ops/"launch.json",launch)
    for name,text in (("self/stat","fake"),("self/cgroup","/job_11\n"),("sys/kernel/random/boot_id","boot\n")):
        target=proc/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
    (proc/"self/cmdline").write_bytes(b"\0".join(x.encode() for x in command)+b"\0")
    called=[]
    def resources(allocation):
        called.append("resources")
        if mutation=="resources":raise ContinuationRefused("wrong resource allocation")
    fake=SimpleNamespace(_proc_stat=lambda text:{"start_ticks":7},
        predecessor_terminal=lambda job:{"status":"UNKNOWN" if mutation=="core_active" else "VERIFIED_TERMINAL"},
        verify_launch_process=lambda *a:{"status":"UNKNOWN"},completed=lambda *a:True,validate_sources=lambda *a:None,
        current_allocation=lambda *a,**k:{"job_id":"11","cgroup":"/job_11\n"},validate_allocation_resources=resources)
    if mutation is None:
        proof=study.registered_launch(run,plan,reports=reports,proc=proc,environment={"SLURM_JOB_ID":"11"},resume=fake)
        assert proof["pid"]==pid and called==["resources"]
    else:
        with pytest.raises((study.Invalid,ContinuationRefused)):
            study.registered_launch(run,plan,reports=reports,proc=proc,environment={"SLURM_JOB_ID":"11"},resume=fake)


def test_calibration_old_asserted_hash_cannot_hide_changed_parameters(tmp_path):
    payload={"family":"lda","parameters":{"0":{"mean_native":[1.0]}},"metadata":{"protocol_hash":"proto"}}
    payload={**payload,"calibration_hash":study.canonical(payload)}
    for arm in ("M_N","L_N",*study.ARMS):put(tmp_path/f"formal/calibration/{arm}.json",payload)
    changed=copy.deepcopy(payload);changed["parameters"]["0"]["mean_native"]=[99.0]
    put(tmp_path/"formal/calibration/M_NT.json",changed)
    with pytest.raises(study.Invalid,match="self hash/protocol/formula"):
        study._calibrations(tmp_path,{"protocol_hash":"proto"})


def test_delivery_missing_keeps_completed_state_raw_complete(tmp_path,monkeypatch):
    import state_study_validation512_report as report
    put(tmp_path/"completion.json",{})
    monkeypatch.setattr(study,"inspect_status",lambda *a,**k:{"status":"READY","output_dir":str(tmp_path)})
    monkeypatch.setattr(report,"check_validation512",lambda *a,**k:{"complete":True})
    monkeypatch.setattr(report,"verify_delivery",lambda *a,**k:{"verified":False,"delivery_complete":False},raising=False)
    assert study.status("/fake")["status"]=="RAW_COMPLETE"
    monkeypatch.setattr(report,"verify_delivery",lambda *a,**k:{"verified":True,"delivery_complete":True},raising=False)
    result=study.status("/fake")
    assert result["status"]=="COMPLETE" and result["delivery_complete"] is True


def test_supplement_scientific_error_precedes_ready_or_completed_state(tmp_path,monkeypatch):
    run=tmp_path/"run";reports=tmp_path/"reports";plan={"plan_hash":"plan"}
    monkeypatch.setattr(study,"load_plan",lambda *a,**k:plan)
    directory=study.output_dir(run,plan,reports=reports)
    put(directory/"manifest.json",{"plan_hash":"plan","status":"running","stages":{"failure":{"status":"FAILED_IMPLEMENTATION"}}})
    result=study.status(run,reports=reports)
    assert result["status"]=="SCIENTIFIC_ERROR"


def test_freeze_cannot_publish_in_original_run_or_symlink_escape(tmp_path):
    run=tmp_path/"run";run.mkdir();reports=tmp_path/"reports"
    with pytest.raises(study.Invalid,match="independent scratch"):
        study.freeze_plan(run,run/"unexpected-plan.json",reports=reports)
    allowed=reports/run.name/"validation512";allowed.mkdir(parents=True)
    (allowed/"escape").symlink_to(run,target_is_directory=True)
    with pytest.raises(study.Invalid,match="independent scratch"):
        study.freeze_plan(run,allowed/"escape/unexpected-plan.json",reports=reports)
    assert list(run.iterdir())==[]


@pytest.mark.parametrize("fstype,source",[("nfs4","server:/scratch"),("lustre","cluster:/data"),("beegfs","beegfs_nodev"),("overlay","overlay")])
def test_control_network_or_unknown_mount_refused_before_any_lock_file(tmp_path,fstype,source):
    def query(command,**kwargs):return SimpleNamespace(returncode=0,stdout=json.dumps({"filesystems":[{"target":"/scratch","fstype":fstype,"source":source}]}))
    control=tmp_path/"not-created"
    with pytest.raises(study.Invalid,match="local disk"):
        study.local_lock("run",{"plan_hash":"p"},control=control,fs_probe=lambda p:study.local_filesystem(p,query=query))
    assert not control.exists()


def test_positive_local_xfs_mount_proof(tmp_path):
    def query(command,**kwargs):return SimpleNamespace(returncode=0,stdout=json.dumps({"filesystems":[{"target":"/raid","fstype":"xfs","source":"/dev/md126"}]}))
    proof=study.local_filesystem(tmp_path,query=query)
    assert proof["verified_local_disk"] is True and proof["mount"]["source"]=="/dev/md126"
