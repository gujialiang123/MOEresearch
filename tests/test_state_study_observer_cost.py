"""CPU-only observer-cost contract tests; every actual torch.cuda call fails."""
from pathlib import Path
import copy
import json
import subprocess
import sys
from types import SimpleNamespace
import pytest

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'scripts'));sys.path.insert(0,str(REPO))
import state_study_observer_cost as cost


@pytest.fixture(autouse=True)
def forbid_cuda(monkeypatch):
    import torch
    for name in ('synchronize','empty_cache','is_available','device_count','get_device_name','current_device','init','manual_seed_all','reset_peak_memory_stats','memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved'):
        monkeypatch.setattr(torch.cuda,name,lambda *a,**k:pytest.fail('CPU observer-cost test touched CUDA'))


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def cals():return {'L_N':{'calibration_hash':'cal','numerical_protection':cost.base.EPS}}


def unit(scenario='C',trace=None):
    trace=[3,4,9] if trace is None else trace
    scenario_index=cost.SCENARIOS.index(scenario);sid=128
    body={'sample_id':sid,'scenario':scenario,'scenario_index':scenario_index,
        'driver_arm':'N' if scenario in ('A','D') else 'L_N','observed_calibration_arm':'L_N',
        'trace':trace,'source_argmax_required':scenario in ('A','C','M_NT','M_ST'),
        'order':['on','off'] if (sid+scenario_index)%2==0 else ['off','on'],
        'common_decode_limit':len(trace)-1,'sample':{'sample_id':sid}}
    return cost.sealed(body,'unit_hash')


def result(u,argmax=None):
    return {'argmaxes':u['trace'] if argmax is None else argmax,'decode_forwards':len(u['trace'])-1,
        'mask_hash':cost.canonical([[j,t] for j,t in enumerate(u['trace'][:-1],1)]),
        'logits_finiteness_check':'all returned logits before every argmax; included common driver overhead'}


class FakeCUDA:
    def __init__(self):self.calls=[]
    def empty_cache(self):self.calls.append('empty_cache')
    def synchronize(self):self.calls.append('synchronize')
    def reset_peak_memory_stats(self):self.calls.append('reset_peak')
    def memory_allocated(self):self.calls.append('baseline_allocated');return 100
    def memory_reserved(self):self.calls.append('baseline_reserved');return 200
    def max_memory_allocated(self):self.calls.append('peak_allocated');return 130
    def max_memory_reserved(self):self.calls.append('peak_reserved');return 250


def test_import_cpu_only():
    response=subprocess.run([sys.executable,'-c',"import sys;sys.path.insert(0,'scripts');import state_study_observer_cost;assert 'torch' not in sys.modules;assert 'transformers' not in sys.modules"],cwd=REPO,capture_output=True,text=True)
    assert response.returncode==0,response.stderr


@pytest.mark.parametrize('scenario',['A','C','M_NT','M_ST'])
def test_own_source_argmax_is_required(scenario):
    u=unit(scenario)
    with pytest.raises(cost.Invalid,match='argmax'):cost.replay_proof(u,result(u,[8,8,9]))


@pytest.mark.parametrize('scenario',['B','D'])
def test_cross_text_argmax_need_not_equal_source(scenario):
    u=unit(scenario)
    assert cost.replay_proof(u,result(u,[8,8,9]))['argmaxes']==[8,8,9]


def test_empty_generation_trace_is_invalid_instead_of_imaginary_state():
    u=unit(trace=[])
    with pytest.raises(cost.Invalid,match='nonempty'):cost.replay_proof(u,result(u))


def test_signed_negative_differences_are_retained():
    def mode(t,n):return {'timing':{'replay_seconds':t},'memory':{k:n for k in ('baseline_allocated_bytes','baseline_reserved_bytes','peak_allocated_bytes','peak_reserved_bytes','incremental_peak_allocated_bytes','incremental_peak_reserved_bytes')}}
    d=cost.signed_differences(mode(1,10),mode(2,20))
    assert d['replay_seconds']==-1 and d['peak_reserved_bytes']==-10


def write_real_on_mode(attempt,u,plan):
    import torch
    from moe_research.state_collection import StateCollector
    tensor=torch.linspace(.5,1.5,2048).reshape(1,1,2048)
    driver=SimpleNamespace(residuals={l:tensor*2 for l in range(48)})
    def replay(sample,trace,arm,calibrations,observer,capture_logits=False):
        assert capture_logits is False
        if observer:
            for step,_ in enumerate(trace[:-1],1):
                for layer in range(48):observer(None,tensor,tensor*.95,'decode',layer,step,None)
        return result(u),[]
    driver.replay=replay
    fake_cal=SimpleNamespace(apply=lambda layer,x:x*.95)
    def observer_factory(*args,**kwargs):
        return cost.make_observer(*args,**kwargs,torch_module=torch,
            shadow_fn=lambda block,h:{'y8':tensor,'u4':tensor*.7,'y4_full':tensor})
    cuda=FakeCUDA()
    on=cost.measured_mode(driver,u,'on',{'L_N':fake_cal},cals(),plan,attempt,cuda=cuda,
        collector_factory=StateCollector,observer_factory=observer_factory)
    return on,driver,cuda


def test_real_frozen_collector_full48_bins_and_real_observer_roundtrip(tmp_path):
    u=unit(trace=list(range(100,133))+[9]);plan={'plan_hash':'plan','protocol_hash':'proto'}
    on,driver,cuda=write_real_on_mode(tmp_path,u,plan)
    assert on['observer_calls']==48*33
    assert on['archive_receipt']['decode_forwards']==33 and on['archive_receipt']['layer_count']==48
    assert on['memory']['incremental_peak_reserved_bytes']==50
    assert cuda.calls[:5]==['empty_cache','synchronize','reset_peak','baseline_allocated','baseline_reserved']
    assert cuda.calls[5]=='synchronize'
    metadata_file=next((tmp_path/'diagnostics').glob('*.npz'))
    import numpy as np
    with np.load(metadata_file,allow_pickle=False) as arrays:
        meta=json.loads(str(arrays['metadata_json'].item()))
    assert {r['key'][3] for r in meta['records']}=={'all','1-32','33-128','common','common:1-32','common:33-128'}
    receipt=cost.audit_observer_archive(tmp_path,u,cals(),'proto')
    assert receipt==on['archive_receipt']
    # Independently implemented report reader accepts the actual written mode.
    import state_study_observer_cost_report as report
    off=cost.measured_mode(driver,u,'off',{},cals(),plan,tmp_path,cuda=cuda,
        collector_factory=lambda *a:pytest.fail('off constructed collector'))
    assert report.validate_modes(u,{'signed_differences':cost.signed_differences(on,off)},
        {'on':on,'off':off},'plan')==cost.signed_differences(on,off)


def test_off_mode_excludes_observer_and_archive_and_times_full_trace(tmp_path):
    u=unit('B');calls=[];cuda=FakeCUDA()
    class Driver:
        def replay(self,sample,trace,arm,calibrations,observer,capture_logits=False):
            calls.append((trace,observer,capture_logits));return result(u,[8,8,9]),[]
    off=cost.measured_mode(Driver(),u,'off',{},cals(),{'plan_hash':'plan','protocol_hash':'proto'},tmp_path,
        cuda=cuda,collector_factory=lambda *a:pytest.fail('off constructed collector'))
    assert calls==[(u['trace'],None,False)]
    assert off['archive_receipt'] is None and off['timing']['observer_callback_host_seconds']==0 and off['timing']['archive_seconds']==0


def complete_pair(tmp_path,u):
    plan={'plan_hash':'plan','protocol_hash':'proto'};attempt=tmp_path/'attempts'/u['unit_hash']/'attempt'
    on,driver,cuda=write_real_on_mode(attempt,u,plan)
    off=cost.measured_mode(driver,u,'off',{},cals(),plan,attempt,cuda=cuda,collector_factory=lambda *a:pytest.fail('off collector'))
    intent={'plan_hash':'plan','source_snapshot_hash':'snapshot','unit_hash':u['unit_hash'],'order':u['order']}
    put(attempt/'intent.json',intent)
    warm={**u,'trace':u['trace'][:8]}
    put(attempt/'warmup.json',{'excluded_from_measurement':True,'trace_length':len(warm['trace']),'elapsed_seconds':.1,
        'observer_calls':48*(len(warm['trace'])-1),'replay':result(warm)})
    body={'schema_version':'state_observer_cost.pair.v1','plan_hash':'plan','source_snapshot_hash':'snapshot',
        'unit_hash':u['unit_hash'],'order':u['order'],'attempt_relative_dir':str(attempt.relative_to(tmp_path)),
        'modes':{m:{'relative_path':str((attempt/(m+'.mode.json')).relative_to(tmp_path)),'sha256':cost.sha(attempt/(m+'.mode.json'))} for m in ('on','off')},
        'signed_differences':cost.signed_differences(on,off)}
    pair=cost.sealed(body,'pair_hash');put(tmp_path/'pairs'/(u['unit_hash']+'.json'),pair)
    return attempt,pair


def test_pair_commit_binds_real_archives_warmup_intent_and_modes(tmp_path):
    u=unit();attempt,pair=complete_pair(tmp_path,u)
    receipt=cost.audit_pair(tmp_path,u,cals(),'proto')
    names={Path(f['relative_path']).name for f in receipt['files']}
    assert {'warmup.json','intent.json','on.mode.json','off.mode.json'}.issubset(names)
    assert receipt['pair_hash']==pair['pair_hash']
    warm=cost.load(attempt/'warmup.json');warm['observer_calls']=0;put(attempt/'warmup.json',warm)
    with pytest.raises(cost.Invalid,match='warmup'):cost.audit_pair(tmp_path,u,cals(),'proto')


def test_half_pair_not_complete_and_bytes_preserved(tmp_path):
    u=unit();attempt=tmp_path/'attempts'/u['unit_hash']/'half';plan={'plan_hash':'plan','protocol_hash':'proto'}
    write_real_on_mode(attempt,u,plan);old=cost.sha(attempt/'on.mode.json')
    with pytest.raises(cost.Pending):cost.audit_pair(tmp_path,u,cals(),'proto')
    assert cost.sha(attempt/'on.mode.json')==old and not (tmp_path/'pairs').exists()


def test_pair_detects_output_neutrality_failure_even_cross_text(tmp_path):
    u=unit('B');attempt,pair=complete_pair(tmp_path,u)
    mode=cost.load(attempt/'off.mode.json');mode['replay']['argmaxes']=[8,8,9]
    mode=cost.sealed({k:v for k,v in mode.items() if k!='mode_hash'},'mode_hash');put(attempt/'off.mode.json',mode)
    pair['modes']['off']['sha256']=cost.sha(attempt/'off.mode.json')
    pair=cost.sealed({k:v for k,v in pair.items() if k!='pair_hash'},'pair_hash');put(tmp_path/'pairs'/(u['unit_hash']+'.json'),pair)
    with pytest.raises(cost.Invalid,match='changed actual driver'):cost.audit_pair(tmp_path,u,cals(),'proto')


def test_empty_immediate_eos_actual_collector(tmp_path):
    u=unit(trace=[9]);on,_,_=write_real_on_mode(tmp_path,u,{'plan_hash':'plan','protocol_hash':'proto'})
    assert on['observer_calls']==0 and on['archive_receipt']['layer_count']==0
    assert on['archive_receipt']['decode_forwards']==0


def test_original_run_target_refused_before_any_publication(tmp_path):
    run=tmp_path/'run';run.mkdir()
    with pytest.raises(cost.Invalid,match='independent scratch'):cost.freeze_plan(run,run/'plan.json',reports=tmp_path/'reports')
    assert not list(run.iterdir())


def test_sealed_cost_source_and_full512_binding(tmp_path,monkeypatch):
    repo=tmp_path/'repo';run=tmp_path/'run';run.mkdir();reports=tmp_path/'reports'
    for name in cost.SOURCE_FILES:
        p=repo/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(name)
    full={'protocol_hash':'proto','source':{'frozen_source_hash':'source'}};reference={'path':'full','sha256':'sha','plan_hash':'fullplan'}
    monkeypatch.setattr(cost,'_context',lambda *a,**k:(full,reference))
    plan=cost.freeze_plan(run,repo=repo,reports=reports)
    assert cost.load_plan(run,repo=repo,reports=reports)==plan
    changed={**reference,'plan_hash':'other'};monkeypatch.setattr(cost,'_context',lambda *a,**k:(full,changed))
    with pytest.raises(cost.Invalid,match='binding'):cost.load_plan(run,repo=repo,reports=reports)


def test_prerequisites_do_not_skip_missing_validation_delivery(monkeypatch):
    monkeypatch.setattr(cost.base,'inspect_status',lambda *a:{'status':'NOT_REQUIRED','core_complete':True})
    monkeypatch.setattr(cost.base,'status',lambda *a:{'status':'NOT_REQUIRED','delivery_complete':False})
    with pytest.raises(cost.Pending,match='WAITING_VALIDATION512'):cost.prerequisites('/run')
    monkeypatch.setattr(cost.base,'status',lambda *a:{'status':'NOT_REQUIRED','delivery_complete':True})
    assert cost.prerequisites('/run')[1]['delivery_complete'] is True


def test_stage_scientific_error_stops_before_ready(tmp_path,monkeypatch):
    plan={'plan_hash':'plan'};monkeypatch.setattr(cost,'load_plan',lambda *a,**k:plan)
    put(tmp_path/'run/manifest.json',{'stages':{}})
    out=cost.output_dir(tmp_path/'run',plan,reports=tmp_path/'reports')
    put(out/'manifest.json',{'plan_hash':'plan','status':'running','stages':{'failure':{'status':'FAILED_IMPLEMENTATION'}}})
    assert cost.status(tmp_path/'run',reports=tmp_path/'reports')['status']=='SCIENTIFIC_ERROR'


def test_main_refuses_explicit_scientific_error_without_registering_or_loading(tmp_path,monkeypatch):
    monkeypatch.setattr(cost,'load_plan',lambda *a,**k:{'plan_hash':'plan'})
    monkeypatch.setattr(cost,'status',lambda *a,**k:{'status':'SCIENTIFIC_ERROR'})
    assert cost.main(['run','--run-dir',str(tmp_path)])==3


def test_analyze_and_delivery_resume_do_not_replay_or_bootstrap(tmp_path,monkeypatch):
    u=unit();attempt,pair=complete_pair(tmp_path,u);plan={'plan_hash':'plan','protocol_hash':'proto'}
    snapshot={'snapshot_hash':'snapshot','scenarios':['C']};put(tmp_path/'source_snapshot.json',snapshot)
    raw=cost.raw_receipt(tmp_path,plan,snapshot,[u],cals());put(tmp_path/'raw_completion.json',raw)
    monkeypatch.setattr(cost,'source_snapshot',lambda *a:(snapshot,[u],cals()))
    done=cost.analyze('/unused',plan,tmp_path)
    summary=cost.load(tmp_path/'summaries/observer_cost_summary.json')
    assert summary['bootstrap_performed'] is False and summary['pair_count']==1
    assert done['completed_work'] is True
    assert cost.analyze('/unused',plan,tmp_path)==done


@pytest.mark.parametrize('triggered',[False,True])
def test_source_snapshot_actual_raw_readers_freeze_exact_units_and_balanced_order(tmp_path,monkeypatch,triggered):
    from state_study_positive_evidence import arm_hash
    run=tmp_path/'run';fullout=tmp_path/'full';put(fullout/'completion.json',{'work':'done'})
    put(run/'formal/summaries/diagnostic_summary.json',{'gate_recalibration':triggered})
    protocol={'config':{'model_revision':'model','dataset_revision':'data'},'parser_version':'parser'}
    put(run/'protocol.json',protocol);put(run/'manifest.json',{'runtime':{'eos_ids':[9]}})
    metadata={'protocol_hash':'proto','fit_mask_hash':'fit','prompt_ids':list(range(128)),
        'fit_decode_token_count':1,'source_student_calibration_hash':'source'}
    calibrations={}
    for arm in ('M_N','L_N','M_NT','M_ST'):
        body={'family':'scalar' if arm=='M_N' else 'lda','metadata':metadata}
        value={**body,'calibration_hash':cost.canonical(body)}
        put(run/f'formal/calibration/{arm}.json',value);calibrations[arm]=value
    samples={sid:{'question':str(sid),'question_hash':'q'+str(sid)} for sid in range(128,256)}
    arms=('N','L_N','M_NT','M_ST') if triggered else ('N','L_N')
    for arm in arms:
        scope='formal/diag_rollouts' if arm in ('N','L_N') else 'formal/updated_diag_rollouts'
        path=run/scope/'generations'/(arm+'.jsonl');path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w') as handle:
            for sid in range(128,256):
                row={'sample_id':sid,'arm':arm,'arm_hash':arm_hash(arm,calibrations),'scope':scope,'protocol_hash':'proto',
                    'calibration_hash':calibrations[arm]['calibration_hash'] if arm in calibrations else None,
                    'model_revision':'model','dataset_revision':'data','parser_version':'parser','max_new_tokens':2048,
                    'observer_enabled':False,'timing_debug_logits':False,'gen_token_ids':[3,9],'eos_ids':[9],
                    'generated_tokens':2,'decode_forward_count':1,'eos_position':1,'stop_reason':'eos',
                    'rendered_prompt':'prompt','prompt_token_ids':[1],'prompt_tokens':1,**samples[sid]}
                handle.write(json.dumps(row)+'\n')
    monkeypatch.setattr(cost,'prerequisites',lambda *a:({'status':'READY'},{'status':'COMPLETE' if triggered else 'NOT_REQUIRED',
        'delivery_complete':True,'delivery_proof':{'delivery_hash':'mutable-report'}}))
    monkeypatch.setattr(cost.base,'load_plan',lambda *a:{'plan_hash':'full'})
    monkeypatch.setattr(cost.base,'output_dir',lambda *a:fullout)
    monkeypatch.setattr(cost.base,'_p4_trigger',lambda *a:triggered)
    monkeypatch.setattr(cost.base,'dataset_samples',lambda *a:samples)
    plan={'plan_hash':'cost','protocol_hash':'proto','validation512_plan':{'plan_hash':'full'}}
    snapshot,units,cals_payload=cost.source_snapshot(run,plan)
    assert cost.SPEC['sample_ids']==[128,129,130,131] and cost.SPEC['measured_pairs_per_unit']==1
    assert len(units)==(24 if triggered else 16)
    assert snapshot['scenarios']==list(cost.SCENARIOS if triggered else cost.SCENARIOS[:4])
    for scenario in snapshot['scenarios']:
        selected=[u for u in units if u['scenario']==scenario]
        assert sum(u['order'][0]=='on' for u in selected)==2
        assert sum(u['order'][0]=='off' for u in selected)==2
    for u in units:
        assert u['sample_id'] in cost.IDS
        assert (u['order'][0]=='on')==((u['sample_id']+u['scenario_index'])%2==0)
        assert u['source_argmax_required']==(u['scenario'] in ('A','C','M_NT','M_ST'))
        assert u['unit_hash']==cost.canonical({k:v for k,v in u.items() if k!='unit_hash'})
        assert (u['common_decode_limit'] is None)==(u['scenario'] in ('M_NT','M_ST'))
