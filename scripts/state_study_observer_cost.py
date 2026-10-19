#!/usr/bin/env python3
"""Separate fixed-subset observer resource benchmark; original run is read-only.

No model/GPU import occurs in freeze/status/check. The actual registered run
replays complete saved traces; it never fits or generates a new trace.
"""
from __future__ import annotations
import argparse
import gc
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import uuid
import zipfile

REPO=Path(__file__).resolve().parents[1]
for _path in (str(REPO),str(REPO/'scripts')):
    if _path not in sys.path:sys.path.insert(0,_path)
import state_study_validation512 as base
quality=base.quality
Pending=base.Pending;Invalid=base.Invalid
load=base.load;sha=base.sha;canonical=base.canonical;sealed=base.sealed;verify_seal=base.verify_seal
publish_equal=base.publish_equal;utc=base.utc
REPORTS=base.REPORTS
CONTROL=Path('/raid/jgu7/state-study-observer-cost')
IDS=(128,129,130,131)
SCENARIOS=('A','B','C','D','M_NT','M_ST')
SOURCE_FILES=('scripts/state_study_observer_cost.py','scripts/state_study_observer_cost_report.py',
              'docs/2026-10-19/state_observer_cost.md','docs/2026-10-19/state_observer_cost_reporting.md')
SPEC={'sample_ids':list(IDS),'base_scenarios':list(SCENARIOS[:4]),'updated_scenarios_if_P4_triggered':list(SCENARIOS[4:]),
      'measured_pairs_per_unit':1,'full_saved_trace':True,'decode_positions':'trace[:-1]; T-1; emitted EOS never consumed',
      'layer_ids':list(range(48)),'hidden_size':2048,'position_bins':[list(x) for x in base.BINS],
      'warmup':'one real observer-on replay of trace[:min(8,T)] per scenario/prompt before either measured mode; excluded',
      'order':'on first iff (sample_id+scenario_index)%2==0; otherwise off first',
      'mode_preparation':['gc.collect','CUDA empty_cache','CUDA synchronize','reset_peak_memory_stats','read allocated/reserved baseline'],
      'timing':'perf_counter around complete replay plus final CUDA synchronize; capture_logits=False; shared finite-logit guard unchanged',
      'observer_callback_time':'host callback wall time including its implicit synchronizations; no inserted per-callback synchronization',
      'memory_unit':'bytes; absolute baseline/peak and peak-minus-baseline; signed on-minus-off differences retained',
      'archive_io_outside_online_measurement':True,'pair_commit_only_after_both_modes':True,
      'bootstrap':False,'no_fitting_or_free_generation':True,'no_subset_extrapolation_or_serving_speedup':True,
      'requires_core_and_validation512_delivery':True}


def plan_path(run,*,reports=REPORTS):return Path(reports)/Path(run).name/'observer_cost/plan.json'
def output_dir(run,plan,*,reports=REPORTS):return Path(reports)/Path(run).name/'observer_cost'/plan['plan_hash']


def _sources(repo=REPO):
    files={}
    for name in SOURCE_FILES:
        path=Path(repo)/name
        if not path.is_file():raise Pending('Observer-cost scientific source is not ready: '+name)
        files[name]=sha(path)
    digest=canonical(files)
    return {'analysis_source_hashes':files,'analysis_source_hash':digest,'analysis_archive_relative_dir':'source_archives/'+digest}


def _context(run,*,repo=REPO,reports=REPORTS):
    full=base.load_plan(run,repo=repo,reports=reports)
    path=base.plan_path(run,reports=reports)
    return full,{'path':str(path.resolve()),'sha256':sha(path),'plan_hash':full['plan_hash']}


def freeze_plan(run,target=None,*,repo=REPO,reports=REPORTS):
    run=Path(run).resolve();target=Path(target).resolve() if target else plan_path(run,reports=reports).resolve()
    allowed=(Path(reports)/run.name/'observer_cost').resolve()
    if not target.is_relative_to(allowed) or target==allowed or target.is_relative_to(run) or target.is_relative_to(Path.home().resolve()):
        raise Invalid('Observer-cost plan must stay in independent scratch reports, outside original run/Home')
    if target.exists():return load_plan(run,target,repo=repo,reports=reports)
    full,reference=_context(run,repo=repo,reports=reports);source=_sources(repo)
    archive=quality.publish_analysis_sources(target.parent,source,repo=repo)
    value=sealed({'schema_version':'state_observer_cost.plan.v1','status':'FROZEN_ENGINEERING_RESOURCE_SUBSET',
        'created_at_utc':utc(),'run_dir':str(run),'protocol_hash':full['protocol_hash'],'source':full['source'],
        'validation512_plan':reference,'spec':SPEC,'observer_cost_source':source,'observer_cost_source_archive':archive,
        'scope':'Original section8.4 observer resource accounting; fixed subset marginal paired engineering benchmark only',
        'P6_authorized':False},'plan_hash')
    quality.publish_once(target,value);return value


def load_plan(run,plan=None,*,repo=REPO,reports=REPORTS):
    run=Path(run).resolve();path=Path(plan) if plan else plan_path(run,reports=reports)
    if not path.is_file():raise Pending('Observer-cost engineering plan is not sealed')
    value=load(path);verify_seal(value,'plan_hash');full,reference=_context(run,repo=repo,reports=reports)
    expected={'schema_version':'state_observer_cost.plan.v1','status':'FROZEN_ENGINEERING_RESOURCE_SUBSET',
        'run_dir':str(run),'protocol_hash':full['protocol_hash'],'source':full['source'],'validation512_plan':reference,'spec':SPEC,
        'observer_cost_source':_sources(repo)}
    if any(value.get(k)!=v for k,v in expected.items()):raise Invalid('Observer-cost plan scope/source/full512 binding differs')
    quality.verify_source_archive({'source':value['observer_cost_source'],'analysis_source_archive':value['observer_cost_source_archive']})
    return value


def prerequisites(run,plan=None):
    state=base.inspect_status(run)
    if state['status']=='SCIENTIFIC_ERROR':raise Invalid('Current scientific error: '+state.get('reason',''))
    if state['status'] in ('PLAN_PENDING','INVALID_PLAN'):raise Invalid('Full512 scope plan is not valid: '+state['status'])
    if not state.get('core_complete'):raise Pending('WAITING_CORE')
    delivery=base.status(run)
    if not (delivery.get('status') in ('COMPLETE','NOT_REQUIRED') and delivery.get('delivery_complete') is True):
        raise Pending('WAITING_VALIDATION512')
    return state,delivery


def status(run,plan=None,*,repo=REPO,reports=REPORTS):
    try:
        value=load_plan(run,plan,repo=repo,reports=reports);directory=output_dir(run,value,reports=reports)
        base._error_guard(run,load(Path(run)/'manifest.json'))
        if (directory/'manifest.json').is_file():
            own=load(directory/'manifest.json')
            if own.get('plan_hash')!=value['plan_hash']:raise Invalid('Observer-cost manifest plan differs')
            base._error_guard(run,own)
        try:state,delivery=prerequisites(run,value)
        except Pending as exc:return {'status':str(exc),'plan_hash':value['plan_hash'],'output_dir':str(directory),'delivery_complete':False}
        result={'status':'READY','plan_hash':value['plan_hash'],'output_dir':str(directory),'delivery_complete':False,
                'validation512_delivery':delivery.get('delivery_proof')}
        if (directory/'completion.json').is_file():
            report=importlib.import_module('state_study_observer_cost_report')
            proof=report.check_observer_cost(run,plan_path=plan)
            if proof.get('complete') is not True:return {**result,'status':'INVALID_PLAN','work_proof':proof}
            done=report.verify_delivery(run,plan_path=plan,work_proof=proof)
            verified=done.get('verified') is True and done.get('delivery_complete') is True
            return {**result,'status':'COMPLETE' if verified else 'RAW_COMPLETE','delivery_complete':verified,'work_proof':proof,'delivery_proof':done}
        if (directory/'raw_completion.json').is_file():
            snapshot,units,cals=source_snapshot(run,value);verify_raw_completion(directory,value,snapshot,units,cals)
            return {**result,'status':'RAW_COMPLETE'}
        return result
    except Pending as exc:return {'status':'PLAN_PENDING','reason':str(exc),'delivery_complete':False}
    except Exception as exc:return {'status':'SCIENTIFIC_ERROR' if 'scientific error' in str(exc).lower() else 'INVALID_PLAN','reason':str(exc),'delivery_complete':False}


def source_snapshot(run,plan):
    """Deterministic actual source proof, fixed units list, calibration payloads."""
    run=Path(run).resolve();state,delivery=prerequisites(run,plan)
    protocol=load(run/'protocol.json');triggered=base._p4_trigger(run,base.load_plan(run))
    if triggered is None:raise Pending('P4 decision is pending')
    scenarios=SCENARIOS if triggered else SCENARIOS[:4]
    from summarize_state_study_results import Evidence,load_calibrations
    from state_study_positive_evidence import arm_hash
    ev=Evidence(run,True);descriptors=load_calibrations(ev,plan['protocol_hash'])['arms']
    needed={'L_N',*(SCENARIOS[4:] if triggered else ())}
    if ev.errors or not needed.issubset(descriptors):raise Invalid('Observer-cost calibration payload hash/protocol differs: '+str(ev.errors))
    cals={a:load(run/f'formal/calibration/{a}.json') for a in descriptors}
    samples=base.dataset_samples(protocol);rows={};trace_sources={};units=[]
    runtime_eos=load(run/'manifest.json').get('runtime',{}).get('eos_ids')
    for scope,arms in (('formal/diag_rollouts',('N','L_N')),('formal/updated_diag_rollouts',SCENARIOS[4:] if triggered else ())):
        for arm in arms:
            path=run/scope/'generations'/(arm+'.jsonl');rr=base.read_jsonl(path)
            mapped={r['sample_id']:r for r in rr}
            if set(mapped)!=set(range(128,256)):raise Invalid('Formal Cdiag source is not the original128 prompts')
            for sid in IDS:
                row=mapped[sid];trace=row.get('gen_token_ids',[]);T=len(trace)
                expected={'protocol_hash':plan['protocol_hash'],'scope':scope,'arm':arm,'arm_hash':arm_hash(arm,cals),
                    'calibration_hash':cals[arm]['calibration_hash'] if arm in cals else None,'model_revision':protocol['config']['model_revision'],
                    'dataset_revision':protocol['config']['dataset_revision'],'parser_version':protocol['parser_version'],
                    'max_new_tokens':2048,'observer_enabled':False,'timing_debug_logits':False}
                if (any(row.get(k)!=v for k,v in expected.items()) or not 1<=T<=2048
                    or any(type(t) is not int or t<0 for t in trace)):raise Invalid('Saved cost trace source identity differs')
                eos=row.get('eos_ids',[]);hit=trace[-1] in eos
                if (not eos or eos!=runtime_eos or any(t in eos for t in trace[:-1]) or row.get('generated_tokens')!=T or row.get('decode_forward_count')!=T-1
                    or row.get('eos_position')!=(T-1 if hit else None) or row.get('stop_reason')!=('eos' if hit else 'max_new_tokens')
                    or (not hit and T!=2048) or row.get('question_hash')!=samples[sid]['question_hash'] or row.get('question')!=samples[sid]['question']):
                    raise Invalid('Saved cost trace T-1/EOS/prompt differs')
            rows[arm]=mapped;trace_sources[arm]={'relative_path':str(path.relative_to(run)),'sha256':sha(path)}
    for sid in IDS:
        if any(rows['N'][sid].get(k)!=rows['L_N'][sid].get(k) for k in ('question_hash','rendered_prompt','prompt_token_ids','prompt_tokens','eos_ids')):
            raise Invalid('Native/student source prompt identity differs')
    for index,scenario in enumerate(scenarios):
        driver={'A':'N','B':'L_N','C':'L_N','D':'N'}.get(scenario,scenario)
        text={'A':'N','B':'N','C':'L_N','D':'L_N'}.get(scenario,scenario)
        observed='L_N' if scenario in SCENARIOS[:4] else scenario
        for sid in IDS:
            row=rows[text][sid]
            common=min(len(rows['N'][sid]['gen_token_ids']),len(rows['L_N'][sid]['gen_token_ids']))-1 if scenario in SCENARIOS[:4] else None
            body={'sample_id':sid,'scenario':scenario,'scenario_index':index,'driver_arm':driver,'observed_calibration_arm':observed,
                'trace_source_scope':'formal/diag_rollouts' if scenario in SCENARIOS[:4] else 'formal/updated_diag_rollouts',
                'trace_source_arm':text,'trace':row['gen_token_ids'],'generation_row':row,'common_decode_limit':common,
                'source_argmax_required':scenario in ('A','C','M_NT','M_ST'),'order':['on','off'] if (sid+index)%2==0 else ['off','on']}
            units.append(sealed(body,'unit_hash'))
    snapshot=sealed({'schema_version':'state_observer_cost.sources.v1','plan_hash':plan['plan_hash'],'protocol_hash':plan['protocol_hash'],
        'validation512_plan_hash':plan['validation512_plan']['plan_hash'],'P4_triggered':triggered,
        'scenarios':list(scenarios),'sample_ids':list(IDS),'unit_hashes':[u['unit_hash'] for u in units],
        'trace_sources':trace_sources,'calibrations':{a:{'calibration_hash':v['calibration_hash'],'sha256':sha(run/f'formal/calibration/{a}.json')} for a,v in cals.items()},
        'validation512_work_identity':sha(base.output_dir(run,base.load_plan(run))/'completion.json')
            if delivery['status']=='COMPLETE' else sha(run/'formal/summaries/diagnostic_summary.json')},'snapshot_hash')
    return snapshot,units,cals


def identity(unit,cals,protocol_hash):
    from state_study_positive_evidence import arm_hash
    return {'protocol_hash':protocol_hash,'driver_hash':arm_hash(unit['driver_arm'],cals),
            'calibration_hash':cals[unit['observed_calibration_arm']]['calibration_hash'],'trace_hash':canonical(unit['trace'])}


def replay_proof(unit,result):
    trace=unit['trace']
    if not trace or any(type(t) is not int or t<0 for t in trace):raise Invalid('Replay requires a nonempty valid saved token trace')
    expected={'decode_forwards':len(trace)-1,'mask_hash':canonical([[j,t] for j,t in enumerate(trace[:-1],1)])}
    if (any(result.get(k)!=v for k,v in expected.items()) or len(result.get('argmaxes',[]))!=len(trace)
        or not result.get('logits_finiteness_check') or (unit['source_argmax_required'] and result['argmaxes']!=trace)):
        raise Invalid('Measured replay actual trace/argmax/mask/finiteness proof differs')
    return result


def audit_observer_archive(attempt,unit,cals,protocol_hash):
    """Original full/bin/common collector contract; metadata/positions only."""
    import numpy as np
    attempt=Path(attempt);data=attempt/'diagnostics';sid=unit['sample_id'];scenario=unit['scenario'];trace=unit['trace']
    stem=canonical([str(sid),scenario]);ids=identity(unit,cals,protocol_hash)
    positions=[[j,t] for j,t in enumerate(trace[:-1],1)];files=[]
    if not positions:
        marker=data/(stem+'.empty.json')
        if not marker.is_file():raise Pending('Missing immediate-EOS observer marker')
        m=load(marker)
        expected={'schema_version':'state_collection.empty.v1','collection_schema_version':'state_collection.v2','prompt_id':str(sid),
                  'path':scenario,'count':0,'dtype':'float64','ddof':0,'metadata':ids,'position_bins':[list(x) for x in base.BINS],
                  'numerical_protection':base.EPS}
        if any(m.get(k)!=v for k,v in expected.items()) or not m.get('reason'):raise Invalid('Empty observer identity differs')
        if (data/(stem+'.complete.json')).exists():raise Invalid('Conflicting empty/archive observer evidence')
        files=[marker];records=0
    else:
        cp=data/(stem+'.complete.json');archive=data/(stem+'.npz');tokens=data/(stem+'.tokens.jsonl')
        if not cp.is_file():raise Pending('Observer archive has no committed pair')
        commit=load(cp)
        if (commit.get('schema_version')!='state_collection.commit.v1' or commit.get('collection_schema_version')!='state_collection.v2'
            or commit.get('archive')!=archive.name or commit.get('tokens')!=tokens.name or commit.get('token_rows')!=48*len(positions)
            or not archive.is_file() or not tokens.is_file() or sha(archive)!=commit.get('archive_sha256') or sha(tokens)!=commit.get('tokens_sha256')):
            raise Invalid('Observer committed archive/token pair differs')
        with tokens.open('rb') as handle:
            if sum(1 for line in handle if line.endswith(b'\n'))!=48*len(positions):raise Invalid('Observer token row count differs')
        groups={'all':positions,**{f'{lo}-{hi}':[p for p in positions if lo<=p[0]<=hi] for lo,hi in base.BINS}}
        if unit['common_decode_limit'] is not None:
            common=positions[:unit['common_decode_limit']]
            groups.update({'common':common,**{f'common:{lo}-{hi}':[p for p in common if lo<=p[0]<=hi] for lo,hi in base.BINS}})
        groups={k:v for k,v in groups.items() if v};seen=set()
        with np.load(archive,allow_pickle=False) as arrays,zipfile.ZipFile(archive) as zipped:
            meta=json.loads(str(arrays['metadata_json'].item()))
            if any(meta.get(k)!=v for k,v in {'schema_version':'state_collection.v2','dtype':'float64','ddof':0,'moment_representation':'count_mean_centered_m2','position_bins':[list(x) for x in base.BINS],'numerical_protection':base.EPS}.items()):raise Invalid('Observer numerical contract differs')
            for record in meta['records']:
                pid,path,layer,group=record['key'];key=(layer,group)
                if pid!=str(sid) or path!=scenario or type(layer) is not int or layer not in range(48) or group not in groups or key in seen:raise Invalid('Observer48 layer/group identity differs')
                selected=groups[group];actual=arrays[record['prefix']+'_positions']
                if (record.get('count')!=len(selected) or record.get('mask_hash')!=canonical(selected)
                    or record.get('driver_hash')!=ids['driver_hash'] or record.get('calibration_hash')!=ids['calibration_hash']
                    or actual.dtype!=np.int64 or actual.tolist()!=selected):raise Invalid('Observer full/bin/common actual positions differ')
                for suffix in ('native_mean','native_m2','calibrated_mean','calibrated_m2','sums'):
                    with zipped.open(record['prefix']+'_'+suffix+'.npy') as handle:
                        version=np.lib.format.read_magic(handle)
                        shape,fortran,dtype=(np.lib.format.read_array_header_1_0(handle) if version==(1,0) else np.lib.format.read_array_header_2_0(handle))
                    if dtype!=np.dtype('float64') or shape!=((10,) if suffix=='sums' else (2048,)):raise Invalid('Observer FP64 moment/sum header differs')
                seen.add(key)
            if seen!={(l,g) for l in range(48) for g in groups}:raise Invalid('Observer incomplete layer/bin rectangle')
            records=len(seen)
        files=[archive,tokens,cp]
    return {'sample_id':sid,'scenario':scenario,'decode_forwards':len(positions),'layer_count':48 if positions else 0,
            'statistic_records':records,'moment_headers_checked':records*5,'full_moment_value_scan':False,
            'identity':ids,'files':[{'relative_path':str(p.relative_to(attempt)),'sha256':sha(p),'bytes':p.stat().st_size} for p in files]}


def signed_differences(on,off):
    return {'replay_seconds':on['timing']['replay_seconds']-off['timing']['replay_seconds'],
            **{key:on['memory'][key]-off['memory'][key] for key in ('baseline_allocated_bytes','baseline_reserved_bytes','peak_allocated_bytes','peak_reserved_bytes','incremental_peak_allocated_bytes','incremental_peak_reserved_bytes')}}


def audit_pair(directory,unit,cals,protocol_hash):
    directory=Path(directory);path=directory/'pairs'/(unit['unit_hash']+'.json')
    if not path.is_file():raise Pending('Measured on/off pair is not committed')
    pair=load(path);verify_seal(pair,'pair_hash')
    if pair.get('schema_version')!='state_observer_cost.pair.v1' or pair.get('unit_hash')!=unit['unit_hash'] or pair.get('order')!=unit['order']:
        raise Invalid('Pair specification/order differs')
    attempt=(directory/pair['attempt_relative_dir']).resolve()
    if not attempt.is_relative_to((directory/'attempts'/unit['unit_hash']).resolve()):raise Invalid('Pair attempt escapes its exact unit')
    modes={};files=[]
    intent_path=attempt/'intent.json';warm_path=attempt/'warmup.json'
    intent=load(intent_path);warm=load(warm_path)
    expected_intent={'plan_hash':pair.get('plan_hash'),'source_snapshot_hash':pair.get('source_snapshot_hash'),
                     'unit_hash':unit['unit_hash'],'order':unit['order']}
    if any(intent.get(k)!=v for k,v in expected_intent.items()):raise Invalid('Attempt intent plan/source/unit differs')
    warm_unit={**unit,'trace':unit['trace'][:8]}
    replay_proof(warm_unit,warm.get('replay',{}))
    if (warm.get('excluded_from_measurement') is not True or warm.get('trace_length')!=len(warm_unit['trace'])
        or warm.get('observer_calls')!=48*(len(warm_unit['trace'])-1)
        or not isinstance(warm.get('elapsed_seconds'),(int,float)) or isinstance(warm['elapsed_seconds'],bool)
        or not math.isfinite(warm['elapsed_seconds']) or warm['elapsed_seconds']<0):raise Invalid('Actual observer warmup differs')
    for p in (intent_path,warm_path):files.append({'relative_path':str(p.relative_to(directory)),'sha256':sha(p),'bytes':p.stat().st_size})
    for mode in ('on','off'):
        ref=pair['modes'][mode];p=directory/ref['relative_path']
        if p.resolve()!=attempt/(mode+'.mode.json') or sha(p)!=ref['sha256']:raise Invalid('Mode receipt path/hash differs')
        value=load(p);verify_seal(value,'mode_hash')
        if (value.get('schema_version')!='state_observer_cost.mode.v1' or value.get('mode')!=mode
            or value.get('unit_hash')!=unit['unit_hash'] or value.get('plan_hash')!=pair.get('plan_hash')
            or value.get('capture_logits') is not False):raise Invalid('Mode scope differs')
        replay_proof(unit,value['replay'])
        expected_calls=48*(len(unit['trace'])-1) if mode=='on' else 0
        if value.get('observer_calls')!=expected_calls:raise Invalid('Actual observer callback count differs')
        for key in ('replay_seconds','observer_callback_host_seconds','archive_seconds'):
            x=value['timing'].get(key)
            if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0:raise Invalid('Invalid measured timing')
        memory=value['memory']
        for key in ('baseline_allocated_bytes','baseline_reserved_bytes','peak_allocated_bytes','peak_reserved_bytes','incremental_peak_allocated_bytes','incremental_peak_reserved_bytes'):
            if type(memory.get(key)) is not int or memory[key]<0:raise Invalid('Invalid measured byte count')
        for kind in ('allocated','reserved'):
            if memory[f'peak_{kind}_bytes']-memory[f'baseline_{kind}_bytes']!=memory[f'incremental_peak_{kind}_bytes']:raise Invalid('Peak-minus-baseline differs')
        if mode=='on':
            archived=audit_observer_archive(attempt,unit,cals,protocol_hash)
            if value.get('archive_receipt')!=archived:raise Invalid('Mode observer archive no longer matches')
            files.extend({'relative_path':str((attempt/f['relative_path']).relative_to(directory)),**{k:v for k,v in f.items() if k!='relative_path'}} for f in archived['files'])
        elif value.get('archive_receipt') is not None or value['timing']['observer_callback_host_seconds']!=0 or value['timing']['archive_seconds']!=0:
            raise Invalid('Observer-off mode contains observer/archive work')
        modes[mode]=value;files.append({'relative_path':str(p.relative_to(directory)),'sha256':sha(p),'bytes':p.stat().st_size})
    if modes['on']['replay']['argmaxes']!=modes['off']['replay']['argmaxes']:raise Invalid('Observer changed actual driver outputs')
    delta=signed_differences(modes['on'],modes['off'])
    if pair.get('signed_differences')!=delta:raise Invalid('Pair signed differences no longer match')
    files.append({'relative_path':str(path.relative_to(directory)),'sha256':sha(path),'bytes':path.stat().st_size})
    return {'unit_hash':unit['unit_hash'],'sample_id':unit['sample_id'],'scenario':unit['scenario'],'pair_hash':pair['pair_hash'],
            'files':files,'signed_differences':delta}


def raw_receipt(directory,plan,snapshot,units,cals):
    directory=Path(directory)
    if load(directory/'source_snapshot.json')!=snapshot:raise Invalid('Observer source snapshot differs')
    pairs=[audit_pair(directory,u,cals,plan['protocol_hash']) for u in units]
    for u in units:
        p=load(directory/'pairs'/(u['unit_hash']+'.json'))
        if p.get('plan_hash')!=plan['plan_hash'] or p.get('source_snapshot_hash')!=snapshot['snapshot_hash']:raise Invalid('Pair plan/source differs')
    return sealed({'schema_version':'state_observer_cost.raw.v1','plan_hash':plan['plan_hash'],
                   'source_snapshot_hash':snapshot['snapshot_hash'],'pairs':pairs},'raw_completion_hash')


def verify_raw_completion(directory,plan,snapshot,units,cals):
    saved=load(Path(directory)/'raw_completion.json');verify_seal(saved,'raw_completion_hash')
    if saved!=raw_receipt(directory,plan,snapshot,units,cals):raise Invalid('Observer raw completion changed')
    return saved


def write_manifest(directory,plan,stage,status_value,**details):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);path=directory/'manifest.json'
    value=load(path) if path.is_file() else {'schema_version':'state_observer_cost.manifest.v1','plan_hash':plan['plan_hash'],'stages':{}}
    if value.get('plan_hash')!=plan['plan_hash']:raise Invalid('Observer-cost manifest belongs to another plan')
    value['stages'][stage]={'status':status_value,'at_utc':utc(),**details};value['status']=status_value
    temp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.pending')
    try:
        with temp.open('x') as handle:json.dump(value,handle,allow_nan=False,indent=2);handle.write('\n');handle.flush();os.fsync(handle.fileno())
        os.replace(temp,path)
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:temp.unlink(missing_ok=True)


def make_observer(driver,collector,unit,calibrations,cals,protocol_hash,*,clock=time.perf_counter,torch_module=None,shadow_fn=None):
    if torch_module is None:
        import torch as torch_module
    if shadow_fn is None:
        from moe_research.state_adapter import shadow_branches as shadow_fn
    calibration=calibrations[unit['observed_calibration_arm']];trace=unit['trace'];ids=identity(unit,cals,protocol_hash)
    counters={'calls':0,'host_seconds':0.0}
    def observer(block,h,output,phase,layer,step,route):
        if phase!='decode':return
        start=clock()
        try:
            branches=shadow_fn(block,h);z=calibration.apply(layer,branches['y4_full'])
            if unit['driver_arm']!='N' and not torch_module.equal(z,output):raise Invalid('Observer actual driver/shadow output mismatch')
            collector.observe(prompt_id=unit['sample_id'],path=unit['scenario'],layer=layer,decode_forward_index=step,
                token_id=trace[step-1],y8=branches['y8'],u4=branches['u4'],z4=z,residual=driver.residuals[layer],
                driver_hash=ids['driver_hash'],calibration_hash=ids['calibration_hash'],common_decode_limit=unit['common_decode_limit'])
            counters['calls']+=1
        finally:counters['host_seconds']+=clock()-start
    return observer,counters


def measured_mode(driver,unit,mode,calibrations,cals,plan,attempt,*,cuda,collector_factory,clock=time.perf_counter,observer_factory=make_observer):
    """Injectable CUDA facade is fake in tests; real calls only after ops guard."""
    attempt=Path(attempt);publish_equal(attempt/(mode+'.intent.json'),{'mode':mode,'unit_hash':unit['unit_hash'],'at_utc':utc()})
    collector=collector_factory(attempt/'diagnostics') if mode=='on' else None
    observer,counters=observer_factory(driver,collector,unit,calibrations,cals,plan['protocol_hash']) if mode=='on' else (None,{'calls':0,'host_seconds':0.0})
    gc.collect();cuda.empty_cache();cuda.synchronize();cuda.reset_peak_memory_stats()
    memory={'baseline_allocated_bytes':int(cuda.memory_allocated()),'baseline_reserved_bytes':int(cuda.memory_reserved())}
    start=clock();result,_=driver.replay(unit['sample'],unit['trace'],unit['driver_arm'],calibrations,observer,capture_logits=False)
    cuda.synchronize();elapsed=clock()-start
    memory.update(peak_allocated_bytes=int(cuda.max_memory_allocated()),peak_reserved_bytes=int(cuda.max_memory_reserved()))
    for kind in ('allocated','reserved'):memory[f'incremental_peak_{kind}_bytes']=memory[f'peak_{kind}_bytes']-memory[f'baseline_{kind}_bytes']
    replay_proof(unit,result);archive=None;archive_seconds=0.0
    if mode=='on':
        archive_start=clock()
        collector.finish_prompt(unit['sample_id'],unit['scenario'],allow_empty=len(unit['trace'])==1,
            empty_metadata=identity(unit,cals,plan['protocol_hash']))
        archive=audit_observer_archive(attempt,unit,cals,plan['protocol_hash']);archive_seconds=clock()-archive_start
    value=sealed({'schema_version':'state_observer_cost.mode.v1','mode':mode,'unit_hash':unit['unit_hash'],'plan_hash':plan['plan_hash'],
        'capture_logits':False,'replay':result,'observer_calls':counters['calls'],
        'timing':{'replay_seconds':elapsed,'observer_callback_host_seconds':counters['host_seconds'],'archive_seconds':archive_seconds},
        'memory':memory,'archive_receipt':archive},'mode_hash')
    publish_equal(attempt/(mode+'.mode.json'),value);return value


def collect(run,plan,directory,snapshot,units,cals,*,driver_factory=None,cuda=None):
    import torch
    from moe_research.state_calibration import FrozenStateCalibration
    from moe_research.state_collection import StateCollector
    directory=Path(directory);publish_equal(directory/'source_snapshot.json',snapshot)
    pending=[]
    for unit in units:
        try:audit_pair(directory,unit,cals,plan['protocol_hash'])
        except Pending:pending.append(unit)
    if not pending:
        receipt=raw_receipt(directory,plan,snapshot,units,cals);publish_equal(directory/'raw_completion.json',receipt);return receipt
    if driver_factory is not None and cuda is None:raise Invalid('Fake driver requires an explicit fake CUDA facade')
    cuda=torch.cuda if cuda is None else cuda;driver=None
    calibrations={a:FrozenStateCalibration.from_dict(v) for a,v in cals.items()};protocol=load(Path(run)/'protocol.json');samples=base.dataset_samples(protocol)
    try:
        start=time.perf_counter();driver=(driver_factory or base._load_driver)(protocol)
        publish_equal(directory/'model_load_attempts'/(uuid.uuid4().hex+'.json'),{'elapsed_seconds':time.perf_counter()-start,'at_utc':utc(),'plan_hash':plan['plan_hash']})
        if sorted(driver.eos)!=load(Path(run)/'manifest.json').get('runtime',{}).get('eos_ids'):raise Invalid('Actual benchmark EOS configuration differs')
        for original in pending:
            unit={**original,'sample':samples[original['sample_id']]}
            text,encoded=driver.prompt(unit['sample'])
            if text!=unit['generation_row']['rendered_prompt'] or encoded[0].tolist()!=unit['generation_row']['prompt_token_ids']:raise Invalid('Actual benchmark prompt differs from saved source')
            attempt=directory/'attempts'/unit['unit_hash']/uuid.uuid4().hex
            publish_equal(attempt/'intent.json',{'plan_hash':plan['plan_hash'],'source_snapshot_hash':snapshot['snapshot_hash'],
                'unit_hash':unit['unit_hash'],'order':unit['order'],'at_utc':utc(),'pid':os.getpid()})
            warm={**unit,'trace':unit['trace'][:8]};warm_collector=StateCollector()
            observer,counters=make_observer(driver,warm_collector,warm,calibrations,cals,plan['protocol_hash'])
            cuda.synchronize();warm_start=time.perf_counter()
            warm_result,_=driver.replay(unit['sample'],warm['trace'],unit['driver_arm'],calibrations,observer,capture_logits=False)
            cuda.synchronize();replay_proof(warm,warm_result)
            publish_equal(attempt/'warmup.json',{'excluded_from_measurement':True,'trace_length':len(warm['trace']),
                'elapsed_seconds':time.perf_counter()-warm_start,'replay':warm_result,'observer_calls':counters['calls']})
            del observer,warm_collector
            modes={}
            for mode in unit['order']:
                modes[mode]=measured_mode(driver,unit,mode,calibrations,cals,plan,attempt,cuda=cuda,collector_factory=StateCollector)
            if modes['on']['replay']['argmaxes']!=modes['off']['replay']['argmaxes']:raise Invalid('Measured observer on/off argmaxes differ')
            pair=sealed({'schema_version':'state_observer_cost.pair.v1','plan_hash':plan['plan_hash'],'source_snapshot_hash':snapshot['snapshot_hash'],
                'unit_hash':unit['unit_hash'],'order':unit['order'],'attempt_relative_dir':str(attempt.relative_to(directory)),
                'modes':{m:{'relative_path':str((attempt/(m+'.mode.json')).relative_to(directory)),'sha256':sha(attempt/(m+'.mode.json'))} for m in ('on','off')},
                'signed_differences':signed_differences(modes['on'],modes['off'])},'pair_hash')
            publish_equal(directory/'pairs'/(unit['unit_hash']+'.json'),pair)
            audit_pair(directory,original,cals,plan['protocol_hash'])
            write_manifest(directory,plan,'collection','running',last_unit_hash=unit['unit_hash'])
        receipt=raw_receipt(directory,plan,snapshot,units,cals);publish_equal(directory/'raw_completion.json',receipt)
        write_manifest(directory,plan,'collection','complete',pair_count=len(units));return receipt
    finally:
        if driver is not None:driver.close();del driver
        gc.collect();cuda.empty_cache()


def analyze(run,plan,directory):
    directory=Path(directory)
    if not (directory/'raw_completion.json').is_file():raise Pending('All measured pairs must commit before descriptive analysis')
    snapshot,units,cals=source_snapshot(run,plan);receipt=verify_raw_completion(directory,plan,snapshot,units,cals)
    completion_path=directory/'completion.json';summary_path=directory/'summaries/observer_cost_summary.json'
    if completion_path.is_file():
        done=load(completion_path);verify_seal(done,'completion_hash')
        if done.get('raw_completion_hash')!=receipt['raw_completion_hash'] or done.get('summary_sha256')!=sha(summary_path):raise Invalid('Cost completion/summary changed')
        return done
    pairs=[]
    for unit in units:
        pair=load(directory/'pairs'/(unit['unit_hash']+'.json'));attempt=directory/pair['attempt_relative_dir']
        pairs.append({'sample_id':unit['sample_id'],'scenario':unit['scenario'],'order':unit['order'],'trace_tokens':len(unit['trace']),
            'decode_forwards':len(unit['trace'])-1,'signed_differences':pair['signed_differences'],
            'modes':{m:load(attempt/(m+'.mode.json')) for m in ('on','off')}})
    by_scenario={}
    for scenario in snapshot['scenarios']:
        selected=[p for p in pairs if p['scenario']==scenario]
        by_scenario[scenario]={'n_pairs':len(selected),'signed_difference_descriptive':{
            k:{'mean':statistics.mean(p['signed_differences'][k] for p in selected),'median':statistics.median(p['signed_differences'][k] for p in selected)}
            for k in selected[0]['signed_differences']}}
    committed_attempts={load(directory/'pairs'/(u['unit_hash']+'.json'))['attempt_relative_dir'] for u in units}
    all_attempts={str(p.parent.relative_to(directory)) for p in (directory/'attempts').glob('*/*/intent.json')}
    summary={'schema_version':'state_observer_cost.summary.v1','plan_hash':plan['plan_hash'],'source_snapshot_hash':snapshot['snapshot_hash'],
        'status':'ENGINEERING_SUBSET_ONLY','sample_ids':list(IDS),'scenarios':snapshot['scenarios'],'pair_count':len(pairs),'pairs':pairs,
        'by_scenario':by_scenario,'unit':'seconds and bytes; on-minus-off signed differences','bootstrap_performed':False,
        'incomplete_or_uncommitted_attempts_retained':sorted(all_attempts-committed_attempts),
        'limitations':['One pair per fixed unit, no inferential CI','Not historical observer cost and not full-run cumulative cost',
            'No extrapolation to unmeasured prompts/arms or production serving','Callback host time is not pure device-kernel time',
            'Archive I/O is outside online timing; memory baselines and absolute peaks are reported separately']}
    publish_equal(summary_path,summary)
    done=sealed({'schema_version':'state_observer_cost.complete.v1','plan_hash':plan['plan_hash'],'source_snapshot_hash':snapshot['snapshot_hash'],
        'raw_completion_hash':receipt['raw_completion_hash'],'summary_sha256':sha(summary_path),'completed_work':True},'completion_hash')
    publish_equal(completion_path,done);write_manifest(directory,plan,'analysis','complete');return done


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('freeze','check','status','run','analyze'))
    parser.add_argument('--run-dir',required=True);parser.add_argument('--plan');args=parser.parse_args(argv)
    run=Path(args.run_dir).resolve();plan=None;directory=None;lock=None
    try:
        if args.command=='freeze':result=freeze_plan(run,args.plan)
        elif args.command in ('check','status'):
            result=status(run,args.plan);print(json.dumps(result,allow_nan=False),flush=True)
            return 0 if result['status'] in ('READY','RAW_COMPLETE','COMPLETE') else 2 if result['status'] in ('PLAN_PENDING','WAITING_CORE','WAITING_VALIDATION512') else 3
        else:
            plan=load_plan(run,args.plan);state=status(run,args.plan)
            if state['status']=='COMPLETE':print(json.dumps(state),flush=True);return 0
            if state['status'] in ('INVALID_PLAN','SCIENTIFIC_ERROR'):raise Invalid('Observer-cost prerequisites: '+state['status'])
            if state['status'] not in ('READY','RAW_COMPLETE'):raise Pending('Observer-cost prerequisites: '+state['status'])
            directory=output_dir(run,plan)
            if args.command=='run':
                from state_study_supplement_ops import registered_cost_launch
                registered_cost_launch(run,plan)
            lock=base.local_lock(run,plan,control=CONTROL)
            if args.command=='run' and not (directory/'raw_completion.json').is_file():
                snapshot,units,cals=source_snapshot(run,plan);collect(run,plan,directory,snapshot,units,cals)
            result=analyze(run,plan,directory)
            report=importlib.import_module('state_study_observer_cost_report');delivery=report.ensure_delivery(run,plan_path=args.plan)
            if delivery.get('verified') is not True or delivery.get('delivery_complete') is not True:raise Pending('Engineering delivery is pending: '+str(delivery))
            result={'status':'COMPLETE','completion':result,'delivery_complete':True,'delivery':delivery}
        print(json.dumps(result,allow_nan=False),flush=True);return 0
    except (Pending,quality.Pending) as exc:print(json.dumps({'status':'PENDING','reason':str(exc)}),flush=True);return 2
    except Exception as exc:
        if lock is not None and plan is not None and directory is not None:
            try:write_manifest(directory,plan,'failure','FAILED_IMPLEMENTATION',error=repr(exc))
            except OSError:pass
        print(json.dumps({'status':'REFUSED','reason':repr(exc)}),file=sys.stderr,flush=True);return 3
    finally:
        if lock is not None:lock.close()


if __name__=='__main__':raise SystemExit(main())
