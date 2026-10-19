#!/usr/bin/env python3
"""P0–P5 original-checkpoint BF16 state-source calibration study.

All large data live on scratch, while the repository stores source and reports.
Rows are durable per request; all identity hashes are validated on resume.
"""
from __future__ import annotations
import argparse
import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from moe_research.state_driver import (StudyDriver, digest, write_json, append_jsonl,
                                      read_rows, arm_hash, SEED, SUFFIX, PARSER_VERSION)
from moe_research.state_calibration import CalibrationFitter, FrozenStateCalibration
from moe_research.state_adapter import shadow_branches, ADAPTER_NAME
from moe_research.state_collection import StateCollector


class Study:
    def __init__(self,args):
        import yaml
        self.args=args
        self.config=yaml.safe_load(Path(args.config).read_text())
        supported={'model_id':'Qwen/Qwen3-30B-A3B-Instruct-2507','precision':'bfloat16',
            'batch_size':1,'do_sample':False,'seed':SEED,'native_k':8,'decode_k':4,'prefill':'native',
            'main_max_new_tokens':2048,'smoke_max_new_tokens':128,'followup_max_new_tokens':4096,
            'followup_truncation_trigger':0.01,'followup_select_prompts':128,
            'fit_range':[0,128],'diag_range':[128,256],'select_range':[256,512],'validate_range':[512,1024],
            'pilot_fit':16,'pilot_diag':16,'pilot_select':32,'chosen_calibration_family':'lda',
            'chosen_student':'L_N','chosen_mismatch_metric':'E_moment','bootstrap_resamples':10000,
            'bootstrap_seed':SEED,'max_method_updates':1,'max_method_arms':8,
            'baseline_arms':['N','F','R8','R16','M_N','L_N']}
        for key,value in supported.items():
            if self.config.get(key)!=value:raise ValueError(f'Unsupported protocol config {key}; implement and preflight a new protocol before changing it')
        self.out=Path(args.run_dir).resolve()
        if self.out.is_relative_to(Path.home().resolve()):
            raise ValueError('Results must be on the data volume, outside Home')
        self.out.mkdir(parents=True,exist_ok=True)
        self.data_dir=Path(self.config['data_dir'])
        dataset_manifest=json.loads((self.data_dir/'data_manifest.json').read_text())
        train_bytes=(self.data_dir/'train.jsonl').read_bytes()
        if hashlib.sha256(train_bytes).hexdigest()!=dataset_manifest['sources']['train']['jsonl_sha256']:
            raise RuntimeError('Pinned train JSONL checksum mismatch')
        self.samples={}
        # JSONL framing is the physical LF, not Unicode separators that may
        # legally occur inside a question string (present in pinned GSM8K).
        for raw in train_bytes.split(b'\n'):
            if not raw:continue
            row=json.loads(raw)
            if row['sample_id'] in self.samples:raise RuntimeError('Duplicate dataset sample ID')
            if row['question_hash']!=hashlib.sha256(row['question'].encode()).hexdigest():raise RuntimeError('Question hash mismatch')
            if row['dataset_revision']!=self.config['dataset_revision'] or row['split']!='train':raise RuntimeError('Sample source revision/split mismatch')
            self.samples[row['sample_id']]=row
        self.calibrations={}
        self.driver=None
        self._freeze_protocol()

    def _freeze_protocol(self):
        sources=sorted([*REPO.glob('moe_research/*.py'),REPO/'scripts/_harness.py',Path(__file__),
                        REPO/'configs/state_calibration_study.yaml'])
        source_hashes={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        model=Path(self.config['model_path'])
        provenance=json.loads((model/'study_download_provenance.json').read_text())
        if provenance['status']!='complete' or provenance['revision']!=self.config['model_revision']:
            raise RuntimeError('Pinned checkpoint download/revision is not complete')
        data=json.loads((self.data_dir/'data_manifest.json').read_text())
        if data['revision']!=self.config['dataset_revision']:
            raise RuntimeError('Dataset revision mismatch')
        protocol={'config':self.config,'source_hash':digest(source_hashes),'model':provenance,
                  'dataset':data,'prompt_suffix':SUFFIX,'parser_version':PARSER_VERSION,
                  'adapter':ADAPTER_NAME,'phase_index_contract':{
                      'first_generated_token':'prefill prediction',
                      'decode_forward_index':'1 consumes first generated token and predicts second',
                      'valid_replay_inputs':'trace[:-1]; emitted EOS never consumed',
                      'positions_weighted':'all effective decode tokens; population moments ddof=0'},
                  'floating_order':'router softmax/normalization FP32; final weights cast BF16 before expert multiplication and ascending-expert BF16 accumulation; scalar/LDA applied in FP32 after BF16 aggregate then cast BF16',
                  'smoke':{'fit':4,'diag':4,'select':4,'cap':128},
                  'pilot':{'fit':16,'diag':16,'select':32,'cap':2048},
                  'formal':{'fit':128,'diag':128,'select':256,'validate':512,'cap':2048},
                  'confirmation':'sealed candidate pool only; independence unverified; P6/P7 gated, not automatically unsealed',
                  'timing':'CUDA synchronize before/after generation and prefill boundary; no observer or residual hooks; one warmup per arm; no serving-speedup claim',
                  'nonfinite_policy':'Check all returned next-token logits before each greedy argmax in generation and replay; abort on NaN/Inf, with the shared checking overhead included in all arms timings',
                  'historical_checkpoint':'same model ID, old local weight digest unavailable; no old/new paired comparisons'}
        self.protocol_hash=digest(protocol)
        target=self.out/'protocol.json'
        if target.exists():
            old=json.loads(target.read_text())
            if old['protocol_hash']!=self.protocol_hash:
                raise RuntimeError('Protocol/source changed. Use a NEW --run-dir; existing raw rows will not be mixed or overwritten.')
        else:
            write_json(target,dict(protocol,protocol_hash=self.protocol_hash))
            write_json(self.out/'source_hashes.json',source_hashes)
            with tarfile.open(self.out/'source_code.tar.gz','w:gz') as archive:
                for p in sources:archive.add(p,arcname=str(p.relative_to(REPO)))
            (self.out/'working_tree.patch').write_bytes(subprocess.check_output(['git','diff'],cwd=REPO))
        self.manifest_path=self.out/'manifest.json'
        if not self.manifest_path.exists():
            write_json(self.manifest_path,{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'protocol_hash':self.protocol_hash,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                'model_revision':self.config['model_revision'],'dataset_revision':self.config['dataset_revision'],
                'stages':{},'out':str(self.out),'status':'running'})
        command=' '.join([str(Path(sys.executable)),str(Path(__file__)), 'auto','--run-dir',str(self.out),'--config',str(self.args.config)])
        (self.out/'commands.sh').write_text('#!/bin/sh\n# Source/protocol must match; use a new run directory after code changes.\nexport HF_HOME=/scratch/jgu7/cache/huggingface\nexport UV_CACHE_DIR=/scratch/jgu7/cache/uv\nexport TMPDIR=/scratch/jgu7/tmp\nexport PYTHONDONTWRITEBYTECODE=1\nexport OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8\n'+command+'\n')

    def status(self,stage,status,detail=None):
        manifest=json.loads(self.manifest_path.read_text())
        manifest['stages'][stage]={'status':status,'detail':detail,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        manifest['status']='running' if status not in ('error','FAILED_IMPLEMENTATION') else status
        if self.driver:
            import torch
            manifest['runtime']={'python':sys.version,'torch':torch.__version__,
                'transformers':importlib.metadata.version('transformers'),'cuda':torch.version.cuda,
                'gpu':torch.cuda.get_device_name(self.driver.device),'device':str(self.driver.device),
                'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'dtype':str(next(self.driver.model.parameters()).dtype),
                'attention_backend':self.driver.model.config._attn_implementation,
                'packages':{name:importlib.metadata.version(name) for name in ('torch','transformers','tokenizers','huggingface-hub','safetensors','numpy','kernels','accelerate')},
                'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'slurm_job_end_time':os.environ.get('SLURM_JOB_END_TIME'),
                'slurm_cpus_per_task':os.environ.get('SLURM_CPUS_PER_TASK'),
                'expert_backend':ADAPTER_NAME,'device_map':str(getattr(self.driver.model,'hf_device_map',None)),
                'eos_ids':sorted(self.driver.eos)}
        write_json(self.manifest_path,manifest)
        lines=['# 实验状态',f'协议：`{self.protocol_hash}`','']
        for name,item in manifest['stages'].items():lines.append(f'- {name}: {item["status"]}；{item["detail"]}')
        lines+=['','真实结果、原始数据与续跑命令均在本目录；正式结论尚未证明。','',f'续跑：`sh {self.out}/commands.sh`']
        (self.out/'STATUS.md').write_text('\n'.join(lines)+'\n')
        print(f'[{datetime.datetime.now().isoformat(timespec="seconds")}] {stage}: {status} {detail or ""}',flush=True)

    def load(self):
        if self.driver:return
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        source=self.config['model_path']
        tok=AutoTokenizer.from_pretrained(source,local_files_only=True)
        if tok.pad_token_id is None:tok.pad_token=tok.eos_token
        model=AutoModelForCausalLM.from_pretrained(source,dtype=torch.bfloat16,
            attn_implementation=self.config['attention_backend'],device_map={'':'cuda:0'},local_files_only=True)
        model.eval()
        self.driver=StudyDriver(model,tok)
        self.status('runtime','ready','original Qwen BF16, batch=1, one visible GPU; no quantization')

    def preflight(self):
        self.load()
        target=self.out/'preflight.json'
        if target.exists():
            proof=json.loads(target.read_text())
            if proof.get('status') in ('PASS','PASSED'):
                if proof.get('protocol_hash')!=self.protocol_hash:raise RuntimeError('Cached preflight protocol mismatch')
                for name,hash_value in proof['interface']['source_fingerprints'].items():
                    if hashlib.sha256((REPO/'moe_research'/f'{name}.py').read_bytes()).hexdigest()!=hash_value:
                        raise RuntimeError('Cached preflight source mismatch')
                if proof['interface']['checkpoint_revision']!=self.config['model_revision']:raise RuntimeError('Cached P0 checkpoint mismatch')
                return
        from moe_research.state_preflight import run_preflight
        inner=self.out/'preflight/preflight.json'
        if inner.exists():
            previous=json.loads(inner.read_text())
            if previous.get('status')=='FAILED_IMPLEMENTATION':
                raise RuntimeError('Failed implementation proof is preserved; fix the implementation and use a new run directory')
            if previous.get('status') not in ('RUNNING','PASSED'):
                raise RuntimeError('Unknown previous P0 status; preserve and inspect before resuming')
            # A killed process can leave a partial P0, or finish its inner proof
            # just before the outer commit. Preserve that attempt and rerun P0.
            attempts=self.out/'preflight_attempts'
            attempts.mkdir(exist_ok=True)
            (self.out/'preflight').rename(attempts/f'interrupted_{time.time_ns()}')
        self.status('P0','running','native identity, arm shadows, neutrality, replay and phase checks')
        result=run_preflight(self.driver,self.out/'preflight',samples=[self.samples[i] for i in range(128,132)])
        result['protocol_hash']=self.protocol_hash
        write_json(target,result)
        if result.get('status') not in ('PASS','PASSED'):
            self.status('P0','FAILED_IMPLEMENTATION',{'report':str(target),'status':result.get('status')})
            raise RuntimeError('P0 failed; no scientific generation is authorized until implementation is fixed')
        self.status('P0','complete',{'report':str(target),'status':result['status'],
                                  'case_count':result.get('case_count')})

    def rows(self,scope,arm):
        rows=read_rows(self.out/scope/'generations'/f'{arm}.jsonl',
                         {'protocol_hash':self.protocol_hash,'arm_hash':arm_hash(arm,self.calibrations)})
        for sid,row in rows.items():
            if sid not in self.samples or row.get('question_hash')!=self.samples[sid]['question_hash']:
                raise RuntimeError('Resume question identity mismatch')
            for field,expected in (('dataset_revision',self.config['dataset_revision']),
                                   ('model_revision',self.config['model_revision']),('scope',scope),('arm',arm)):
                if row.get(field)!=expected:raise RuntimeError(f'Resume {field} identity mismatch')
        return rows

    def generate(self,scope,ids,arms,cap):
        self.load()
        existing={arm:self.rows(scope,arm) for arm in arms}
        expected=set(ids)
        for arm,rows in existing.items():
            if not set(rows).issubset(expected):raise RuntimeError(f'Unexpected sample IDs in {scope}/{arm}')
            if any(r.get('max_new_tokens')!=cap for r in rows.values()):raise RuntimeError('Resume generation cap mismatch')
        warmed=set()
        # Complete common prompt blocks, so interrupted experiments preserve pairing.
        for start in range(0,len(ids),8):
            block=ids[start:start+8]
            for arm in arms:
                todo=[i for i in block if i not in existing[arm]]
                if todo and arm not in warmed:
                    self.driver.generate(self.samples[todo[0]],arm,self.calibrations,cap=8)
                    warmed.add(arm)
                for i in todo:
                    try:
                        row,_=self.driver.generate(self.samples[i],arm,self.calibrations,cap=cap)
                        row.update(protocol_hash=self.protocol_hash,dataset_revision=self.config['dataset_revision'],
                                   model_revision=self.config['model_revision'],scope=scope)
                        append_jsonl(self.out/scope/'generations'/f'{arm}.jsonl',row)
                        existing[arm][i]=row
                        if len(existing[arm])%4==0:
                            self.status(scope,'running',{a:len(r) for a,r in existing.items()})
                    except Exception as exc:
                        append_jsonl(self.out/scope/'failures.jsonl',{'sample_id':i,'arm':arm,
                            'protocol_hash':self.protocol_hash,'stop_reason':'oom' if 'out of memory' in str(exc).lower() else 'error',
                            'error':repr(exc),'traceback':traceback.format_exc(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
                        raise
        self.status(scope,'complete',{a:len(r) for a,r in existing.items()})
        return existing

    def calibrate_native(self,scope,count,cap):
        directory=self.out/scope/'calibration'
        directory.mkdir(parents=True,exist_ok=True)
        paths={arm:directory/f'{arm}.json' for arm in ('M_N','L_N')}
        if all(p.exists() for p in paths.values()):
            self.calibrations={a:FrozenStateCalibration.load(p) for a,p in paths.items()}
            for a,c in self.calibrations.items():
                if c.metadata.get('protocol_hash')!=self.protocol_hash or c.metadata.get('prompt_ids')!=list(range(count)) or c.metadata.get('max_new_tokens')!=cap:
                    raise RuntimeError(f'Calibration resume provenance mismatch for {a}')
                expected_family='scalar' if a=='M_N' else 'lda'
                if c.family!=expected_family:raise RuntimeError('Calibration family mismatch')
            masks=json.loads((directory/'fit_position_masks.json').read_text())
            if masks['mask_hash']!=digest(masks['positions']):raise RuntimeError('Saved calibration mask corrupt')
            if any(c.metadata['fit_mask_hash']!=masks['mask_hash'] for c in self.calibrations.values()):raise RuntimeError('Calibration saved mask differs from parameters')
            native=self.rows(scope+'/fit_native','N')
            if set(native)!=set(range(count)):raise RuntimeError('Calibration source rollout IDs differ')
            source_masks={str(i):[[j,t] for j,t in enumerate(native[i]['gen_token_ids'][:-1],1)] for i in range(count)}
            if source_masks!=masks['positions']:raise RuntimeError('Calibration masks differ from saved source traces')
            self.calibration_arm_preflight(scope)
            return
        native=self.generate(scope+'/fit_native',list(range(count)),['N'],cap)['N']
        # Persist valid positions BEFORE any fitting on these traces.
        mask={str(i):[[j,t] for j,t in enumerate(native[i]['gen_token_ids'][:-1],1)] for i in range(count)}
        write_json(directory/'fit_position_masks.json',{'positions':mask,'mask_hash':digest(mask),'phase':'decode'})
        scalar,lda=CalibrationFitter('scalar'),CalibrationFitter('lda')
        start=time.perf_counter()
        for i in range(count):
            def observer(block,h,output,phase,layer,step,route):
                if phase!='decode':return
                b=shadow_branches(block,h)
                scalar.update(layer,**b,prompt_id=i)
                lda.update(layer,**b,prompt_id=i)
            replay,_=self.driver.replay(self.samples[i],native[i]['gen_token_ids'],'N',{},observer)
            if replay['argmaxes']!=native[i]['gen_token_ids']:
                raise RuntimeError('Native fitting replay does not reproduce its unobserved generation')
            if (i+1)%4==0:self.status(scope+'/calibration','running',f'{i+1}/{count} full native traces replayed')
        metadata={'source':'N on native C_fit text','scope':scope,'prompt_ids':list(range(count)),
                  'fit_mask_hash':digest(mask),'protocol_hash':self.protocol_hash,
                  'rollout_token_count':sum(r['generated_tokens'] for r in native.values()),
                  'fit_decode_token_count':sum(len(x) for x in mask.values()),
                  'max_new_tokens':cap,
                  'shadow_expert_token_assignments':48*8*sum(len(x) for x in mask.values()),
                  'source_driver':'native K8; native propagation between layers; full cached replay'}
        write_json(directory/'calibration_cost.json',{'replay_seconds':time.perf_counter()-start,
                   'rollout_seconds':sum(r['generation_seconds'] for r in native.values()),
                   'fit_decode_token_count':metadata['fit_decode_token_count'],
                   'shadow_expert_token_assignments':metadata['shadow_expert_token_assignments']})
        self.calibrations={'M_N':scalar.finalize(metadata),'L_N':lda.finalize(metadata)}
        for arm,calibration in self.calibrations.items():
            if paths[arm].exists():
                if FrozenStateCalibration.load(paths[arm]).hash!=calibration.hash:
                    # Timings are metadata, so interrupted refits must be a new artifact.
                    raise RuntimeError('Partial calibration exists; use a new run dir to preserve its provenance')
            else:calibration.save(paths[arm])
        self.calibration_arm_preflight(scope)
        self.status(scope+'/calibration','complete',{a:c.hash for a,c in self.calibrations.items()})

    def calibration_arm_preflight(self,scope):
        """Check the actual fitted coefficients, not only synthetic identities."""
        from moe_research.state_preflight import _shadow_observer,compare_logits,assert_logit_gate
        path=self.out/scope/'calibration/calibration_arm_preflight.json'
        hashes={a:self.calibrations[a].hash for a in ('M_N','L_N')}
        if path.exists():
            proof=json.loads(path.read_text())
            if proof.get('protocol_hash')!=self.protocol_hash or proof.get('calibration_hashes')!=hashes or proof.get('status')!='PASSED':
                raise RuntimeError('Fitted-arm preflight provenance mismatch')
            return
        results={}
        for arm in self.config['baseline_arms']:
            native,logits=self.driver.generate(self.samples[128],arm,self.calibrations,cap=8,capture_logits=True)
            shadow=[]
            observer=_shadow_observer(arm,self.calibrations,shadow)
            replay,replay_logits=self.driver.replay(self.samples[128],native['gen_token_ids'],arm,
                                   self.calibrations,observer,capture_logits=True)
            metrics=compare_logits(logits,replay_logits)
            assert_logit_gate(metrics)
            if replay['argmaxes']!=native['gen_token_ids']:raise RuntimeError('Fitted-arm replay trace mismatch')
            results[arm]={'logits':metrics,'shadow_max_abs':max(x['max_abs'] for x in shadow),
                          'observed_layer_calls':len(shadow),'calibration_hash':hashes.get(arm)}
        write_json(path,{'status':'PASSED','protocol_hash':self.protocol_hash,
                        'calibration_hashes':hashes,'engineering_cap':8,'arms':results})

    def diagnose(self,scope,ids,cap,student='L_N'):
        self.generate(scope+'/diag_rollouts',ids,['N',student],cap)
        native=self.rows(scope+'/diag_rollouts','N');student_rows=self.rows(scope+'/diag_rollouts',student)
        directory=self.out/scope/'diagnostics'
        collector=self.load_collector(directory)
        collector.output_dir=directory
        calibration=self.calibrations[student]
        calibration_hash=calibration.hash
        for i in ids:
            tn,ts=native[i]['gen_token_ids'],student_rows[i]['gen_token_ids']
            common=min(len(tn),len(ts))-1
            for path,arm,trace in (('A','N',tn),('B',student,tn),('C',student,ts),('D','N',ts)):
                driver_hash=arm_hash(arm,self.calibrations)
                identity={'protocol_hash':self.protocol_hash,'driver_hash':driver_hash,
                          'calibration_hash':calibration_hash,'trace_hash':digest(trace)}
                if self.diagnostic_finished(collector,i,path,trace,identity,common):continue
                def observer(block,h,output,phase,layer,step,route):
                    if phase!='decode':return
                    branches=shadow_branches(block,h)
                    reduced=branches['u4'] if calibration.family=='scalar' else branches['y4_full']
                    z=calibration.apply(layer,reduced)
                    if arm!='N' and not __import__('torch').equal(z,output):
                        raise RuntimeError(f'Actual/shadow arm mismatch {arm}, layer {layer}, step {step}')
                    collector.observe(prompt_id=i,path=path,layer=layer,
                        decode_forward_index=step,token_id=trace[step-1],y8=branches['y8'],
                        u4=branches['u4'],z4=z,residual=self.driver.residuals[layer],
                        driver_hash=driver_hash,calibration_hash=calibration_hash,
                        common_decode_limit=common)
                result,_=self.driver.replay(self.samples[i],trace,arm,self.calibrations,observer)
                if path in ('A','C') and result['argmaxes']!=trace:
                    raise RuntimeError(f'{path} teacher-forced replay failed to reproduce its own greedy logits argmax sequence')
                collector.finish_prompt(i,path,allow_empty=len(trace)==1,empty_metadata=identity)
                append_jsonl(directory/'replay_checks.jsonl',{'sample_id':i,'path':path,**result,**identity})
            self.status(scope+'/diagnostics','running',f'{ids.index(i)+1}/{len(ids)} prompts complete A/B/C/D')
        self.status(scope+'/diagnostics','complete',f'{len(ids)} complete four-path traces; full suffix preserved')
        return collector

    @staticmethod
    def load_collector(directory):
        directory=Path(directory)
        existing=directory.exists() and (list(directory.glob('*.npz')) or list(directory.glob('*.empty.json')))
        return StateCollector.load(directory) if existing else StateCollector(directory)

    def diagnostic_finished(self,collector,pid,path,trace,identity,common=None):
        key=(str(pid),path)
        if key not in collector.finished:return False
        if len(trace)==1:
            if collector.empty_metadata.get(key)!=identity:raise RuntimeError('Empty diagnostic resume identity mismatch')
            return True
        entries=[(k,r) for k,r in collector.records.items() if k[:2]==key]
        all_rows=[(k,r) for k,r in entries if k[3]=='all']
        if {k[2] for k,r in all_rows}!=set(range(48)):raise RuntimeError('Diagnostic resume missing layer records')
        full=[(j,t) for j,t in enumerate(trace[:-1],1)]
        for (_,_,layer,group),record in entries:
            expected=full
            if group=='common' or group.startswith('common:'):
                if common is None:raise RuntimeError('Unexpected common-prefix diagnostic record')
                expected=full[:common]
            bin_name=group.split(':',1)[-1]
            if bin_name not in ('all','common'):
                lo,hi=map(int,bin_name.split('-'))
                expected=[item for item in expected if lo<=item[0]<=hi]
            if record.driver_hash!=identity['driver_hash'] or record.calibration_hash!=identity['calibration_hash'] or record.positions!=expected:
                raise RuntimeError('Diagnostic resume driver/calibration/token-mask identity mismatch')
        return True

    def generation_report(self,scope,ids,arms,comparisons=None):
        from moe_research.state_reporting import build_generation_summary,render_generation_report
        summary=build_generation_summary({a:self.rows(scope,a) for a in arms},expected_ids=ids,
                    bootstrap_iters=self.config['bootstrap_resamples'],comparisons=comparisons)
        write_json(self.out/scope/'summaries/generation_summary.json',summary)
        (self.out/scope/'RUN_REPORT.md').write_text(render_generation_report(summary,context={
                    'scope':scope,'protocol_hash':self.protocol_hash,'model_revision':self.config['model_revision']}))
        return summary

    def analysis(self,scope,iters=None):
        from moe_research.state_study_analysis import analyze_diagnostics
        expected_ids={'smoke':list(range(128,132)),'pilot':list(range(128,144)),
                      'formal':list(range(128,256)),
                      'formal/validation_state':list(range(512,640))}[scope]
        return analyze_diagnostics(self.out/scope/'diagnostics',self.out/scope/'summaries',
            iters=iters or self.config['bootstrap_resamples'],family='lda',
            native_match_tolerance=self.config['native_match_tolerance'],
            state_effect_tolerance=self.config['state_effect_tolerance'],
            expected_prompt_ids=expected_ids)

    def pilot(self,smoke=False):
        scope='smoke' if smoke else 'pilot'
        fit,diag,n,cap=(4,4,4,128) if smoke else (16,16,32,2048)
        self.calibrate_native(scope,fit,cap)
        ids=list(range(256,256+n))
        self.generate(scope+'/select',ids,self.config['baseline_arms'],cap)
        self.generation_report(scope+'/select',ids,self.config['baseline_arms'])
        self.diagnose(scope,list(range(128,128+diag)),cap)
        # Engineering pilot is not a scientific effect/significance gate.
        self.analysis(scope,iters=1000)
        self.status('P1_'+scope,'complete','pipeline only; not pooled into formal results or used to tune sample size')

    def baselines(self):
        self.calibrate_native('formal',128,2048)
        ids=list(range(256,512))
        self.generate('formal/select',ids,self.config['baseline_arms'],2048)
        summary=self.generation_report('formal/select',ids,self.config['baseline_arms'])
        self.status('P2','complete','C_fit=128; six arms on D_select=256; paired generation saved')
        return summary

    def formal_diagnose(self):
        self.calibrate_native('formal',128,2048)
        self.diagnose('formal',list(range(128,256)),2048)
        result=self.analysis('formal')
        self.status('P3','complete',{'report':str(self.out/'formal/summaries/diagnostic_summary.json'),
                                  'status':result['status'],'gate_recalibration':result['gate_recalibration']})
        return result

    def recalibrate(self,diagnostic=None):
        """One matched-text source update, only after the declared P3 gate."""
        self.calibrate_native('formal',128,2048)
        diagnostic=diagnostic or json.loads((self.out/'formal/summaries/diagnostic_summary.json').read_text())
        if not diagnostic['gate_recalibration']:
            self.status('P4','scientifically_not_triggered',diagnostic['status'])
            return False
        directory=self.out/'formal/calibration'
        targets={a:directory/f'{a}.json' for a in ('M_NT','M_ST')}
        if not all(p.exists() for p in targets.values()):
            student=self.generate('formal/fit_student',list(range(128)),['L_N'],2048)['L_N']
            masks={str(i):[[j,t] for j,t in enumerate(student[i]['gen_token_ids'][:-1],1)] for i in range(128)}
            write_json(directory/'student_position_masks.json',{'positions':masks,'mask_hash':digest(masks)})
            costs={}
            for name,driver_arm in (('M_NT','N'),('M_ST','L_N')):
                if targets[name].exists():continue
                fitter=CalibrationFitter('lda')
                start=time.perf_counter()
                for i in range(128):
                    def observer(block,h,output,phase,layer,step,route):
                        if phase=='decode':fitter.update(layer,**shadow_branches(block,h),prompt_id=i)
                    replay,_=self.driver.replay(self.samples[i],student[i]['gen_token_ids'],driver_arm,self.calibrations,observer)
                    if driver_arm=='L_N' and replay['argmaxes']!=student[i]['gen_token_ids']:
                        raise RuntimeError('S0 calibration replay differs from its observer-free rollout')
                    if (i+1)%4==0:self.status('P4_fit_'+name,'running',f'{i+1}/128 matched-text traces')
                metadata={'source':name,'family':'lda','source_driver':driver_arm,
                    'source_driver_hash':arm_hash(driver_arm,self.calibrations),
                    'source_student_calibration_hash':self.calibrations['L_N'].hash,
                    'fit_mask_hash':digest(masks),'prompt_ids':list(range(128)),
                    'protocol_hash':self.protocol_hash,'fit_decode_token_count':sum(map(len,masks.values())),
                    'rollout_token_count':sum(r['generated_tokens'] for r in student.values()),
                    'target':'same-h shadow y8','max_new_tokens':2048}
                fitter.finalize(metadata).save(targets[name])
                costs[name]={'replay_seconds':time.perf_counter()-start,
                    'fit_decode_tokens':metadata['fit_decode_token_count'],
                    'shadow_expert_token_assignments':48*8*metadata['fit_decode_token_count'],
                    'rollout_seconds_shared':sum(r['generation_seconds'] for r in student.values())}
            append_jsonl(directory/'updated_calibration_cost_attempts.jsonl',costs)
        for name,path in targets.items():self.calibrations[name]=FrozenStateCalibration.load(path)
        saved_masks=json.loads((directory/'student_position_masks.json').read_text())
        if saved_masks['mask_hash']!=digest(saved_masks['positions']):raise RuntimeError('NT/ST saved mask is corrupt')
        student_sources=self.rows('formal/fit_student','L_N')
        if set(student_sources)!=set(range(128)):raise RuntimeError('NT/ST source rollout IDs mismatch')
        source_masks={str(i):[[j,t] for j,t in enumerate(student_sources[i]['gen_token_ids'][:-1],1)] for i in range(128)}
        if source_masks!=saved_masks['positions']:raise RuntimeError('NT/ST masks differ from source student traces')
        for name in ('M_NT','M_ST'):
            c=self.calibrations[name]
            if (c.family!='lda' or c.metadata.get('protocol_hash')!=self.protocol_hash
                or c.metadata.get('prompt_ids')!=list(range(128)) or c.metadata.get('max_new_tokens')!=2048
                or c.metadata.get('fit_mask_hash')!=saved_masks['mask_hash']
                or c.metadata.get('source_student_calibration_hash')!=self.calibrations['L_N'].hash
                or c.metadata.get('source_driver_hash')!=arm_hash('N' if name=='M_NT' else 'L_N',self.calibrations)
                or c.metadata.get('source')!=name):
                raise RuntimeError(f'Updated calibration provenance mismatch for {name}')
        if self.calibrations['M_NT'].metadata['fit_mask_hash']!=self.calibrations['M_ST'].metadata['fit_mask_hash']:
            raise RuntimeError('NT/ST fitting masks differ')
        self.own_trajectories('formal',list(range(128,256)),['M_NT','M_ST'])
        self.generate('formal/select',list(range(256,512)),['M_NT','M_ST'],2048)
        self.generation_report('formal/select',list(range(256,512)),self.config['baseline_arms']+['M_NT','M_ST'],
                               comparisons=[('M_ST','M_NT')])
        updated=self.analysis('formal')
        self.status('P4','complete','one update; both updated models rerun on their own C_diag trajectories and D_select')
        return True

    def own_trajectories(self,scope,ids,arms):
        rows=self.generate(scope+'/updated_diag_rollouts',ids,arms,2048)
        directory=self.out/scope/'diagnostics'
        collector=self.load_collector(directory)
        collector.output_dir=directory
        import torch
        for arm in arms:
            calibration=self.calibrations[arm]
            driver_hash,calibration_hash=arm_hash(arm,self.calibrations),calibration.hash
            for i in ids:
                trace=rows[arm][i]['gen_token_ids']
                identity={'protocol_hash':self.protocol_hash,'driver_hash':driver_hash,
                          'calibration_hash':calibration_hash,'trace_hash':digest(trace)}
                if self.diagnostic_finished(collector,i,arm,trace,identity):continue
                def observer(block,h,output,phase,layer,step,route):
                    if phase!='decode':return
                    branches=shadow_branches(block,h)
                    reduced=branches['u4'] if calibration.family=='scalar' else branches['y4_full']
                    z=calibration.apply(layer,reduced)
                    if not torch.equal(z,output):raise RuntimeError(f'Updated driver/shadow mismatch {arm}, {layer}, {step}')
                    collector.observe(prompt_id=i,path=arm,layer=layer,decode_forward_index=step,
                        token_id=trace[step-1],y8=branches['y8'],u4=branches['u4'],z4=z,
                        residual=self.driver.residuals[layer],driver_hash=driver_hash,calibration_hash=calibration_hash)
                result,_=self.driver.replay(self.samples[i],trace,arm,self.calibrations,observer)
                if result['argmaxes']!=trace:raise RuntimeError(f'{arm} own replay argmax mismatch')
                collector.finish_prompt(i,arm,allow_empty=len(trace)==1,empty_metadata=identity)
                append_jsonl(directory/'replay_checks.jsonl',{'sample_id':i,'path':arm,**result,**identity})
                if (i+1)%4==0:self.status(scope+'/updated_diagnostics','running',f'{arm} sample {i}')

    def validate(self,updated=False):
        """Freeze D_select choices before any D_validate observation."""
        from moe_research.state_reporting import choose_baseline
        arms=self.config['baseline_arms']+(['M_NT','M_ST'] if updated else [])
        summary=self.generation_report('formal/select',list(range(256,512)),arms)
        reference=choose_baseline(summary)['selected_reference']
        ranked=sorted((a for a in self.config['baseline_arms'] if a!='N'),
            key=lambda a:(-min(summary['arms'][a]['strict_accuracy'],summary['arms'][a]['tolerant_accuracy']),
                          summary['arms'][a]['length']['mean'],self.config['baseline_arms'].index(a)))
        strongest=ranked[0]
        finalists=list(dict.fromkeys(['N',reference,strongest,'L_N']+(['M_NT','M_ST'] if updated else [])))
        freeze={'protocol_hash':self.protocol_hash,'selected_reference':reference,
                'strongest_existing_baseline':strongest,'initial_calibration':'L_N',
                'finalist_arms':finalists,'selection_sample_ids':list(range(256,512)),
                'validation_sample_ids':list(range(512,1024)),
                'primary_downstream_target':'complete generated-token cost, paired; minimum useful relative reduction 5%',
                'minimum_useful_relative_token_reduction':0.05,
                'quality_noninferiority_margin':None,
                'quality_rule':'Report both strict and tolerant paired CIs. No quality-preservation or noninferiority claim is inferred; confirmation is not triggered without a powered prospective quality plan.',
                'updated':updated,'max_method_updates':1,
                'negative_state_replication':'No refitting; A/B/C/D on fixed D_validate first128 prompts',
                'followup_trigger':0.01,'followup_prompt_ids':list(range(256,384)),
                'followup_cap':4096}
        path=self.out/'validation_freeze.json'
        if path.exists() and json.loads(path.read_text())!=freeze:raise RuntimeError('Frozen validation choices changed')
        if not path.exists():write_json(path,freeze)
        if any(summary['arms'][a]['administrative_cap_rate']>0.01 for a in arms):
            self.generate('formal/select_cap4096',list(range(256,384)),finalists,4096)
            self.generation_report('formal/select_cap4096',list(range(256,384)),finalists,
                comparisons=[('M_ST','M_NT'),('M_ST',strongest)] if updated else [('L_N',strongest)])
        comparisons=[('M_ST','M_NT'),('M_ST',strongest)] if updated else [('L_N',reference),('L_N','N')]
        self.generate('formal/validate',list(range(512,1024)),finalists,2048)
        self.generation_report('formal/validate',list(range(512,1024)),finalists,comparisons)
        if updated:
            self.own_trajectories('formal/validation_state',list(range(512,640)),['M_NT','M_ST'])
            self.analysis('formal/validation_state')
        else:
            self.diagnose('formal/validation_state',list(range(512,640)),2048)
            self.analysis('formal/validation_state')
        self.status('P5','complete','D_validate=512 with frozen comparisons; first128 fixed state replication; no confirmation unsealed')

    def report(self):
        lines=['# 当前实验报告','',
          f'实际模型：Qwen3-30B-A3B-Instruct-2507，revision `{self.config["model_revision"]}`，BF16，eager attention，batch=1。',
          '所有新结果使用同一模型与精度；旧本地权重 hash 不可核实，旧实验不并入新配对比较。',
          '校准来源假设尚须完整证据链检验；工程 smoke/pilot 不作为机制结论。',
          'C_diag/D_select/D_validate 只是本次拟合留出；历史独立性未证明。',
          '状态：INCONCLUSIVE；最终综合结论以 formal/summaries 及正式内部复核为依据，未运行阶段不能算负结果。','']
        m=json.loads(self.manifest_path.read_text())
        for name,item in m['stages'].items():lines.append(f'- {name}: {item["status"]}；{item["detail"]}')
        lines+=['','原始数据可用于重算。完成到哪个阶段，以 STATUS.md、manifest 与逐样本文件为准。',
                f'续跑：`sh {self.out}/commands.sh`']
        (self.out/'RUN_REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['preflight','smoke','pilot','baselines','diagnose','recalibrate','validate','report','auto'])
    parser.add_argument('--config',default=str(REPO/'configs/state_calibration_study.yaml'))
    parser.add_argument('--run-dir',required=True)
    args=parser.parse_args()
    study=Study(args)
    with (study.out/'study.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if args.command=='report':study.report();return
            study.preflight()
            if args.command=='smoke':study.pilot(smoke=True)
            elif args.command=='pilot':study.pilot()
            elif args.command=='baselines':study.baselines()
            elif args.command=='diagnose':study.formal_diagnose()
            elif args.command=='recalibrate':study.recalibrate()
            elif args.command=='validate':
                study.calibrate_native('formal',128,2048)
                for a in ('M_NT','M_ST'):
                    p=study.out/'formal/calibration'/f'{a}.json'
                    if p.exists():study.calibrations[a]=FrozenStateCalibration.load(p)
                study.validate(updated=all(a in study.calibrations for a in ('M_NT','M_ST')))
            elif args.command=='auto':
                study.pilot(smoke=True)
                study.pilot()
                study.baselines()
                diagnostic=study.formal_diagnose()
                updated=study.recalibrate(diagnostic)
                study.validate(updated=updated)
                study.status('P6_P7','gated_pending','independent confirmation and paper expansion require the complete positive evidence chain and a prospectively powered quality plan')
            study.report()
        except Exception as exc:
            study.status(args.command,'error',repr(exc))
            study.report()
            raise
        finally:
            if study.driver:study.driver.close()


if __name__=='__main__':main()
