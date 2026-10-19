"""CPU-only orchestrator provenance checks with tiny scratch fixtures."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import yaml
import torch

from moe_research.state_calibration import FrozenStateCalibration
from moe_research.state_driver import append_jsonl, arm_hash, digest


SOURCE=Path(__file__).resolve().parents[1]/"scripts/run_state_calibration_study.py"
SPEC=importlib.util.spec_from_file_location("state_runner_fixture",SOURCE)
runner=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class StateRunnerTests(unittest.TestCase):
    def setUp(self):
        scratch=Path("/scratch/jgu7/tmp");scratch.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.repo=self.root/"repo"
        (self.repo/"scripts").mkdir(parents=True)
        (self.repo/"configs").mkdir()
        (self.repo/"moe_research").mkdir()
        for name in ("scripts/run_state_calibration_study.py","scripts/_harness.py"):
            (self.repo/name).write_text("# provenance fixture\n")
        self.data=self.root/"data";self.data.mkdir()
        self.model=self.root/"model";self.model.mkdir()
        self.config=yaml.safe_load((SOURCE.parents[1]/"configs/state_calibration_study.yaml").read_text())
        self.config.update(data_dir=str(self.data),model_path=str(self.model))
        (self.model/"study_download_provenance.json").write_text(json.dumps({
            "status":"complete","revision":self.config["model_revision"]}))
        self.config_path=self.repo/"configs/state_calibration_study.yaml"
        self.config_path.write_text(yaml.safe_dump(self.config))
        self.records=[{"sample_id":i,"split":"train","question":f"q{i}","question_hash":hashlib.sha256(f"q{i}".encode()).hexdigest(),
            "answer":"#### 1","dataset_revision":self.config["dataset_revision"]} for i in range(2)]
        self.write_data(self.records)
        self.args=types.SimpleNamespace(config=str(self.config_path),run_dir=str(self.root/"run"))
        self.patches=[mock.patch.object(runner,"REPO",self.repo),
            mock.patch.object(runner,"__file__",str(self.repo/"scripts/run_state_calibration_study.py")),
            mock.patch.object(runner.subprocess,"check_output",side_effect=self.git_output)]
        for patch in self.patches:patch.start();self.addCleanup(patch.stop)

    @staticmethod
    def git_output(command,**kwargs):
        return "fixture-commit\n" if "rev-parse" in command else b""

    def write_data(self,records):
        content="".join(json.dumps(row)+"\n" for row in records).encode()
        path=self.data/"train.jsonl";path.write_bytes(content)
        (self.data/"data_manifest.json").write_text(json.dumps({"revision":self.config["dataset_revision"],
            "sources":{"train":{"jsonl_path":str(path),"jsonl_sha256":hashlib.sha256(content).hexdigest(),"rows":len(records)}}}))

    def test_unchanged_protocol_resumes_and_config_change_requires_new_run(self):
        first=runner.Study(self.args)
        again=runner.Study(self.args)
        self.assertEqual(first.protocol_hash,again.protocol_hash)
        self.config["native_match_tolerance"]+=.001
        self.config_path.write_text(yaml.safe_dump(self.config))
        with self.assertRaisesRegex(RuntimeError,"Protocol/source changed"):
            runner.Study(self.args)

    def test_duplicate_dataset_ids_are_rejected_before_dict_overwrite(self):
        self.write_data(self.records+[self.records[0]])
        with self.assertRaises((RuntimeError,ValueError)):
            runner.Study(self.args)

    def test_unicode_question_separators_do_not_split_json_records(self):
        records=[dict(row) for row in self.records]
        records[0]['question']='Colored balls.\u0085Still one question.\u2028Final sentence.'
        records[0]['question_hash']=hashlib.sha256(records[0]['question'].encode()).hexdigest()
        content=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in records).encode()
        (self.data/'train.jsonl').write_bytes(content)
        (self.data/'data_manifest.json').write_text(json.dumps({'revision':self.config['dataset_revision'],
            'sources':{'train':{'jsonl_sha256':hashlib.sha256(content).hexdigest(),'rows':len(records)}}}))
        study=runner.Study(self.args)
        self.assertEqual(study.samples[0]['question'],records[0]['question'])
        self.assertEqual(set(study.samples),{0,1})

    def test_dataset_bytes_cannot_change_under_same_revision_and_manifest(self):
        runner.Study(self.args)
        modified=[dict(row) for row in self.records]
        modified[0]["question"]="Changed after provenance was frozen"
        (self.data/"train.jsonl").write_text("".join(json.dumps(row)+"\n" for row in modified))
        with self.assertRaises((RuntimeError,ValueError)):
            runner.Study(self.args)

    def test_calibration_fast_resume_rejects_a_different_protocol_fit(self):
        study=runner.Study(self.args)
        directory=study.out/"formal/calibration";directory.mkdir(parents=True)
        wrong={"protocol_hash":"foreign-protocol","prompt_ids":[500,501],
               "source":"N on native C_fit text","max_new_tokens":2048}
        FrozenStateCalibration("scalar",{"0":{"scale":1.}},metadata=wrong).save(directory/"M_N.json")
        FrozenStateCalibration("lda",{"0":{"mean_reduced":[0.],"mean_native":[0.],
            "std_reduced":[1.],"std_native":[1.]}},metadata=wrong).save(directory/"L_N.json")
        with self.assertRaises((RuntimeError,ValueError)):
            study.calibrate_native("formal",128,2048)

    def test_pass_status_alone_cannot_reuse_a_foreign_preflight(self):
        study=runner.Study(self.args)
        study.load=mock.Mock()
        (study.out/"preflight.json").write_text(json.dumps({"status":"PASS","protocol_hash":"foreign"}))
        with mock.patch("moe_research.state_preflight.run_preflight",side_effect=RuntimeError("must rerun stale gate")):
            with self.assertRaises((RuntimeError,ValueError)):
                study.preflight()

    def test_interrupted_preflight_is_preserved_and_rerun_but_failed_proof_blocks(self):
        study=runner.Study(self.args)
        study.load=mock.Mock()
        study.samples={i:dict(self.records[0],sample_id=i) for i in range(128,132)}
        inner=study.out/'preflight/preflight.json'
        inner.parent.mkdir()
        inner.write_text(json.dumps({'status':'RUNNING','cases':[{'name':'completed_before_interrupt'}]}))
        with mock.patch('moe_research.state_preflight.run_preflight',return_value={'status':'PASSED','case_count':52}) as run:
            study.preflight()
        self.assertEqual(run.call_count,1)
        archived=list((study.out/'preflight_attempts').glob('*/preflight.json'))
        self.assertEqual(len(archived),1)
        self.assertEqual(json.loads(archived[0].read_text())['cases'][0]['name'],'completed_before_interrupt')
        (study.out/'preflight.json').unlink()
        inner.parent.mkdir(exist_ok=True)
        inner.write_text(json.dumps({'status':'FAILED_IMPLEMENTATION'}))
        with mock.patch('moe_research.state_preflight.run_preflight') as run:
            with self.assertRaisesRegex(RuntimeError,'Failed implementation proof'):
                study.preflight()
        run.assert_not_called()

    def test_generation_resume_checks_question_identity_not_only_run_labels(self):
        study=runner.Study(self.args)
        scope="smoke/select"
        append_jsonl(study.out/scope/"generations/N.jsonl",{
            "sample_id":0,"arm":"N","arm_hash":arm_hash("N",{}),
            "protocol_hash":study.protocol_hash,"question_hash":"wrong-question",
            "dataset_revision":self.config["dataset_revision"],
            "model_revision":self.config["model_revision"],"scope":scope})
        with self.assertRaises((RuntimeError,ValueError)):
            study.rows(scope,"N")

    def test_generation_resume_rejects_a_different_token_cap(self):
        study=runner.Study(self.args)
        scope="smoke/select"
        append_jsonl(study.out/scope/"generations/N.jsonl",{
            "sample_id":0,"arm":"N","arm_hash":arm_hash("N",{}),
            "protocol_hash":study.protocol_hash,"question_hash":self.records[0]["question_hash"],
            "dataset_revision":self.config["dataset_revision"],"max_new_tokens":128,
            "model_revision":self.config["model_revision"],"scope":scope})
        study.load=mock.Mock();study.driver=mock.Mock();study.status=mock.Mock()
        with self.assertRaises((RuntimeError,ValueError)):
            study.generate(scope,[0],["N"],2048)

    def test_updated_calibration_resume_rejects_foreign_source_provenance(self):
        study=runner.Study(self.args)
        directory=study.out/"formal/calibration";directory.mkdir(parents=True)
        params={"0":{"mean_reduced":[0.],"mean_native":[0.],"std_reduced":[1.],"std_native":[1.]}}
        study.calibrations={"L_N":FrozenStateCalibration("lda",params,metadata={"protocol_hash":study.protocol_hash})}
        masks={str(i):[[1,10]] for i in range(128)}
        (directory/"student_position_masks.json").write_text(json.dumps({"positions":masks,"mask_hash":digest(masks)}))
        for arm,source in (("M_NT","N"),("M_ST","L_N")):
            wrong={"protocol_hash":"foreign-protocol","source":arm,"source_driver":source,
                   "source_student_calibration_hash":"foreign-initial-fit","fit_mask_hash":digest(masks),
                   "prompt_ids":list(range(128)),"fit_decode_token_count":128,"max_new_tokens":2048}
            FrozenStateCalibration("lda",params,metadata=wrong).save(directory/f"{arm}.json")
        study.calibrate_native=mock.Mock()
        study.own_trajectories=mock.Mock()
        study.generate=mock.Mock(return_value={})
        study.generation_report=mock.Mock(return_value={})
        study.analysis=mock.Mock(return_value={})
        study.status=mock.Mock()
        with self.assertRaises((RuntimeError,ValueError)):
            study.recalibrate({"gate_recalibration":True})
        study.own_trajectories.assert_not_called()

    def test_both_updated_models_replay_their_own_new_text_and_calibration(self):
        study=runner.Study(self.args)
        params={"0":{"mean_reduced":[0.],"mean_native":[.2],"std_reduced":[1.],"std_native":[1.1]}}
        study.calibrations={a:FrozenStateCalibration("lda",params,metadata={"source":a})
                            for a in ("M_NT","M_ST")}
        own={"M_NT":{0:{"gen_token_ids":[10,11,99]}},
             "M_ST":{0:{"gen_token_ids":[17,99]}}}
        study.generate=mock.Mock(return_value=own)
        study.status=mock.Mock()
        branch={"y8":torch.ones(1,1,1),"u4":torch.full((1,1,1),.5),"y4_full":torch.full((1,1,1),.75)}
        visited=[];collected=[]
        class Collector:
            def __init__(self,output):self.finished=set()
            def observe(self,**kwargs):collected.append(kwargs)
            def finish_prompt(self,prompt,path,**kwargs):self.finished.add((str(prompt),path))
        def replay(sample,trace,arm,calibrations,observer):
            visited.append((arm,list(trace)))
            for step in range(1,len(trace)):
                actual=calibrations[arm].apply(0,branch["y4_full"])
                observer(object(),torch.ones(1,1,1),actual,"decode",0,step,{})
            return {"argmaxes":list(trace),"decode_forwards":len(trace)-1},[]
        study.driver=types.SimpleNamespace(residuals={0:torch.ones(1,1,1)},replay=replay)
        with mock.patch.object(runner,"StateCollector",Collector),mock.patch.object(runner,"shadow_branches",return_value=branch):
            study.own_trajectories("formal/validation_state",[0],["M_NT","M_ST"])
        self.assertEqual(visited,[("M_NT",[10,11,99]),("M_ST",[17,99])])
        self.assertEqual([(x["path"],x["token_id"]) for x in collected],[("M_NT",10),("M_NT",11),("M_ST",17)])
        for item in collected:
            self.assertEqual(item["calibration_hash"],study.calibrations[item["path"]].hash)


if __name__=="__main__":unittest.main()
