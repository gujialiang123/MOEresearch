"""Small scratch-only tests of the prospective freeze, not model execution."""
import json
import os
from pathlib import Path
import tempfile
import unittest

from scripts import freeze_state_study_quality_plan as plan


class QualityPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir="/scratch/jgu7/tmp")
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.repo=self.root/"repo";self.run=self.root/"run"
        self.run.mkdir();(self.repo/"moe_research").mkdir(parents=True)
        for i in range(15):(self.repo/f"moe_research/m{i}.py").write_text("# frozen fixture\n")
        for name in ("scripts/_harness.py","scripts/run_state_calibration_study.py","configs/state_calibration_study.yaml",*plan.ANALYSIS_FILES):
            path=self.repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("# fixture source\n")
        frozen={str(p.relative_to(self.repo)):plan.sha(p) for p in self.repo.glob("moe_research/*.py")}
        frozen.update({name:plan.sha(self.repo/name) for name in ("scripts/_harness.py","scripts/run_state_calibration_study.py","configs/state_calibration_study.yaml")})
        (self.run/"source_hashes.json").write_text(json.dumps(frozen))
        self.protocol={"source_hash":plan.canonical(frozen),"parser_version":"parser", "config":{
            "model_revision":"model","dataset_revision":"dataset","quality_constraint_for_internal_validation":"Original NI-only implementation text"}}
        self.protocol["protocol_hash"]=plan.canonical(self.protocol)
        (self.run/"protocol.json").write_text(json.dumps(self.protocol))
        self.reports=self.root/"reports"

    def draft(self):
        value=plan.make_draft(self.run,self.repo)
        archive=plan.publish_analysis_sources(self.reports,value["source"],self.repo)
        return plan.seal({**{k:v for k,v in value.items() if k!="plan_hash"},"analysis_source_archive":archive})

    def selection(self):
        calibrations={}
        for arm,family in (("M_N","scalar"),("L_N","lda")):
            value={"family":family,"metadata":{"protocol_hash":self.protocol["protocol_hash"]},"parameters":{"0":{}}}
            value["calibration_hash"]=plan.canonical(value);calibrations[arm]=value["calibration_hash"]
            path=self.run/f"formal/calibration/{arm}.json";path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        for arm in plan.BASELINES:
            path=self.run/f"formal/select/generations/{arm}.jsonl";path.parent.mkdir(parents=True,exist_ok=True)
            rows=[]
            for sid in plan.SELECT_IDS:
                n={"N":12,"F":8,"R8":9,"R16":10,"M_N":11,"L_N":10}[arm]
                correct=sid<256+({"N":256,"F":200,"R8":200,"R16":200,"M_N":210,"L_N":210}[arm])
                question=f"question {sid}"
                cal_hash=calibrations.get(arm)
                arm_hash=plan.canonical({"arm":arm,"k_compute":8 if arm=="N" else 4,
                    "k_reference":{"N":8,"F":4,"R8":8,"R16":16,"M_N":8,"L_N":4}[arm],
                    "calibration_hash":cal_hash,"prefill":"native","seed":20261019})
                rows.append({"sample_id":sid,"question":question,"question_hash":__import__('hashlib').sha256(question.encode()).hexdigest(),
                    "arm":arm,"scope":"formal/select","protocol_hash":self.protocol["protocol_hash"],"parser_version":"parser",
                    "model_revision":"model","dataset_revision":"dataset","max_new_tokens":2048,
                    "rendered_prompt":question,"prompt_token_ids":[1,sid],"strict_correct":correct,"tolerant_correct":correct,
                    "gen_token_ids":[10]*(n-1)+[99],"generated_tokens":n,"eos_ids":[99],"stop_reason":"eos",
                    "observer_enabled":False,"timing_debug_logits":False,"arm_hash":arm_hash,"calibration_hash":cal_hash})
            path.write_text("".join(json.dumps(row)+"\n" for row in rows))

    def final(self):
        draft=self.draft();self.selection();value=plan.prepare_plan(self.run,draft,self.repo)
        archive=plan.publish_analysis_sources(self.reports,value["source"],self.repo)
        return plan.seal({**{k:v for k,v in value.items() if k!="plan_hash"},"analysis_source_archive":archive})

    def test_incomplete_p2_is_pending_and_no_final_is_written(self):
        draft=self.draft()
        with self.assertRaises(plan.Pending):plan.prepare_plan(self.run,draft,self.repo)
        self.assertFalse((self.reports/"quality_plan.json").exists())

    def test_draft_is_not_a_frozen_quality_plan(self):
        draft=self.draft();self.assertEqual(draft["status"],"DRAFT_NOT_FROZEN")
        self.assertTrue(draft["analysis_supplement"]["not_an_original_protocol_condition"])
        self.assertFalse(draft["analysis_supplement"]["original_NI_only_condition_satisfied"])

    def test_full_p2_binds_actual_reference_and_strongest(self):
        value=self.final();proof=plan.verify_plan(self.run,value,self.repo)
        self.assertEqual(proof["selected_reference"],"F")
        self.assertEqual(proof["strongest_existing_baseline"],"L_N")
        self.assertEqual(value["selection"]["arms"]["N"]["sample_ids"],list(range(256,512)))
        self.assertIsNone(value["original_noninferiority_margin"])

    def test_any_validation_file_refuses_new_freeze(self):
        draft=self.draft();self.selection();path=self.run/"formal/validation_state/diagnostics/one.empty.json"
        path.parent.mkdir(parents=True);path.write_text("{}")
        with self.assertRaisesRegex(plan.Invalid,"Validation evidence"):plan.prepare_plan(self.run,draft,self.repo)

    def test_validation_file_older_than_final_timestamp_is_rejected(self):
        value=self.final();path=self.run/"formal/validate/generations/M_ST.jsonl"
        path.parent.mkdir(parents=True);path.write_text("{}\n")
        os.utime(path,ns=(0,0))
        with self.assertRaisesRegex(plan.Invalid,"mtime is earlier"):plan.verify_plan(self.run,value,self.repo)

    def test_draft_cannot_be_created_after_selection_output(self):
        self.selection()
        with self.assertRaisesRegex(plan.Invalid,"no late draft"):plan.make_draft(self.run,self.repo)

    def test_changed_selection_outcome_breaks_the_final_binding(self):
        value=self.final();path=self.run/"formal/select/generations/F.jsonl"
        with path.open() as handle:rows=[json.loads(line) for line in handle]
        rows[0]["strict_correct"]=False
        path.write_text("".join(json.dumps(row)+"\n" for row in rows))
        with self.assertRaisesRegex(plan.Invalid,"D_select"):plan.verify_plan(self.run,value,self.repo)

    def test_actual_source_archive_is_required_and_immutable(self):
        value=self.final();directory=Path(value["analysis_source_archive"]["directory"])
        source=directory/plan.ANALYSIS_FILES[0];self.assertEqual(source.read_bytes(),(self.repo/plan.ANALYSIS_FILES[0]).read_bytes())
        source.write_text("changed fixture archive")
        with self.assertRaisesRegex(plan.Invalid,"Archived"):plan.verify_plan(self.run,value,self.repo)

    def test_plan_and_source_publication_never_overwrite(self):
        value=self.final();path=self.reports/"quality_plan.json";plan.publish_once(path,value)
        with self.assertRaises(FileExistsError):plan.publish_once(path,value)
        self.assertEqual(plan.load(path)["plan_hash"],value["plan_hash"])


if __name__=="__main__":unittest.main()
