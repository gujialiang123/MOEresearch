"""Focused CPU evidence-chain regressions, with no real-result bootstrap."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import state_study_positive_evidence as evidence_gate
from scripts import state_study_positive_gate as statistics_gate
from scripts import summarize_state_study_results as reporting


def interval(estimate,lo,hi):
    return {"status":"ok","estimate":estimate,"ci_low":lo,"ci_high":hi}


def fixture_report():
    config={"native_match_tolerance":.05,"state_effect_tolerance":.01}
    own={"status":"ok","prompt_count":128,"same_calibration_required":False,"same_mask_required":False,
         "matched_masks":False,"metrics":{"E_moment":interval(-.02,-.03,-.01)}}
    formal={"status":"STATE_MISMATCH_SUPPORTED","gate_recalibration":True,"family":"lda","chosen_mismatch_metric":"E_moment",
        "decision":{k:True for k in ("native_matching_established","student_deployment_mismatch_established","state_effect_exceeds_practical_bound")},
        "decision_protocol":config,"expected_prompt_ids":[str(i) for i in range(128,256)],
        "prompt_completeness":{p:{"complete_expected_ids":True} for p in ("A","B","C","D","M_NT","M_ST")},
        "analyses":{"all":{"token":{"paths":{p:{"intervals":{"E_moment":value}} for p,value in {
            "A":interval(.01,.005,.02),"C":interval(.2,.1,.3),"D":interval(.03,.02,.04)}.items()},
            "comparisons":{"C_minus_D":{"status":"ok","metrics":{"E_moment":interval(.17,.06,.2)}},
                "B_minus_A":{"status":"ok","metrics":{"E_moment":interval(.001,-.002,.004)}},"M_ST_minus_M_NT_own_trajectories":own}}}}}
    validation={"status":"INCONCLUSIVE","reason":"required_A_B_C_D_observations_missing",
        "expected_prompt_ids":[str(i) for i in range(512,640)],
        "prompt_completeness":{p:{"complete_expected_ids":True} for p in ("M_NT","M_ST")},
        "analyses":{"all":{"token":{"comparisons":{"M_ST_minus_M_NT_own_trajectories":copy.deepcopy(own)}}}}}
    def pack(summary,paths):
        summary["bootstrap"]={"unit":"prompt","paired":True,"resamples":10000,"seed":20261019,
            "confidence_level":.95,"calibration_parameters_refitted":False,"includes_empty_and_inactive_prompt_ids":True}
        return {"summary":summary,"archive_hashes_verified":True,"current_inventory_matches":True,
            "full_trace_masks_reconciled":{p:True for p in paths}}
    freeze={"protocol_hash":"protocol","selected_reference":"F","strongest_existing_baseline":"F","initial_calibration":"L_N",
        "selection_sample_ids":list(range(256,512)),"validation_sample_ids":list(range(512,1024)),"minimum_useful_relative_token_reduction":.05,
        "quality_noninferiority_margin":None,"updated":True,"max_method_updates":1,"followup_trigger":.01,
        "followup_prompt_ids":list(range(256,384)),"followup_cap":4096,"finalist_arms":["N","F","L_N","M_NT","M_ST"]}
    groups={"formal/select":{"arms":{a:{"complete":True,"cap_rate":0.} for a in evidence_gate.METHODS}},
        "formal/validate":{"arms":{a:{"complete":True} for a in freeze["finalist_arms"]}}}
    return {"protocol":{"protocol_hash":"protocol","config":config},"preflight":{"status":"PASSED","protocol_hash":"protocol"},
        "calibration":{"arms":{}},"validation_freeze":freeze,"generations":groups,
        "diagnostics":{"formal":pack(formal,("A","B","C","D","M_NT","M_ST")),
            "formal/validation_state":pack(validation,("M_NT","M_ST"))}}


class PositiveChainTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir="/scratch/jgu7/tmp");self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.plan=self.root/"quality_plan.json"
        self.plan.write_text(json.dumps({"plan_hash":"plan","protocol_hash":"protocol","analysis_supplement":{"type":"prospective correction"},
            "selection":{"strongest_existing_baseline":"F","selected_reference":"F"}}))
        self.evidence=type("Evidence",(),{"run":self.root,"errors":[]})()
        self.patches=[mock.patch.object(evidence_gate,"verify_plan",return_value={"status":"VERIFIED_PROSPECTIVE_PLAN"}),
            mock.patch.object(evidence_gate,"calibration_provenance",return_value={"status":"VERIFIED"}),
            mock.patch.object(evidence_gate,"generation_identities",return_value=True),
            mock.patch.object(evidence_gate,"diagnostic_identities",return_value={"status":"VERIFIED"})]
        for p in self.patches:p.start();self.addCleanup(p.stop)

    def test_complete_positive_branch_and_none_ni_are_not_permanently_disabled(self):
        report=fixture_report()
        with mock.patch.object(evidence_gate,"downstream_statistics",return_value={"passed":True}) as downstream:
            result=evidence_gate.evaluate_positive_evidence(report,{},self.evidence,self.plan)
        self.assertEqual(result["status"],"POSITIVE_EXPLORATORY");self.assertTrue(result["passed"])
        self.assertFalse(result["p6_authorized"]);downstream.assert_called_once()

    def test_validate_overall_inconclusive_does_not_override_valid_own_comparison(self):
        report=fixture_report();self.assertEqual(report["diagnostics"]["formal/validation_state"]["summary"]["status"],"INCONCLUSIVE")
        with mock.patch.object(evidence_gate,"downstream_statistics",return_value={"passed":True}):
            self.assertTrue(evidence_gate.evaluate_positive_evidence(report,{},self.evidence,self.plan)["passed"])

    def test_status_boolean_alone_cannot_replace_p3_numeric_bound(self):
        report=fixture_report();report["diagnostics"]["formal"]["summary"]["analyses"]["all"]["token"]["comparisons"]["C_minus_D"]["metrics"]["E_moment"]=interval(.02,-.01,.03)
        with mock.patch.object(evidence_gate,"downstream_statistics") as downstream:
            result=evidence_gate.evaluate_positive_evidence(report,{},self.evidence,self.plan)
        self.assertFalse(result["passed"]);downstream.assert_not_called()

    def test_six_base_complete_without_updated_select_pairs_cannot_pass(self):
        report=fixture_report();del report["generations"]["formal/select"]["arms"]["M_NT"]
        with mock.patch.object(evidence_gate,"downstream_statistics") as downstream:
            self.assertFalse(evidence_gate.evaluate_positive_evidence(report,{},self.evidence,self.plan)["passed"])
        downstream.assert_not_called()

    def test_missing_or_undefined_B_minus_A_interval_withholds_positive(self):
        for value in ({"status":"ok"},{"status":"ok","metrics":{"E_moment":interval(0,None,None)}},
                      {"status":"ok","metrics":{"E_moment":interval(0,-.01,float("inf"))}}):
            with self.subTest(value=value):
                report=fixture_report()
                report["diagnostics"]["formal"]["summary"]["analyses"]["all"]["token"]["comparisons"]["B_minus_A"]=value
                with mock.patch.object(evidence_gate,"downstream_statistics") as downstream:
                    self.assertFalse(evidence_gate.evaluate_positive_evidence(report,{},self.evidence,self.plan)["passed"])
                downstream.assert_not_called()

    def test_missing_bottom_hash_or_own_improvement_withholds_positive(self):
        for kind in ("hash","own"):
            report=fixture_report()
            if kind=="hash":report["diagnostics"]["formal"]["archive_hashes_verified"]=False
            else:report["diagnostics"]["formal/validation_state"]["summary"]["analyses"]["all"]["token"]["comparisons"]["M_ST_minus_M_NT_own_trajectories"]["metrics"]["E_moment"]=interval(-.01,-.03,.01)
            with mock.patch.object(evidence_gate,"downstream_statistics") as downstream:
                self.assertFalse(evidence_gate.evaluate_positive_evidence(report,{},self.evidence,self.plan)["passed"])
            downstream.assert_not_called()

    def test_absent_plan_withholds_positive_without_new_statistics(self):
        with mock.patch.object(evidence_gate,"downstream_statistics") as downstream:
            result=evidence_gate.evaluate_positive_evidence(fixture_report(),{},self.evidence,self.root/"missing.json")
        self.assertEqual(result["quality_plan_status"],"PENDING_NOT_FROZEN");downstream.assert_not_called()

    def test_ratio_5pct_condition_and_cache_hit(self):
        ids=[1,2];raw={"formal/validate":{a:[{"sample_id":i,"strict_correct":True,"tolerant_correct":True,"generated_tokens":n} for i in ids]
            for a,n in (("M_ST",90),("M_NT",100),("F",100))}}
        quality={"status":"SUPERIORITY_ESTABLISHED","exact_lower_bound":.01}
        cost={"status":"OK","lower_bound":.049,"upper_bound":.2}
        with mock.patch.object(statistics_gate,"quality_superiority",return_value=quality),mock.patch.object(statistics_gate,"paired_relative_token_reduction",return_value=cost):
            result=evidence_gate.downstream_statistics(raw,"formal/validate",("M_NT","F"),ids,{"plan_hash":"p"},self.root)
        self.assertFalse(result["passed"])
        with mock.patch.object(statistics_gate,"quality_superiority",side_effect=AssertionError("cached result must not recompute")),mock.patch.object(statistics_gate,"paired_relative_token_reduction",side_effect=AssertionError("must not bootstrap twice")):
            reused=evidence_gate.downstream_statistics(raw,"formal/validate",("M_NT","F"),ids,{"plan_hash":"p"},self.root)
        self.assertTrue(reused["cache_reused"])


class ArchiveIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir="/scratch/jgu7/tmp");self.addCleanup(self.temp.cleanup)
        self.run=Path(self.temp.name);self.directory=self.run/"formal/diagnostics";self.directory.mkdir(parents=True)
        self.evidence=reporting.Evidence(self.run,True);self.cal={"L_N":{"numerical_protection":{"eps_lda_std":1e-6}}}
        self.target={"A":{"rows":{128:{"gen_token_ids":[10,99],"protocol_hash":"p"}},"driver_hash":"native-driver","calibration_hash":"observed-L_N"}}
        self.diagnostic={"archive_hashes_verified":True,"current_inventory_matches":True}

    def archive(self,wrong_calibration=False):
        stem=evidence_gate.canonical(["128","A"]);arrays={};records=[]
        for layer in range(48):
            prefix=f"r{layer}";records.append({"key":["128","A",layer,"all"],"prefix":prefix,"count":1,
                "mask_hash":evidence_gate.canonical([[1,10]]),"driver_hash":"native-driver",
                "calibration_hash":None if wrong_calibration else "observed-L_N"})
            arrays[prefix+"_positions"]=np.asarray([[1,10]],dtype=np.int64)
        metadata={"schema_version":"state_collection.v2","dtype":"float64","ddof":0,
            "moment_representation":"count_mean_centered_m2","numerical_protection":{"eps_lda_std":1e-6},"records":records}
        arrays["metadata_json"]=np.asarray(json.dumps(metadata));np.savez_compressed(self.directory/(stem+".npz"),**arrays)
        tokens=self.directory/(stem+".tokens.jsonl");tokens.write_bytes(b'{}\n'*48)
        (self.directory/(stem+".complete.json")).write_text(json.dumps({"tokens":tokens.name,"token_rows":48}))
        self.evidence.inventory[str(tokens.relative_to(self.run))]={"physical_lines":48}

    def verify(self):
        with mock.patch.object(evidence_gate,"expected_diagnostics",return_value=self.target):
            return evidence_gate.diagnostic_identities(self.evidence,self.diagnostic,{},self.cal,"formal",("A",))

    def test_native_driver_still_observes_the_LN_calibrator_hash(self):
        self.archive();self.assertEqual(self.verify()["status"],"VERIFIED")

    def test_native_generation_none_calibration_is_wrong_for_observed_A_path(self):
        self.archive(True);self.assertEqual(self.verify()["status"],"UNVERIFIED")

    def test_immediate_eos_marker_binds_trace_and_zero_observation(self):
        self.target["A"]["rows"][128]["gen_token_ids"]=[99]
        stem=evidence_gate.canonical(["128","A"])
        marker={"schema_version":"state_collection.empty.v1","collection_schema_version":"state_collection.v2",
            "prompt_id":"128","path":"A","count":0,"dtype":"float64","ddof":0,"reason":"no_decode",
            "metadata":{"protocol_hash":"p","driver_hash":"native-driver","calibration_hash":"observed-L_N","trace_hash":evidence_gate.canonical([99])}}
        (self.directory/(stem+".empty.json")).write_text(json.dumps(marker))
        self.assertEqual(self.verify()["status"],"VERIFIED")


if __name__=="__main__":unittest.main()
