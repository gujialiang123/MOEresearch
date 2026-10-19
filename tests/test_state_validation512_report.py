"""Small CPU acceptance regressions; no real model, bootstrap, or plan seal."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import state_study_validation512_report as report512
from scripts import state_study_validation512 as collection512
from scripts import summarize_state_study_results as legacy_report
from scripts.freeze_state_study_quality_plan import canonical
from tests.test_state_positive_evidence import fixture_report,interval


def summary512():
    comparison={"status":"ok","prompt_count":512,"left":"M_ST","right":"M_NT","contrast":"left_minus_right",
        "same_calibration_required":False,"same_mask_required":False,"metrics":{"E_moment":interval(-.02,-.03,-.01)}}
    return {"family":"lda","chosen_mismatch_metric":"E_moment","expected_prompt_ids":sorted(str(i) for i in report512.IDS),
        "prompt_completeness":{a:{"complete_expected_ids":True} for a in report512.ARMS},
        "bootstrap":{"unit":"prompt","paired":True,"resamples":10000,"seed":20261019,"confidence_level":.95,
            "calibration_parameters_refitted":False,"includes_empty_and_inactive_prompt_ids":True},
        "analyses":{"all":{w:{"comparisons":{report512.OWN:copy.deepcopy(comparison)}} for w in ("token","prompt")}}}


class FullEndpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir="/scratch/jgu7/tmp");self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.plan=self.root/"quality.json"
        self.plan.write_text(json.dumps({"plan_hash":"original_quality_plan","protocol_hash":"protocol","selection":{"strongest_existing_baseline":"F","selected_reference":"F"}}))
        self.ev=type("Evidence",(),{"run":self.root,"errors":[]})()
        self.patches=[mock.patch.object(report512,"verify_plan",return_value={"status":"VERIFIED_PROSPECTIVE_PLAN"}),
            mock.patch.object(report512.legacy_gate,"calibration_provenance",return_value={"status":"VERIFIED"}),
            mock.patch.object(report512.legacy_gate,"generation_identities",return_value=True),
            mock.patch.object(report512.legacy_gate,"diagnostic_identities",return_value={"status":"VERIFIED"})]
        for p in self.patches:p.start();self.addCleanup(p.stop)
        self.state={"state_status":"VERIFIED","coverage_complete":True,"analysis_complete":True,"own_state_improves":True}

    def evaluate(self,base,state=None):
        base.setdefault("manifest",{})
        return report512.evaluate_validation512_result(base,{},self.ev,state or self.state,self.plan,self.root)

    def test_full512_pass_is_not_vetoed_by_inconclusive_or_adverse_old128_CI(self):
        for old_metric in (interval(.01,-.02,.04),{"status":"UNDEFINED","estimate":None,"ci_low":None,"ci_high":None}):
            with self.subTest(old_metric=old_metric):
                base=fixture_report();base["diagnostics"]["formal/validation_state"]["summary"]["analyses"]["all"]["token"]["comparisons"][report512.OWN]["metrics"]["E_moment"]=old_metric
                with mock.patch.object(report512.legacy_gate,"downstream_statistics",return_value={"passed":True}) as stats:
                    result=self.evaluate(base)
                self.assertEqual(result["status"],"POSITIVE_EXPLORATORY");self.assertFalse(result["p6_authorized"])
                self.assertEqual(stats.call_args.args[4]["plan_hash"],"original_quality_plan")

    def test_full512_missing_incomplete_or_nonimproving_never_runs_new_downstream(self):
        for key,value in (("state_status","PENDING"),("coverage_complete",False),("analysis_complete",False),("own_state_improves",False)):
            state={**self.state,key:value}
            with mock.patch.object(report512.legacy_gate,"downstream_statistics") as stats:
                result=self.evaluate(fixture_report(),state)
            self.assertFalse(result["passed"]);stats.assert_not_called()

    def test_other_original_prerequisites_still_veto(self):
        for kind in ("P0","select","P3","Cdiag","generation","calibration","legacy128"):
            with self.subTest(kind=kind):
                base=fixture_report()
                if kind=="P0":base["preflight"]["status"]="FAILED"
                if kind=="select":del base["generations"]["formal/select"]["arms"]["M_NT"]
                if kind=="P3":base["diagnostics"]["formal"]["summary"]["analyses"]["all"]["token"]["comparisons"]["C_minus_D"]["metrics"]["E_moment"]=interval(.02,-.01,.03)
                if kind=="Cdiag":base["diagnostics"]["formal"]["summary"]["analyses"]["all"]["token"]["comparisons"][report512.OWN]["metrics"]["E_moment"]=interval(.02,-.01,.03)
                generation=kind!="generation";calibration="UNVERIFIED" if kind=="calibration" else "VERIFIED"
                def identities(ev,diag,raw,cal,scope,paths):return {"status":"UNVERIFIED" if kind=="legacy128" and scope.endswith("validation_state") else "VERIFIED"}
                with mock.patch.object(report512.legacy_gate,"generation_identities",return_value=generation),mock.patch.object(report512.legacy_gate,"calibration_provenance",return_value={"status":calibration}),mock.patch.object(report512.legacy_gate,"diagnostic_identities",side_effect=identities),mock.patch.object(report512.legacy_gate,"downstream_statistics") as stats:
                    self.assertFalse(self.evaluate(base)["passed"])
                stats.assert_not_called()

    def test_invalid_full512_identity_blocks_even_if_booleans_claim_improvement(self):
        state={**self.state,"state_status":"BLOCKED","errors":["forged commit"]}
        with mock.patch.object(report512.legacy_gate,"downstream_statistics") as stats:
            self.assertEqual(self.evaluate(fixture_report(),state)["status"],"BLOCKED")
        stats.assert_not_called()

    def test_complete_is_separate_from_quality_cost_pass(self):
        with mock.patch.object(report512.legacy_gate,"downstream_statistics",return_value={"passed":False}):
            result=self.evaluate(fixture_report())
        self.assertEqual(result["status"],"INCONCLUSIVE");self.assertFalse(result["passed"])


class AnalysisContractTests(unittest.TestCase):
    def test_full512_token_primary_prompt_sensitivity_can_disagree(self):
        summary=summary512();summary["analyses"]["all"]["prompt"]["comparisons"][report512.OWN]["metrics"]["E_moment"]=interval(.01,-.01,.03)
        value=report512.full512_analysis_contract({"summary":summary})
        self.assertTrue(value["analysis_complete"]);self.assertTrue(value["own_state_improves"])

    def test_forged_512_label_does_not_replace_expected_IDs_or_pair_count(self):
        for key in ("ids","pairs","bootstrap"):
            summary=summary512()
            if key=="ids":summary["expected_prompt_ids"]=sorted(str(i) for i in range(512,640))
            if key=="pairs":summary["analyses"]["all"]["token"]["comparisons"][report512.OWN]["prompt_count"]=128
            if key=="bootstrap":summary["bootstrap"]["resamples"]=1000
            self.assertFalse(report512.full512_analysis_contract({"summary":summary})["analysis_complete"])

    def test_undefined_saved_analysis_is_complete_but_not_positive(self):
        summary=summary512()
        for weight in ("token","prompt"):
            summary["analyses"]["all"][weight]["comparisons"][report512.OWN]["metrics"]["E_moment"]={"status":"undefined_bootstrap_draw","estimate":None,"ci_low":None,"ci_high":None,"finite_draws":9999}
        value=report512.full512_analysis_contract({"summary":summary})
        self.assertTrue(value["analysis_complete"]);self.assertFalse(value["analysis_estimable"]);self.assertFalse(value["own_state_improves"])

    def test_all_empty_has_honest_complete_analysis_without_state_evidence(self):
        summary=summary512();summary["analyses"]["all"]={"status":"NO_OBSERVATIONS","reason":"no_valid_decode_positions_in_predeclared_group"}
        summary["coverage"]={"all":{a:{"zero_observation_prompt_ids":sorted(str(i) for i in report512.IDS)} for a in report512.ARMS}}
        value=report512.full512_analysis_contract({"summary":summary})
        self.assertTrue(value["analysis_complete"]);self.assertFalse(value["own_state_improves"])

    def test_placeholder_metric_or_unknown_undefined_reason_is_not_completed_analysis(self):
        for metric in ({},{"status":"undefined_bootstrap_draw","ci_low":None,"ci_high":None},
                       {"status":"UNDEFINED","estimate":None,"ci_low":None,"ci_high":None}):
            summary=summary512();summary["analyses"]["all"]["token"]["comparisons"][report512.OWN]["metrics"]["E_moment"]=metric
            self.assertFalse(report512.full512_analysis_contract({"summary":summary})["analysis_complete"])


class ArchiveIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir="/scratch/jgu7/tmp");self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.directory=self.root/"diagnostics";self.directory.mkdir()
        self.ev=legacy_report.Evidence(self.root,True)
        self.cal={a:{"calibration_hash":"cal"+a,"numerical_protection":{"eps_lda_std":1e-6}} for a in report512.ARMS}
        self.diag={"archive_hashes_verified":True,"current_inventory_matches":True}

    def marker(self,sid,arm,trace):
        marker={"schema_version":"state_collection.empty.v1","collection_schema_version":"state_collection.v2","count":0,"dtype":"float64","ddof":0,"prompt_id":str(sid),"path":arm,
            "reason":"no_valid_decode_observations","numerical_protection":self.cal[arm]["numerical_protection"],
            "position_bins":[[1,32],[33,128],[129,256],[257,512],[513,1024],[1025,2047]],
            "metadata":{"protocol_hash":"protocol","driver_hash":report512.legacy_gate.arm_hash(arm,self.cal),"calibration_hash":self.cal[arm]["calibration_hash"],"trace_hash":canonical(trace)}}
        path=self.directory/(canonical([str(sid),arm])+".empty.json");path.write_text(json.dumps(marker));return path

    def test_only128_state_markers_cannot_be_called_full512(self):
        rows={a:{sid:{"gen_token_ids":[99]} for sid in report512.IDS} for a in report512.ARMS}
        for arm in report512.ARMS:
            for sid in range(512,640):self.marker(sid,arm,[99])
        value=report512.verify_full512_archives(self.ev,self.diag,rows,self.cal,"protocol")
        self.assertEqual(value["status"],"UNVERIFIED");self.assertEqual(value["verified_prompt_paths"],256)

    def test_all512_empty_markers_preserve_complete_universe_and_trace_identity(self):
        rows={a:{sid:{"gen_token_ids":[99]} for sid in report512.IDS} for a in report512.ARMS}
        for arm in report512.ARMS:
            for sid in report512.IDS:self.marker(sid,arm,[99])
        value=report512.verify_full512_archives(self.ev,self.diag,rows,self.cal,"protocol")
        self.assertEqual(value["status"],"VERIFIED");self.assertEqual(value["verified_prompt_paths"],1024)
        path=self.directory/(canonical(["512","M_ST"])+".empty.json")
        marker=json.loads(path.read_text());marker["metadata"]["trace_hash"]="forged";path.write_text(json.dumps(marker))
        self.assertEqual(report512.verify_full512_archives(self.ev,self.diag,rows,self.cal,"protocol")["status"],"UNVERIFIED")

    def test_cpu_toy_npz_verifies48_layers_T_minus_1_and_own_calibrator(self):
        # Only this low-level archive fixture uses one prompt. The public
        # checker has no ID override; other tests retain all512 expected IDs.
        rows={a:{512:{"gen_token_ids":[10,99]}} for a in report512.ARMS}
        paths={}
        for arm in report512.ARMS:
            arrays={};entries=[]
            for layer in range(48):
                for group in ("all","1-32"):
                    prefix="r"+str(len(entries));arrays[prefix+"_positions"]=np.asarray([[1,10]],dtype=np.int64)
                    entries.append({"key":["512",arm,layer,group],"prefix":prefix,"count":1,"mask_hash":canonical([[1,10]]),
                        "driver_hash":report512.legacy_gate.arm_hash(arm,self.cal),"calibration_hash":self.cal[arm]["calibration_hash"]})
            meta={"schema_version":"state_collection.v2","dtype":"float64","ddof":0,"moment_representation":"count_mean_centered_m2",
                "position_bins":[[1,32],[33,128],[129,256],[257,512],[513,1024],[1025,2047]],"numerical_protection":self.cal[arm]["numerical_protection"],"records":entries}
            archive=self.directory/(canonical(["512",arm])+".npz");np.savez_compressed(archive,metadata_json=np.asarray(json.dumps(meta)),**arrays)
            tokens=archive.with_suffix(".tokens.jsonl");tokens.write_text("{}\n"*48)
            self.ev.record(tokens)
            commit={"schema_version":"state_collection.commit.v1","collection_schema_version":"state_collection.v2","archive":archive.name,"tokens":tokens.name,
                "archive_sha256":report512.sha(archive),"tokens_sha256":report512.sha(tokens),"token_rows":48}
            archive.with_suffix(".complete.json").write_text(json.dumps(commit));paths[arm]=(archive,meta,arrays)
        with mock.patch.object(report512,"IDS",(512,)):
            self.assertEqual(report512.verify_full512_archives(self.ev,self.diag,rows,self.cal,"protocol")["status"],"VERIFIED")
            archive,meta,arrays=paths["M_ST"];meta["records"][0]["calibration_hash"]=None
            np.savez_compressed(archive,metadata_json=np.asarray(json.dumps(meta)),**arrays)
            commit=json.loads(archive.with_suffix(".complete.json").read_text());commit["archive_sha256"]=report512.sha(archive)
            archive.with_suffix(".complete.json").write_text(json.dumps(commit))
            self.assertEqual(report512.verify_full512_archives(self.ev,self.diag,rows,self.cal,"protocol")["status"],"UNVERIFIED")


class ReadOnlyChecks(unittest.TestCase):
    def test_missing_plan_is_pending_and_cannot_be_a_terminal_or_complete_proof(self):
        with tempfile.TemporaryDirectory(dir="/scratch/jgu7/tmp") as directory:
            value=report512.check_validation512(directory,Path(directory)/"absent.json")
        self.assertEqual(value["state_status"],"PLAN_PENDING");self.assertFalse(value["complete"]);self.assertFalse(value["not_required"])


class ReceiptIntegrationTests(unittest.TestCase):
    def test_small_CPU_receipt_chain_accepts_undefined_work_and_rejects_forged_summary_hash(self):
        # Fixed public production IDs remain512. This two-prompt fixture tests
        # cross-module receipt interfaces without thousands of model forwards.
        ids=(512,513)
        with tempfile.TemporaryDirectory(dir="/scratch/jgu7/tmp") as directory:
            root=Path(directory);run=root/"run";run.mkdir();out=root/"out";out.mkdir();data=out/"diagnostics";data.mkdir()
            planfile=root/"plan.json";planfile.write_text("{}")
            plan={"plan_hash":"supplement","protocol_hash":"protocol","spec":{"sample_ids":list(ids),"arms":list(report512.ARMS)}}
            rows={a:{sid:{"sample_id":sid,"gen_token_ids":[99]} for sid in ids} for a in report512.ARMS}
            cal={a:{"calibration_hash":"cal"+a,"numerical_protection":collection512.EPS} for a in report512.ARMS}
            snapshot=collection512.sealed({"plan_hash":"supplement"},"snapshot_hash")
            (out/"source_snapshot.json").write_text(json.dumps(snapshot));checks=[];sources=[]
            for arm in report512.ARMS:
                for sid in ids:
                    trace=[99];stem=canonical([str(sid),arm]);identity=collection512.trace_identity(sid,arm,rows[arm][sid],cal,"protocol")
                    marker={"schema_version":"state_collection.empty.v1","collection_schema_version":"state_collection.v2","prompt_id":str(sid),"path":arm,
                        "count":0,"reason":"no_valid_decode_observations","dtype":"float64","ddof":0,"position_bins":collection512.SPEC["position_bins"],"numerical_protection":collection512.EPS,"metadata":identity}
                    path=data/(stem+".empty.json");path.write_text(json.dumps(marker));sources.append({"path":str(path),"size_bytes":path.stat().st_size})
                    check={"sample_id":sid,"path":arm,"argmaxes":trace,"decode_forwards":0,"mask_hash":canonical([]),"logits_finiteness_check":"all returned logits checked",**identity}
                    per=out/"replay_checks"/(stem+".json");per.parent.mkdir(exist_ok=True);per.write_text(json.dumps(check));checks.append(check)
            (out/"replay_checks.jsonl").write_text("".join(json.dumps(x)+"\n" for x in checks))
            with mock.patch.object(report512,"IDS",ids),mock.patch.object(collection512,"IDS",ids):
                summary=summary512();summary["source_archives"]=sources;summary["analyses"]["all"]={"status":"NO_OBSERVATIONS","reason":"no_valid_decode_positions_in_predeclared_group"}
                summary["coverage"]={"all":{a:{"counts_per_prompt_layer":{},"mask_hashes_per_prompt_layer":{},"zero_observation_prompt_ids":[str(i) for i in ids]} for a in report512.ARMS}}
                saved=out/"summaries/diagnostic_summary.json";saved.parent.mkdir();saved.write_text(json.dumps(summary))
                raw=collection512.raw_receipt(out,plan,snapshot,rows,cal);(out/"raw_completion.json").write_text(json.dumps(raw))
                completion=collection512.sealed({"schema_version":"state_study_validation512.complete.v1","plan_hash":"supplement","source_snapshot_hash":snapshot["snapshot_hash"],
                    "raw_completion_hash":raw["raw_completion_hash"],"summary_sha256":report512.sha(saved),"expected_prompt_ids":list(ids),"arms":list(report512.ARMS),"completed_work":True},"completion_hash")
                (out/"completion.json").write_text(json.dumps(completion))
                base=fixture_report();base["manifest"]={"stages":{"P4":{"status":"complete"}}}
                context=(base,{},legacy_report.Evidence(run,True))
                with mock.patch.object(report512,"_supplement_module",return_value=collection512),mock.patch.object(collection512,"load_plan",return_value=plan),mock.patch.object(collection512,"output_dir",return_value=out),mock.patch.object(collection512,"source_snapshot",return_value=(snapshot,rows,cal)):
                    proof=report512.check_validation512(run,planfile,context=context)
                    self.assertTrue(proof["complete"],proof);self.assertFalse(proof["analysis_estimable"]);self.assertFalse(proof["positive_state_gate"])
                    self.assertEqual(proof["source_snapshot_hash"],snapshot["snapshot_hash"])
                    completion["summary_sha256"]="0"*64;completion=collection512.sealed({k:v for k,v in completion.items() if k!="completion_hash"},"completion_hash")
                    (out/"completion.json").write_text(json.dumps(completion))
                    bad=report512.check_validation512(run,planfile,context=context)
                    self.assertFalse(bad["complete"]);self.assertEqual(bad["state_status"],"BLOCKED")


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir="/scratch/jgu7/tmp");self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.run=self.root/"run";self.run.mkdir();self.out=self.root/"out";self.out.mkdir()
        for name in ("completion.json","raw_completion.json"):(self.out/name).write_text("{}")
        self.plan={"plan_hash":"supplement"};self.proof={"complete":True,"not_required":False,"plan_hash":"supplement","positive_state_gate":False,"source_snapshot_hash":"snapshot"}
        self.result={"schema_version":"state_study_validation512.report.v1","status":"INCONCLUSIVE","passed":False,"run_dir":str(self.run.resolve()),"generated_at_utc":"2026-10-19T12:00:00Z","validation512":self.proof}
        self.markdown="\n".join("## "+str(i)+". question" for i in range(1,10))+"\n"
        self.patches=[mock.patch.object(collection512,"load_plan",return_value=self.plan),mock.patch.object(collection512,"output_dir",return_value=self.out),
            mock.patch.object(report512,"_supplement_module",return_value=collection512),mock.patch.object(report512,"check_validation512",return_value=self.proof),
            mock.patch.object(report512,"_quality_identity",return_value={"path":"missing_quality_plan","exists":False,"sha256":None})]
        for p in self.patches:p.start();self.addCleanup(p.stop)

    def test_missing_delivery_is_not_done_then_write_verify_and_reuse_without_reanalysis(self):
        self.assertFalse(report512.verify_delivery(self.run)["delivery_complete"])
        with mock.patch.object(report512,"build_report",return_value=(self.result,self.markdown)) as build:
            created=report512.ensure_delivery(self.run);self.assertTrue(created["verified"],created)
            reused=report512.ensure_delivery(self.run);self.assertTrue(reused["reused"]);build.assert_called_once()
        self.assertEqual(created["scientific_status"],"INCONCLUSIVE")
        report=self.out/"delivery/RUN_REPORT.md";report.write_text("forged")
        self.assertFalse(report512.verify_delivery(self.run)["verified"])

    def test_delivery_IO_failure_keeps_raw_analysis_and_never_claims_done(self):
        with mock.patch.object(report512,"build_report",return_value=(self.result,self.markdown)),mock.patch.object(report512,"_atomic_text",side_effect=OSError("fixture disk unavailable")):
            value=report512.ensure_delivery(self.run)
        self.assertFalse(value["delivery_complete"]);self.assertEqual(value["status"],"PENDING")
        self.assertTrue((self.out/"completion.json").is_file());self.assertFalse((self.out/"delivery/delivery_receipt.json").exists())

    def test_NOT_REQUIRED_can_deliver_nine_questions_without_claiming_full512(self):
        self.proof.update(complete=False,not_required=True,state_status="NOT_REQUIRED")
        (self.run/"manifest.json").write_text(json.dumps({"stages":{"P5":{"status":"complete"}}}));(self.run/"RUN_REPORT.md").write_text("original core report")
        for name in ("formal/summaries/diagnostic_summary.json","formal/validation_state/summaries/diagnostic_summary.json","validation_freeze.json"):
            path=self.run/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("{}")
        with mock.patch.object(report512,"build_report",return_value=(self.result,self.markdown)):
            value=report512.ensure_delivery(self.run)
        self.assertTrue(value["delivery_complete"],value)
        receipt=json.loads((self.out/"delivery/delivery_receipt.json").read_text())
        self.assertEqual(receipt["work"]["kind"],"NOT_REQUIRED");self.assertFalse(self.result["validation512"]["complete"])


if __name__=="__main__":unittest.main()
