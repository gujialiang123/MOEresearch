"""Focused CPU-only report acceptance; no real plan, model or bootstrap."""
import copy
import json
import math
from pathlib import Path
import tempfile
import shutil
import unittest
from unittest import mock

from scripts import state_study_observer_cost as cost
from scripts import state_study_observer_cost_report as report


def put(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n")


def replay(trace):
    return {"argmaxes": list(trace), "decode_forwards": len(trace)-1,
            "mask_hash": cost.canonical([[j,t] for j,t in enumerate(trace[:-1],1)]),
            "logits_finiteness_check": report.FINITE_LOGIT_GUARD}


def mode(unit, name, archive=None, seconds=1.0):
    return cost.sealed({"schema_version": "state_observer_cost.mode.v1", "mode": name,
        "unit_hash": unit["unit_hash"], "plan_hash": "plan", "capture_logits": False,
        "replay": replay(unit["trace"]), "observer_calls": 48*(len(unit["trace"])-1) if name=="on" else 0,
        "timing": {"replay_seconds": seconds, "observer_callback_host_seconds": .1 if name=="on" else 0.,
                   "archive_seconds": .2 if name=="on" else 0.},
        "memory": {"baseline_allocated_bytes": 10, "baseline_reserved_bytes": 20,
                   "peak_allocated_bytes": 15, "peak_reserved_bytes": 25,
                   "incremental_peak_allocated_bytes": 5, "incremental_peak_reserved_bytes": 5},
        "archive_receipt": archive if name=="on" else None}, "mode_hash")


def reseal(value, field):
    value.pop(field, None); value[field] = cost.canonical(value); return value


class ScalarModeTests(unittest.TestCase):
    def fixture(self, scenario="A"):
        unit={"trace": [3,9], "unit_hash": "unit", "source_argmax_required": scenario not in ("B","D")}
        modes={m: mode(unit,m,{}, .8 if m=="on" else 1.) for m in ("on","off")}
        return unit, {"signed_differences": cost.signed_differences(modes["on"],modes["off"])}, modes

    def test_negative_delta_is_retained_not_a_completion_veto(self):
        unit,pair,modes=self.fixture()
        self.assertAlmostEqual(report.validate_modes(unit,pair,modes,"plan")["replay_seconds"],-.2)

    def test_undefined_negative_or_nonfinite_actual_modes_rejected(self):
        for bad in (None, -1., float("nan"), float("inf"), True):
            unit,pair,modes=self.fixture();modes["on"]["timing"]["replay_seconds"]=bad
            if not isinstance(bad,float) or math.isfinite(bad):reseal(modes["on"],"mode_hash")
            with self.subTest(bad=bad),self.assertRaises(report.Invalid):report.validate_modes(unit,pair,modes,"plan")

    def test_both_modes_fullmask_and_peak_baseline_required(self):
        for fault in ("missing", "mask", "calls", "memory", "plan", "argmax"):
            unit,pair,modes=self.fixture()
            if fault=="missing":del modes["off"]
            elif fault=="mask":modes["on"]["replay"]["mask_hash"]="forged"
            elif fault=="calls":modes["on"]["observer_calls"]=47
            elif fault=="memory":modes["on"]["memory"]["peak_allocated_bytes"]=9
            elif fault=="plan":modes["on"]["plan_hash"]="other"
            else:modes["on"]["replay"]["argmaxes"]=[4,9]
            if "on" in modes:reseal(modes["on"],"mode_hash")
            with self.subTest(fault=fault),self.assertRaises(report.Invalid):report.validate_modes(unit,pair,modes,"plan")

    def test_teacher_forced_B_D_may_differ_from_input_but_not_each_other(self):
        for scenario in ("B","D"):
            unit,pair,modes=self.fixture(scenario)
            for value in modes.values():value["replay"]["argmaxes"]=[5,6];reseal(value,"mode_hash")
            report.validate_modes(unit,pair,modes,"plan")
            modes["off"]["replay"]["argmaxes"]=[7,6];reseal(modes["off"],"mode_hash")
            with self.assertRaises(report.Invalid):report.validate_modes(unit,pair,modes,"plan")


class ActualReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir="/scratch/jgu7/tmp");self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.run=self.root/"run";self.run.mkdir();self.out=self.root/"observer";self.out.mkdir()
        self.plan={"plan_hash":"plan","protocol_hash":"protocol"}
        self.cals={"L_N":{"calibration_hash":"cal","numerical_protection":cost.base.EPS}}
        self.units=[]
        for index,scenario in enumerate(("A","B","C","D")):
            for sid in report.IDS:
                body={"sample_id":sid,"scenario":scenario,"scenario_index":index,
                    "driver_arm":"N" if scenario in ("A","D") else "L_N", "observed_calibration_arm":"L_N",
                    "trace_source_scope":"formal/diag_rollouts","trace_source_arm":"N" if scenario in ("A","B") else "L_N",
                    "trace":[9],"generation_row":{"gen_token_ids":[9]},"common_decode_limit":0,
                    "source_argmax_required":scenario in ("A","C"),
                    "order":["on","off"] if (sid+index)%2==0 else ["off","on"]}
                self.units.append(cost.sealed(body,"unit_hash"))
        self.snapshot=cost.sealed({"P4_triggered":False,"sample_ids":list(report.IDS),"scenarios":["A","B","C","D"]},"snapshot_hash")
        self.patches=[mock.patch.object(report,"_cost_module",return_value=cost),
            mock.patch.object(cost,"load_plan",return_value=self.plan),mock.patch.object(cost,"output_dir",return_value=self.out),
            mock.patch.object(cost,"source_snapshot",side_effect=lambda *a:(self.snapshot,self.units,self.cals))]
        for p in self.patches:p.start();self.addCleanup(p.stop)
        put(self.out/"source_snapshot.json",self.snapshot)
        for unit in self.units:self.write_pair(unit)
        put(self.out/"raw_completion.json",cost.raw_receipt(self.out,self.plan,self.snapshot,self.units,self.cals))
        cost.analyze(self.run,self.plan,self.out)

    def write_pair(self,unit,real_on=False):
        attempt=self.out/"attempts"/unit["unit_hash"]/"try1"
        put(attempt/"intent.json",{"plan_hash":"plan","source_snapshot_hash":self.snapshot["snapshot_hash"],
                                  "unit_hash":unit["unit_hash"],"order":unit["order"]})
        prefix=unit["trace"][:8]
        put(attempt/"warmup.json",{"excluded_from_measurement":True,"trace_length":len(prefix),"elapsed_seconds":.01,
                                   "replay":replay(prefix),"observer_calls":48*(len(prefix)-1)})
        if real_on:
            from tests.test_state_study_observer_cost import write_real_on_mode
            on,_,_=write_real_on_mode(attempt,{**unit,"sample":{"sample_id":unit["sample_id"]}},self.plan)
            modes={"on":on,"off":mode(unit,"off",seconds=1.)}
        else:
            identity=cost.identity(unit,self.cals,"protocol");stem=cost.canonical([str(unit["sample_id"]),unit["scenario"]])
            put(attempt/"diagnostics"/(stem+".empty.json"),{"schema_version":"state_collection.empty.v1",
                "collection_schema_version":"state_collection.v2","prompt_id":str(unit["sample_id"]),"path":unit["scenario"],
                "count":0,"dtype":"float64","ddof":0,"metadata":identity,"position_bins":[list(x) for x in cost.base.BINS],
                "numerical_protection":cost.base.EPS,"reason":"immediate_eos"})
            archive=cost.audit_observer_archive(attempt,unit,self.cals,"protocol")
            modes={m:mode(unit,m,archive,.8 if m=="on" else 1.) for m in ("on","off")}
        for m,value in modes.items():put(attempt/(m+".mode.json"),value)
        pair=cost.sealed({"schema_version":"state_observer_cost.pair.v1","plan_hash":"plan",
            "source_snapshot_hash":self.snapshot["snapshot_hash"],"unit_hash":unit["unit_hash"],"order":unit["order"],
            "attempt_relative_dir":str(attempt.relative_to(self.out)),"modes":{m:{"relative_path":str((attempt/(m+".mode.json")).relative_to(self.out)),
            "sha256":cost.sha(attempt/(m+".mode.json"))} for m in modes},"signed_differences":cost.signed_differences(modes["on"],modes["off"])},"pair_hash")
        put(self.out/"pairs"/(unit["unit_hash"]+".json"),pair)

    def scientific_delivery(self):
        directory=self.root/"science";directory.mkdir(exist_ok=True)
        (directory/"RUN_REPORT.md").write_text("# Scientific\n"+"\n".join(f"## {i}. 问题\n原科学正文" for i in range(1,10))+"\n")
        put(directory/"validation512_summary.json",{"status":"INCONCLUSIVE"})
        put(directory/"delivery_receipt.json",{"delivery_hash":"scientific"})
        return mock.patch.object(report.science,"verify_delivery",return_value={"verified":True,"delivery_complete":True,"delivery_dir":str(directory)})

    def test_actual_16_empty_pairs_complete_even_negative_deltas(self):
        proof=report.check_observer_cost(self.run)
        self.assertTrue(proof["complete"],proof);self.assertTrue(proof["data_complete"])
        self.assertFalse(proof["not_required"]);self.assertEqual(proof["pair_count"],16)

    def test_real_frozen_48x2048_d1_to_33_writer_audit_analyze_delivery(self):
        body={k:v for k,v in self.units[0].items() if k!="unit_hash"}
        body.update(trace=list(range(100,133))+[9],generation_row={"gen_token_ids":list(range(100,133))+[9]},common_decode_limit=33)
        self.units[0]=cost.sealed(body,"unit_hash")
        shutil.rmtree(self.out);self.out.mkdir();put(self.out/"source_snapshot.json",self.snapshot)
        for index,unit in enumerate(self.units):self.write_pair(unit,real_on=index==0)
        put(self.out/"raw_completion.json",cost.raw_receipt(self.out,self.plan,self.snapshot,self.units,self.cals))
        cost.analyze(self.run,self.plan,self.out)
        proof=report.check_observer_cost(self.run);self.assertTrue(proof["complete"],proof)
        first=proof["pairs"][0]["modes"]["on"]["archive_receipt"]
        self.assertEqual(first["layer_count"],48);self.assertEqual(first["decode_forwards"],33)
        self.assertGreater(first["moment_headers_checked"],48*5)
        self.assertFalse(first["full_moment_value_scan"])
        with self.scientific_delivery():
            result=report.ensure_delivery(self.run);self.assertTrue(result["delivery_complete"],result)
            self.assertTrue(report.verify_delivery(self.run)["verified"])

    def test_missing_off_mode_is_not_complete(self):
        unit=self.units[0];(self.out/"attempts"/unit["unit_hash"]/"try1/off.mode.json").unlink()
        self.assertFalse(report.check_observer_cost(self.run)["complete"])

    def test_committed_archive_byte_tamper_is_blocked(self):
        path=next((self.out/"attempts").glob("*/*/diagnostics/*.empty.json"));path.write_text(path.read_text()+" ")
        self.assertEqual(report.check_observer_cost(self.run)["status"],"BLOCKED")

    def test_warmup_proof_is_required_not_inferred_from_pair(self):
        path=next((self.out/"attempts").glob("*/*/warmup.json"));value=cost.load(path);value["excluded_from_measurement"]=False;put(path,value)
        self.assertEqual(report.check_observer_cost(self.run)["status"],"BLOCKED")

    def test_forged_descriptive_summary_cannot_finish_work(self):
        path=self.out/"summaries/observer_cost_summary.json";value=cost.load(path);value["by_scenario"]["A"]["n_pairs"]=99;put(path,value)
        done=cost.load(self.out/"completion.json");done["summary_sha256"]=cost.sha(path);put(self.out/"completion.json",reseal(done,"completion_hash"))
        proof=report.check_observer_cost(self.run)
        self.assertFalse(proof["complete"]);self.assertTrue(proof["data_complete"])

    def test_false_128_label_or_missing_fixed_role_is_rejected(self):
        self.units.pop();self.assertFalse(report.check_observer_cost(self.run)["complete"])

    def test_deliver_links_actual_nine_questions_preserves_status_and_reuses(self):
        with self.scientific_delivery(),mock.patch.object(report.science,"ensure_delivery",side_effect=AssertionError("No statistics owner call")):
            result=report.ensure_delivery(self.run);self.assertTrue(result["delivery_complete"],result)
            self.assertEqual(result["scientific_status"],"INCONCLUSIVE")
            body=(self.out/"delivery/RUN_REPORT.md").read_text()
            self.assertIn("## 9.",body);self.assertIn("不是原run当时开销",body);self.assertIn("host instrumentation",body)
            receipt=(self.out/"delivery/delivery_receipt.json").read_bytes()
            self.assertTrue(report.ensure_delivery(self.run)["reused"])
            self.assertEqual(receipt,(self.out/"delivery/delivery_receipt.json").read_bytes())

    def test_tampered_report_and_scientific_identity_invalidate_delivery(self):
        with self.scientific_delivery():
            self.assertTrue(report.ensure_delivery(self.run)["verified"])
            (self.out/"delivery/RUN_REPORT.md").write_text("forged")
            self.assertFalse(report.verify_delivery(self.run)["verified"])

    def test_io_failure_preserves_work_and_resumes_only_report(self):
        with self.scientific_delivery(),mock.patch.object(report.science,"_atomic_text",side_effect=OSError("disk test")):
            self.assertFalse(report.ensure_delivery(self.run)["delivery_complete"])
        self.assertTrue(report.check_observer_cost(self.run)["complete"])
        with self.scientific_delivery():self.assertTrue(report.ensure_delivery(self.run)["delivery_complete"])

    def test_scientific_delivery_pending_cannot_become_final(self):
        with mock.patch.object(report.science,"verify_delivery",return_value={"verified":False,"delivery_complete":False}):
            self.assertFalse(report.ensure_delivery(self.run)["delivery_complete"])
        self.assertFalse((self.out/"delivery/delivery_receipt.json").exists())


class BudgetAccountingTests(unittest.TestCase):
    def test_physical_jsonl_shared_rollout_and_unknown_total(self):
        with tempfile.TemporaryDirectory(dir="/scratch/jgu7/tmp") as temporary:
            run=Path(temporary);path=run/"formal/calibration/updated_calibration_cost_attempts.jsonl"
            put(path,{"M_NT":{"rollout_seconds_shared":12.,"note":"Unicode\u2028separator"},"M_ST":{"rollout_seconds_shared":12.}})
            result=report.calibration_cost_accounting(run)
            self.assertEqual(len(result["updated_attempt_records"]),1)
            self.assertEqual(result["shared_student_rollout_seconds_once"],12.)
            self.assertIsNone(result["complete_offline_wall_seconds"])
            self.assertTrue(result["no_equal_total_budget_claim"])

    def test_nonfinite_json_cannot_become_cost_evidence(self):
        with tempfile.TemporaryDirectory(dir="/scratch/jgu7/tmp") as temporary:
            path=Path(temporary)/"formal/calibration/updated_calibration_cost_attempts.jsonl";path.parent.mkdir(parents=True)
            path.write_text('{"M_ST":{"replay_seconds":NaN}}\n')
            with self.assertRaises(report.Invalid):report.calibration_cost_accounting(temporary)

    def test_valid_json_tail_without_physical_LF_is_not_committed(self):
        with tempfile.TemporaryDirectory(dir="/scratch/jgu7/tmp") as temporary:
            path=Path(temporary)/"formal/calibration/updated_calibration_cost_attempts.jsonl";path.parent.mkdir(parents=True)
            original=b'{"M_ST":{"replay_seconds":1.0}}';path.write_bytes(original)
            with self.assertRaisesRegex(report.Invalid,"Uncommitted.*tail"):
                report.calibration_cost_accounting(temporary)
            self.assertEqual(path.read_bytes(),original)
