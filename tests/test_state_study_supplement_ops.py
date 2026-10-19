"""Fake CPU handoffs for the independent full512 phase; no real job/model."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts import state_study_supplement_ops as ops
from scripts import state_study_continuation as continuation


REPO = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def launcher_module():
    with mock.patch.object(sys, "path", [str(REPO / "scripts"), *sys.path]):
        spec = importlib.util.spec_from_file_location("supplement_test_launcher", REPO / "scripts/launch_state_study_resume.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/scratch/jgu7/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run, self.reports = self.root / "run", self.root / "reports"
        self.run.mkdir()
        self.launcher = launcher_module()
        self.core_command = [sys.executable, "-u", str(REPO / "scripts/run_state_calibration_study.py"),
                             "auto", "--run-dir", str(self.run)]
        self.old = {"pid": 101, "hostname": "old-node", "boot_id": "old-boot",
                    "proc_start_ticks": "10", "slurm_job_id": "100", "command": self.core_command}
        write(self.run / "launch.json", self.old)
        self.original_core = (self.run / "launch.json").read_bytes()
        self.manifest = {"protocol_hash": "protocol", "stages": {"P5": {"status": "complete"},
                         "P6_P7": {"status": "gated_pending"}}}
        write(self.run / "manifest.json", self.manifest)
        (self.run / "RUN_REPORT.md").write_text("complete core; supplement pending\n")
        self.output = self.reports / self.run.name / "validation512/plan-hash"
        self.state = {"status": "READY", "plan_hash": "plan-hash", "output_dir": str(self.output)}
        self.cost_state = {"status": "COMPLETE", "delivery_complete": True, "plan_hash": "cost-plan"}
        self.args = SimpleNamespace(previous_job_id="100", check_only=False,
            log="/scratch/jgu7/moe-state-study/slurm/fake-supplement-200.log")
        self.audit = {"sources": {"protocol_hash": "protocol", "source_hash": "source"}}
        self.terminal = {"status": "VERIFIED_TERMINAL", "job_id": "100", "state": "TIMEOUT"}
        self.allocation = {"job_id": "200", "cgroup": "0::/jobs/job_200/step_batch", "scontrol": {}}
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in (("verify_launch_process", {"status": "UNKNOWN"}),
                            ("predecessor_terminal", self.terminal),
                            ("predecessor_stop", None),
                            ("current_allocation", self.allocation),
                            ("validate_allocation_resources", None),
                            ("validate_sources", (self.audit["sources"], self.manifest))):
            self.stack.enter_context(mock.patch.object(self.launcher, name, return_value=value))
        self.queue = self.stack.enter_context(mock.patch.object(self.launcher, "ensure_successor",
            return_value={"status": "SUBMISSION_UNKNOWN"}))
        self.delivery = self.stack.enter_context(mock.patch.object(ops, "ensure_not_required_delivery",
            return_value={"verified": True, "delivery_complete": True, "status": "VERIFIED"}))
        self.delivery_verify = self.stack.enter_context(mock.patch.object(ops, "verify_delivery",
            return_value={"verified": False, "delivery_complete": False, "status": "MISSING"}))
        self.stack.enter_context(mock.patch.object(self.launcher, "record_outcome", return_value="fake-outcome.json"))
        self.execute = self.stack.enter_context(mock.patch.object(ops.os, "execvpe", side_effect=OSError("fake exec only")))
        self.stack.enter_context(mock.patch.object(ops.os, "chdir"))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(contextlib.redirect_stderr(io.StringIO()))

    def dispatch(self, state=None):
        return ops.dispatch(self.args, self.run, self.audit, self.manifest, self.launcher,
            reports=self.reports, state_fn=lambda run: state or self.state,
            context_fn=lambda run, state, reports: ({"plan_hash": state["plan_hash"]}, self.output),
            cost_state_fn=lambda run: self.cost_state)

    def test_core_complete_pending_plan_is_not_overall_completion(self):
        for status in ("PLAN_PENDING", "INVALID_PLAN", "WAITING_P4", "WAITING_CORE", "SCIENTIFIC_ERROR"):
            with self.subTest(status=status), self.assertRaises(continuation.ContinuationRefused):
                self.dispatch({"status": status})
        self.queue.assert_not_called()
        self.execute.assert_not_called()
        self.assertFalse(ops.ops_dir(self.run, self.reports).exists())

    def test_real_cpu_status_interface_missing_plan_is_not_done(self):
        # Exercise the real public CPU API, with a unique nonexistent plan;
        # this does not call a freezer, collector, bootstrap, or model.
        result = ops.supplement_state(self.root)
        self.assertEqual(result["status"], "PLAN_PENDING")
        self.assertNotIn(result["status"], ops.DONE)
        self.queue.assert_not_called()
        self.execute.assert_not_called()

    def test_only_explicit_not_required_or_complete_stops_without_model(self):
        for status in ("NOT_REQUIRED", "COMPLETE"):
            self.assertEqual(self.dispatch({"status": status, "delivery_complete": True}), 0)
        self.queue.assert_not_called()
        self.execute.assert_not_called()
        self.assertEqual((self.run / "launch.json").read_bytes(), self.original_core)
        self.delivery.assert_not_called()

    def test_not_required_check_only_never_writes_missing_delivery(self):
        self.args.check_only = True
        self.assertEqual(self.dispatch({"status": "NOT_REQUIRED"}), 0)
        self.delivery.assert_not_called()
        self.delivery_verify.assert_called_once_with(self.run)
        self.queue.assert_not_called()
        self.execute.assert_not_called()

    def test_not_required_failed_delivery_is_not_overall_complete(self):
        self.delivery.side_effect = continuation.ContinuationRefused("delivery failed")
        with self.assertRaises(continuation.ContinuationRefused):
            self.dispatch({"status": "NOT_REQUIRED"})
        self.queue.assert_not_called()
        self.execute.assert_not_called()

    def test_complete_label_without_delivery_proof_cannot_stop(self):
        with self.assertRaises(continuation.ContinuationRefused):
            self.dispatch({"status": "COMPLETE", "delivery_complete": False})
        self.queue.assert_not_called()
        self.execute.assert_not_called()

    def test_live_predecessor_or_unknown_terminal_never_claims(self):
        with mock.patch.object(self.launcher, "verify_launch_process", return_value={"status": "VERIFIED_LIVE"}):
            with self.assertRaises(continuation.ContinuationRefused):self.dispatch()
        with mock.patch.object(self.launcher, "predecessor_terminal", return_value={"status": "UNKNOWN"}):
            with self.assertRaises(continuation.ContinuationRefused):self.dispatch()
        self.assertFalse(ops.ops_dir(self.run, self.reports).exists())
        self.queue.assert_not_called()

    def test_resource_guard_precedes_shared_claim(self):
        with mock.patch.object(self.launcher, "validate_allocation_resources", side_effect=continuation.ContinuationRefused("wrong GPU")):
            with self.assertRaises(continuation.ContinuationRefused):self.dispatch()
        self.assertFalse(ops.ops_dir(self.run, self.reports).exists())
        self.queue.assert_not_called()

    def test_missing_log_refuses_before_consuming_shared_claim(self):
        self.args.log = None
        with self.assertRaises(continuation.ContinuationRefused):self.dispatch()
        self.assertFalse(ops.ops_dir(self.run, self.reports).exists())
        self.queue.assert_not_called()

    def test_checked_handoff_preserves_core_and_execs_supplement_same_pid(self):
        with self.assertRaises(continuation.ContinuationRefused):self.dispatch()
        record = json.loads(ops.launch_path(self.run, self.reports).read_text())
        self.assertEqual(record["pid"], os.getpid())
        self.assertEqual(record["task"], "validation512")
        self.assertEqual(record["plan_hash"], "plan-hash")
        self.assertEqual(record["previous_slurm_job_id"], "100")
        self.assertEqual(record["core_terminal_proof"], self.terminal)
        self.assertEqual((self.run / "launch.json").read_bytes(), self.original_core)
        claim = record["shared_handoff_claim"]
        claim_raw = Path(claim["path"]).read_bytes()
        self.assertEqual(ops.checksum(claim_raw), claim["sha256"])
        self.assertEqual(json.loads(claim_raw)["pid"], os.getpid())
        self.assertEqual(Path(record["previous_launch_history"]).read_bytes(), self.original_core)
        archive = record["ops_source_archive"]
        self.assertEqual(ops.checksum((Path(archive["directory"]) / "MANIFEST.json").read_bytes()),
                         archive["manifest_sha256"])
        self.assertEqual((Path(archive["directory"]) / "scripts/state_study_supplement_ops.py").read_bytes(),
                         (REPO / "scripts/state_study_supplement_ops.py").read_bytes())
        self.queue.assert_called_once_with(self.run, "200", phase="validation512")
        self.assertEqual(self.execute.call_args.args[:2],
                         (sys.executable, ops.command(self.run)))

    def test_same_effective_predecessor_cannot_be_claimed_twice(self):
        with self.assertRaises(continuation.ContinuationRefused):self.dispatch()
        self.args.previous_job_id = "200"
        # A live same-PID supplement must refuse before its history is touched.
        with mock.patch.object(self.launcher, "verify_launch_process", return_value={"status": "VERIFIED_LIVE"}):
            with self.assertRaises(continuation.ContinuationRefused):self.dispatch()
        self.assertEqual(self.queue.call_count, 1)

    def test_later_allocation_uses_recent_independent_launch_not_old_core(self):
        with self.assertRaises(continuation.ContinuationRefused):self.dispatch()
        previous = ops.launch_path(self.run, self.reports).read_bytes()
        self.args.previous_job_id = "100"
        with self.assertRaises(continuation.ContinuationRefused):self.dispatch()
        self.args.previous_job_id = "200"
        terminal = {**self.terminal, "job_id": "200"}
        with mock.patch.object(self.launcher, "current_allocation", return_value={**self.allocation, "job_id": "300"}), \
             mock.patch.object(self.launcher, "predecessor_terminal", return_value=terminal):
            with self.assertRaises(continuation.ContinuationRefused):self.dispatch()
        current = json.loads(ops.launch_path(self.run, self.reports).read_text())
        self.assertEqual(current["previous_slurm_job_id"], "200")
        self.assertEqual(current["slurm_job_id"], "300")
        self.assertEqual(current["core_terminal_proof"], self.terminal)
        self.assertEqual(Path(current["previous_launch_history"]).read_bytes(), previous)
        self.assertEqual((self.run / "launch.json").read_bytes(), self.original_core)

    def test_check_only_never_writes_or_execs_or_queues(self):
        self.args.check_only = True
        self.assertEqual(self.dispatch(), 0)
        self.assertFalse(ops.ops_dir(self.run, self.reports).exists())
        self.queue.assert_not_called()
        self.execute.assert_not_called()

    def test_incomplete_checkpoint_refuses_without_raw_repair(self):
        path = self.output / "replay_checks.jsonl"
        path.parent.mkdir(parents=True)
        path.write_bytes(b'{"sample_id":512}')
        with self.assertRaises(continuation.ContinuationRefused):self.dispatch()
        self.assertEqual(path.read_bytes(), b'{"sample_id":512}')
        self.assertFalse(ops.launch_path(self.run, self.reports).exists())
        self.queue.assert_not_called()

    def test_checkpoint_uses_only_physical_LF_with_U2028_and_CR_whitespace(self):
        path = self.output / "replay_checks.jsonl"
        path.parent.mkdir(parents=True)
        raw = '{"sample_id":512,"text":"a\u2028b",\r"path":"M_ST"}\n'.encode()
        path.write_bytes(raw)
        snapshot = ops.checkpoint_snapshot(self.output)
        self.assertEqual(snapshot["physical_lines"], 1)
        self.assertEqual(snapshot["files"][path.name]["sha256"], ops.checksum(raw))
        path.write_bytes(raw[:-1])
        with self.assertRaises(continuation.ContinuationRefused):
            ops.checkpoint_snapshot(self.output)
        self.assertEqual(path.read_bytes(), raw[:-1])

    def test_actual_ops_source_archive_is_reusable_and_immutable(self):
        archived = ops.publish_ops_sources(self.run, reports=self.reports)
        self.assertEqual(ops.publish_ops_sources(self.run, reports=self.reports), archived)
        copy = Path(archived["directory"]) / "scripts/state_study_supplement_ops.py"
        copy.write_text("corrupted fixture copy")
        with self.assertRaises(continuation.ContinuationRefused):
            ops.publish_ops_sources(self.run, reports=self.reports)


spec = importlib.util.spec_from_file_location("supplement_base_fixture", REPO / "tests/test_state_study_continuation.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


class SupplementContinuationTests(base.ContinuationFixture):
    def setUp(self):
        super().setUp()
        self.manifest["stages"] = {"P5": {"status": "complete"}, "P6_P7": {"status": "gated_pending"}}
        write(self.run / "manifest.json", self.manifest)
        (self.run / "RUN_REPORT.md").write_text("complete\n")
        self.state = {"status": "READY", "plan_hash": "plan"}
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(ops, "supplement_state", return_value=self.state))
        self.cost_state = {"status": "COMPLETE", "delivery_complete": True, "plan_hash": "cost-plan"}
        self.stack.enter_context(mock.patch.object(ops, "observer_cost_state", side_effect=lambda run:self.cost_state))
        core_raw = (self.run / "launch.json").read_bytes()
        claim_path = self.reports / self.run.name / "validation512/ops/claims/claim.json"
        write(claim_path, {"plan_hash": "plan", "pid": os.getpid()})
        self.launch.update(schema_version="state_validation512_launch.v1", task="validation512",
            phase="validation512", plan_hash="plan", supplement_plan_hash="plan",
            core_launch_sha256=ops.checksum(core_raw), command=ops.command(self.run, repo=self.repo),
            shared_handoff_claim={"path": str(claim_path), "sha256": ops.checksum(claim_path.read_bytes())})
        write(ops.launch_path(self.run, self.reports), self.launch)

    def ensure(self):
        return continuation.ensure_successor(self.run, "100", repo=self.repo, reports=self.reports,
            query_fn=self.slurm.query, submit_fn=self.slurm.submit, phase="validation512")

    def test_supplement_queues_one_ahead_even_though_core_complete(self):
        core_before = (self.run / "launch.json").read_bytes()
        self.assertEqual(self.ensure()["status"], "QUEUED")
        self.assertEqual(self.ensure()["status"], "ALREADY_QUEUED")
        self.assertEqual(len(self.slurm.submit_calls), 1)
        self.assertEqual((self.run / "launch.json").read_bytes(), core_before)

    def test_unknown_supplement_submit_does_not_repeat(self):
        self.slurm.submit_code = None
        self.slurm.submit_output = ""
        self.slurm.create_on_submit = False
        self.assertEqual(self.ensure()["status"], "SUBMISSION_UNKNOWN")
        self.assertEqual(self.ensure()["status"], "SUBMISSION_UNKNOWN")
        self.assertEqual(len(self.slurm.submit_calls), 1)

    def test_supplement_uncertain_submit_adopts_one_exact_pending_job(self):
        self.slurm.submit_code = None
        self.slurm.submit_output = ""
        self.assertEqual(self.ensure()["status"], "SUBMISSION_UNKNOWN")
        adopted = self.ensure()
        self.assertEqual(adopted["status"], "QUEUED")
        self.assertTrue(adopted["adopted_existing"])
        self.assertEqual(len(self.slurm.submit_calls), 1)

    def test_missing_or_invalid_plan_cannot_claim_completion(self):
        for status in ("PLAN_PENDING", "INVALID_PLAN", "WAITING_P4"):
            self.state["status"] = status
            self.assertEqual(self.ensure()["status"], "REFUSED")
        self.assertEqual(self.slurm.submit_calls, [])

    def test_only_complete_or_not_required_supplement_stops_chain(self):
        for status in ("COMPLETE", "NOT_REQUIRED"):
            self.state["status"] = status
            self.state["delivery_complete"] = True
            self.assertEqual(self.ensure()["status"], "ALREADY_COMPLETE_NO_SUBMISSION")
        self.assertEqual(self.slurm.submit_calls, [])

    def test_not_required_without_delivery_cannot_stop_as_complete(self):
        self.state.update(status="NOT_REQUIRED", delivery_complete=False)
        self.assertEqual(self.ensure()["status"], "SUPPLEMENT_DELIVERY_REQUIRED")
        self.assertEqual(self.slurm.submit_calls, [])

    def test_supplement_explicit_scientific_error_never_retries(self):
        self.state["status"] = "SCIENTIFIC_ERROR"
        self.assertEqual(self.ensure()["status"], "SCIENTIFIC_STOP")
        self.assertEqual(self.slurm.submit_calls, [])

    def test_tampered_independent_shared_claim_cannot_queue(self):
        Path(self.launch["shared_handoff_claim"]["path"]).write_text("changed")
        self.assertEqual(self.ensure()["status"], "REFUSED")
        self.assertEqual(self.slurm.submit_calls, [])

    def test_supplement_plan_or_original_core_launch_change_blocks_submission(self):
        changed = {**self.launch, "plan_hash": "different"}
        write(ops.launch_path(self.run, self.reports), changed)
        self.assertEqual(self.ensure()["status"], "REFUSED")
        write(ops.launch_path(self.run, self.reports), self.launch)
        write(self.run / "launch.json", {"changed": True})
        self.assertEqual(self.ensure()["status"], "REFUSED")
        self.assertEqual(self.slurm.submit_calls, [])


class DeliveryContractTests(unittest.TestCase):
    def test_not_required_delivery_requires_both_exact_true_booleans(self):
        for field in ("verified", "delivery_complete"):
            incomplete = {"verified": True, "delivery_complete": True, field: 1}
            module = SimpleNamespace(ensure_delivery=mock.Mock(return_value=incomplete),
                                     verify_delivery=mock.Mock())
            with mock.patch.object(ops, "_report", return_value=module):
                with self.assertRaises(continuation.ContinuationRefused):
                    ops.ensure_not_required_delivery(Path("/fake/read-only/run"))
            module.verify_delivery.assert_not_called()

    def test_delivery_is_independently_reverified_after_writer(self):
        good = {"verified": True, "delivery_complete": True, "status": "VERIFIED"}
        module = SimpleNamespace(ensure_delivery=mock.Mock(return_value=good),
            verify_delivery=mock.Mock(return_value={"verified": False, "delivery_complete": False}))
        with mock.patch.object(ops, "_report", return_value=module):
            with self.assertRaises(continuation.ContinuationRefused):
                ops.ensure_not_required_delivery(Path("/fake/read-only/run"))
            module.verify_delivery.return_value = good
            self.assertEqual(ops.ensure_not_required_delivery(Path("/fake/read-only/run")), good)


class ObserverCostDispatchTests(unittest.TestCase):
    """Exercise the additional required phase using real shared ops artifacts."""
    def setUp(self):
        self.f = DispatchTests("test_core_complete_pending_plan_is_not_overall_completion")
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        f = self.f
        f.state = {"status": "NOT_REQUIRED", "delivery_complete": True, "plan_hash": "plan-hash"}
        f.output = f.reports / f.run.name / "observer_cost/cost-plan"
        f.cost_state = {"status": "READY", "plan_hash": "cost-plan", "output_dir": str(f.output)}

    def handoff(self):
        with self.assertRaises(continuation.ContinuationRefused):self.f.dispatch()
        path = ops.launch_path(self.f.run, self.f.reports, phase=ops.COST_PHASE)
        self.assertTrue(path.is_file())  # Only the deliberate fake exec failed.
        return json.loads(path.read_text())

    def test_full512_complete_is_not_overall_done_with_pending_cost_plan(self):
        f = self.f
        f.state["status"] = "COMPLETE"
        for status in ("PLAN_PENDING", "INVALID_PLAN", "WAITING_CORE", "WAITING_VALIDATION512", "SCIENTIFIC_ERROR"):
            f.cost_state["status"] = status
            with self.subTest(status=status), self.assertRaises(continuation.ContinuationRefused):f.dispatch()
        f.queue.assert_not_called(); f.execute.assert_not_called()
        self.assertFalse(ops.launch_path(f.run, f.reports, phase=ops.COST_PHASE).exists())

    def test_cost_complete_without_exact_delivery_true_never_stops(self):
        for delivered in (False, 1, None):
            self.f.cost_state.update(status="COMPLETE", delivery_complete=delivered)
            with self.subTest(delivered=delivered), self.assertRaises(continuation.ContinuationRefused):self.f.dispatch()
        self.f.cost_state["delivery_complete"] = True
        self.assertEqual(self.f.dispatch(), 0)
        self.f.queue.assert_not_called(); self.f.execute.assert_not_called()

    def test_observer_handoff_same_pid_and_independent_metadata(self):
        f = self.f; record = self.handoff()
        self.assertEqual(record["schema_version"], "state_observer_cost_launch.v1")
        self.assertEqual(record["phase"], "observer_cost")
        self.assertEqual(record["pid"], os.getpid())
        self.assertEqual(record["observer_cost_plan_hash"], "cost-plan")
        self.assertEqual(record["validation512_plan_hash"], "plan-hash")
        self.assertEqual(record["command"], ops.command(f.run, phase=ops.COST_PHASE))
        self.assertEqual((f.run / "launch.json").read_bytes(), f.original_core)
        self.assertFalse(ops.launch_path(f.run, f.reports).exists())
        claim = json.loads(Path(record["shared_handoff_claim"]["path"]).read_text())
        self.assertEqual(claim["schema_version"], "state_observer_cost_handoff_claim.v1")
        self.assertEqual(claim["task"], "observer_cost")
        f.queue.assert_called_once_with(f.run, "200", phase="observer_cost")

    def test_not_required_still_requires_observer_benchmark(self):
        self.f.state["status"] = "NOT_REQUIRED"
        self.handoff()
        self.f.delivery.assert_not_called()

    def test_not_required_cpu_delivery_precedes_cost_handoff(self):
        f = self.f; f.state.update(status="NOT_REQUIRED", delivery_complete=False)
        def deliver(run):
            f.state["delivery_complete"] = True
            return {"verified": True, "delivery_complete": True}
        f.delivery.side_effect = deliver
        self.handoff()
        f.delivery.assert_called_once_with(f.run)

    def test_check_only_cost_ready_never_claims_execs_or_queues(self):
        f = self.f; f.args.check_only = True
        self.assertEqual(f.dispatch(), 0)
        f.queue.assert_not_called(); f.execute.assert_not_called()
        self.assertFalse(ops.ops_dir(f.run, f.reports, phase=ops.COST_PHASE).exists())

    def test_latest_cost_launch_is_next_predecessor_and_cannot_regress(self):
        f = self.f; self.handoff()
        f.args.previous_job_id = "100"
        with self.assertRaises(continuation.ContinuationRefused):f.dispatch()
        f.args.previous_job_id = "200"
        with mock.patch.object(f.launcher, "predecessor_terminal", return_value={**f.terminal,"job_id":"200"}), \
             mock.patch.object(f.launcher, "current_allocation", return_value={**f.allocation,"job_id":"300"}):
            record = self.handoff()
        self.assertEqual(record["previous_slurm_job_id"], "200")
        self.assertEqual(record["slurm_job_id"], "300")
        f.args.previous_job_id = "300"; f.state.update(status="RAW_COMPLETE", delivery_complete=False)
        with self.assertRaises(continuation.ContinuationRefused):f.dispatch()

    def test_real_validation_to_cost_transition_preserves_both_predecessors(self):
        f=self.f
        f.state.update(status="READY",delivery_complete=False)
        f.output=f.reports/f.run.name/"validation512/plan-hash"
        with self.assertRaises(continuation.ContinuationRefused):f.dispatch()
        validation_raw=ops.launch_path(f.run,f.reports).read_bytes()
        f.state.update(status="COMPLETE",delivery_complete=True)
        f.output=f.reports/f.run.name/"observer_cost/cost-plan"
        f.args.previous_job_id="200"
        with mock.patch.object(f.launcher,"predecessor_terminal",return_value={**f.terminal,"job_id":"200"}), \
             mock.patch.object(f.launcher,"current_allocation",return_value={**f.allocation,"job_id":"300"}):
            record=self.handoff()
        self.assertEqual(record["previous_slurm_job_id"],"200")
        self.assertEqual(record["core_terminal_proof"]["job_id"],"100")
        self.assertEqual(record["validation512_terminal_proof"]["job_id"],"200")
        self.assertEqual(record["validation512_launch_sha256"],ops.checksum(validation_raw))
        self.assertEqual(ops.launch_path(f.run,f.reports).read_bytes(),validation_raw)
        self.assertEqual((f.run/"launch.json").read_bytes(),f.original_core)

    def registered_fixture(self):
        f = self.f; record = self.handoff()
        validation_path = f.reports / f.run.name / "validation512/plan.json"
        validation_plan = {"plan_hash":"plan-hash","protocol_hash":"protocol","source":{"frozen_source_hash":"source"}}
        write(validation_path, validation_plan)
        plan = {"plan_hash":"cost-plan","protocol_hash":"protocol","source":validation_plan["source"],
                "validation512_plan":{"path":str(validation_path),"sha256":ops.checksum(validation_path.read_bytes()),"plan_hash":"plan-hash"}}
        write(f.run / "protocol.json", {"source_hash":"source","protocol_hash":"protocol"})
        proc = f.root / "proc"
        for name, data in (("self/stat", Path("/proc/self/stat").read_bytes()),
                           ("self/cgroup",record["cgroup"].encode()),
                           ("sys/kernel/random/boot_id",(record["boot_id"]+"\n").encode()),
                           ("self/cmdline",b"\0".join(x.encode() for x in record["command"])+b"\0")):
            path=proc/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        f.stack.enter_context(mock.patch.object(ops,"_cost",return_value=SimpleNamespace(load_plan=lambda run:plan)))
        f.stack.enter_context(mock.patch.object(ops,"_study",return_value=SimpleNamespace(load_plan=lambda run:validation_plan,plan_path=lambda run:validation_path)))
        f.stack.enter_context(mock.patch.object(ops,"supplement_state",side_effect=lambda run:f.state))
        return plan,proc,record

    def register(self, plan, proc):
        f=self.f
        return ops.registered_cost_launch(f.run,plan,reports=f.reports,proc=proc,
                                         environment={"SLURM_JOB_ID":"200"},resume=f.launcher)

    def test_actual_registered_claim_archive_and_same_pid_are_verified_cpu(self):
        plan,proc,record=self.registered_fixture()
        proof=self.register(plan,proc)
        self.assertEqual(proof["pid"],os.getpid());self.assertEqual(proof["phase"],"observer_cost")
        self.assertEqual(proof["claim_sha256"],record["shared_handoff_claim"]["sha256"])

    def test_registered_cost_requires_exact_proc_argv_boot_and_cgroup(self):
        plan,proc,_=self.registered_fixture()
        for name in ("self/cmdline","sys/kernel/random/boot_id","self/cgroup"):
            path=proc/name;raw=path.read_bytes();path.write_bytes(b"mismatch")
            with self.subTest(name=name), self.assertRaises(continuation.ContinuationRefused):self.register(plan,proc)
            path.write_bytes(raw)

    def test_registered_cost_rejects_live_core_or_unknown_terminal(self):
        plan,proc,_=self.registered_fixture()
        with mock.patch.object(self.f.launcher,"verify_launch_process",return_value={"status":"VERIFIED_LIVE"}):
            with self.assertRaises(continuation.ContinuationRefused):self.register(plan,proc)
        with mock.patch.object(self.f.launcher,"predecessor_terminal",return_value={"status":"UNKNOWN"}):
            with self.assertRaises(continuation.ContinuationRefused):self.register(plan,proc)

    def test_registered_cost_refuses_changed_claim_history_and_archive(self):
        plan,proc,record=self.registered_fixture()
        for name in (record["shared_handoff_claim"]["path"],record["previous_launch_history"],
                     str(Path(record["ops_source_archive"]["directory"])/"scripts/state_study_supplement_ops.py")):
            path=Path(name);raw=path.read_bytes();path.write_bytes(b"changed")
            with self.subTest(name=name), self.assertRaises((continuation.ContinuationRefused,ValueError)):self.register(plan,proc)
            path.write_bytes(raw)

    def test_registered_cost_refuses_incomplete_validation_delivery(self):
        plan,proc,_=self.registered_fixture();self.f.state["delivery_complete"]=False
        with self.assertRaises(continuation.ContinuationRefused):self.register(plan,proc)


class ObserverCostContinuationTests(unittest.TestCase):
    def setUp(self):
        self.f=SupplementContinuationTests("test_supplement_queues_one_ahead_even_though_core_complete")
        self.f.setUp();self.addCleanup(self.f.doCleanups);f=self.f
        f.state.update(status="COMPLETE",delivery_complete=True)
        f.cost_state.update(status="READY",plan_hash="cost-plan",delivery_complete=False)
        claim_path=f.reports/f.run.name/"observer_cost/ops/claims/cost.json"
        write(claim_path,{"pid":os.getpid(),"plan_hash":"cost-plan"})
        self.launch={**f.launch,"schema_version":"state_observer_cost_launch.v1","task":"observer_cost","phase":"observer_cost",
          "plan_hash":"cost-plan","observer_cost_plan_hash":"cost-plan","command":ops.command(f.run,repo=f.repo,phase="observer_cost"),
          "shared_handoff_claim":{"path":str(claim_path),"sha256":ops.checksum(claim_path.read_bytes())}}
        write(ops.launch_path(f.run,f.reports,phase="observer_cost"),self.launch)

    def ensure(self):
        f=self.f
        return continuation.ensure_successor(f.run,"100",repo=f.repo,reports=f.reports,
            query_fn=f.slurm.query,submit_fn=f.slurm.submit,phase="observer_cost")

    def test_cost_one_hop_idempotent_and_binds_phase_plan(self):
        self.assertEqual(self.ensure()["status"],"QUEUED")
        self.assertEqual(self.ensure()["status"],"ALREADY_QUEUED")
        self.assertEqual(len(self.f.slurm.submit_calls),1)
        intent=continuation.read_json(self.f.directory/"intent.json")
        self.assertEqual(intent["phase"],"observer_cost");self.assertEqual(intent["observer_cost_plan_hash"],"cost-plan")

    def test_cost_unknown_never_resubmits_and_exact_adoption(self):
        f=self.f;f.slurm.submit_code=None;f.slurm.submit_output=""
        self.assertEqual(self.ensure()["status"],"SUBMISSION_UNKNOWN")
        self.assertEqual(self.ensure()["status"],"QUEUED")
        self.assertEqual(len(f.slurm.submit_calls),1)

    def test_full512_done_does_not_stop_cost_but_delivery_does(self):
        self.assertEqual(self.ensure()["status"],"QUEUED")
        self.f.cost_state.update(status="COMPLETE",delivery_complete=False)
        self.assertEqual(self.ensure()["status"],"REFUSED")
        self.f.cost_state["delivery_complete"]=True
        self.assertEqual(self.ensure()["status"],"ALREADY_COMPLETE_NO_SUBMISSION")

    def test_cost_cannot_skip_validation_delivery_or_bad_claim(self):
        self.f.state["delivery_complete"]=False
        self.assertEqual(self.ensure()["status"],"REFUSED")
        self.f.state["delivery_complete"]=True
        Path(self.launch["shared_handoff_claim"]["path"]).write_text("changed")
        self.assertEqual(self.ensure()["status"],"REFUSED")
        self.assertEqual(self.f.slurm.submit_calls,[])


class DeliveryPendingResumeTests(unittest.TestCase):
    """Real frozen CLI Pending return, old batch record and exact FAILED proof."""
    def setUp(self):
        self.base=ObserverCostDispatchTests("test_observer_handoff_same_pid_and_independent_metadata")
        self.base.setUp();self.addCleanup(self.base.doCleanups);self.f=self.base.f
        self.old=self.base.handoff()
        self.plan={"plan_hash":"cost-plan","protocol_hash":"protocol","source":{"frozen_source_hash":"source"}}
        self.f.cost_state.update(status="RAW_COMPLETE",delivery_complete=False,
            work_proof={"complete":True},delivery_proof={"verified":False,"delivery_complete":False,"status":"PENDING"})
        write(self.f.output/"completion.json",{"completed_work":True,"completion_hash":"complete"})
        self.module=SimpleNamespace(load_plan=lambda run:self.plan,status=lambda run:self.f.cost_state,
                                    output_dir=lambda run,plan:self.f.output)
        self.f.stack.enter_context(mock.patch.object(ops,"_cost",return_value=self.module))
        self.f.args.previous_job_id="200"
        self.f.launcher.current_allocation.return_value={**self.f.allocation,"job_id":"300","cgroup":"0::/jobs/job_300/step_batch"}
        clean=launcher_module()
        self.terminal=clean.predecessor_terminal("200",query_fn=lambda argv:{"returncode":0,
            "stdout":"JobId=200 JobState=FAILED ExitCode=2:0 EndTime=2026-10-20T03:05:52 NodeList=node\n","stderr":""})
        self.assertEqual(self.terminal["status"],"VERIFIED_TERMINAL")
        self.f.launcher.predecessor_terminal.side_effect=lambda job:{**self.terminal,"job_id":str(job),
            "state":"TIMEOUT" if str(job)=="100" else self.terminal["state"]}
        # Use the real legacy stop guard, not the original fixture's no-op.
        self.f.launcher.predecessor_stop.side_effect=lambda run,job,proof:continuation.predecessor_stop(run,job,proof,reports=self.f.reports)
        self.codes=self.f.reports/self.f.run.name/"batch-exit-codes-200.json"

    def actual_cli_pending_exit(self):
        with mock.patch.object(sys,"path",[str(REPO/"scripts"),*sys.path]):
            import state_study_observer_cost as cost
        pending=SimpleNamespace(ensure_delivery=lambda *a,**k:{"verified":False,"delivery_complete":False,"status":"PENDING"})
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(cost,"load_plan",return_value=self.plan))
            stack.enter_context(mock.patch.object(cost,"status",return_value=self.f.cost_state))
            stack.enter_context(mock.patch.object(cost,"output_dir",return_value=self.f.output))
            stack.enter_context(mock.patch.object(cost.base,"local_lock",return_value=SimpleNamespace(close=lambda:None)))
            stack.enter_context(mock.patch.object(cost,"analyze",return_value={"completed_work":True}))
            stack.enter_context(mock.patch.dict(sys.modules,{"state_study_supplement_ops":SimpleNamespace(registered_cost_launch=lambda *a:None),
                                                           "state_study_observer_cost_report":pending}))
            (self.f.output/"raw_completion.json").write_text("{}")
            code=cost.main(["run","--run-dir",str(self.f.run)])
        self.assertEqual(code,2)
        write(self.codes,{"scientific_exit_code":code,"report_exit_code":0,"plot_exit_code":0})
        return code

    def retry(self):
        with self.assertRaises(continuation.ContinuationRefused):self.f.dispatch()
        record=json.loads(ops.launch_path(self.f.run,self.f.reports,phase="observer_cost").read_text())
        self.assertEqual(record["slurm_job_id"],"300")
        return record

    def test_actual_cli2_and_exact_FAILED_resume_once_without_changing_work(self):
        self.actual_cli_pending_exit()
        before=(self.f.output/"completion.json").read_bytes()
        record=self.retry()
        self.assertTrue(record["cpu_delivery_retry"])
        self.assertEqual((self.f.output/"completion.json").read_bytes(),before)
        self.assertEqual((self.f.run/"launch.json").read_bytes(),self.f.original_core)

    def test_second_pending_delivery_stops_without_new_claim_exec_or_queue(self):
        self.actual_cli_pending_exit();self.retry()
        calls=self.f.queue.call_count
        write(self.f.reports/self.f.run.name/"batch-exit-codes-300.json",
              {"scientific_exit_code":2,"report_exit_code":0,"plot_exit_code":0})
        self.f.args.previous_job_id="300"
        self.f.launcher.current_allocation.return_value={**self.f.allocation,"job_id":"400"}
        with self.assertRaises(continuation.ContinuationRefused):self.f.dispatch()
        self.assertEqual(self.f.queue.call_count,calls)
        self.assertEqual(json.loads(ops.launch_path(self.f.run,self.f.reports,phase="observer_cost").read_text())["slurm_job_id"],"300")

    def test_resolved_delivery_after_operator_repair_is_not_permanently_poisoned(self):
        self.actual_cli_pending_exit()
        receipt=self.f.output/"delivery/delivery_receipt.json";write(receipt,{"delivery_hash":"delivered"})
        self.f.cost_state.update(status="COMPLETE",delivery_complete=True,
            delivery_proof={"verified":True,"delivery_complete":True,"status":"VERIFIED","plan_hash":"cost-plan",
                            "receipt_path":str(receipt),"receipt_sha256":ops.checksum(receipt.read_bytes()),"delivery_hash":"delivered"})
        self.assertEqual(self.f.dispatch(),0)
        self.assertEqual(self.f.queue.call_count,1)  # Only the original fake handoff.

    def test_check_only_proves_eligibility_without_spending_retry(self):
        self.actual_cli_pending_exit();self.f.args.check_only=True
        before=ops.launch_path(self.f.run,self.f.reports,phase="observer_cost").read_bytes()
        self.assertEqual(self.f.dispatch(),0)
        local=ops.ops_dir(self.f.run,self.f.reports,phase="observer_cost")
        self.assertFalse((local/"delivery_retries").exists())
        self.assertEqual(ops.launch_path(self.f.run,self.f.reports,phase="observer_cost").read_bytes(),before)
        self.assertEqual(self.f.queue.call_count,1)

    def test_invalid_raw_only_science_failure_and_non2_records_never_resume(self):
        self.actual_cli_pending_exit();initial=dict(self.f.cost_state)
        changes=({"work_proof":{}},{"work_proof":{"complete":1}},
            {"delivery_proof":{"verified":False,"delivery_complete":False,"status":"INVALID"}},
            {"status":"INVALID_PLAN"},{"status":"SCIENTIFIC_ERROR"})
        for change in changes:
            self.f.cost_state.clear();self.f.cost_state.update(initial);self.f.cost_state.update(change)
            with self.subTest(change=change), self.assertRaises(continuation.ContinuationRefused):self.f.dispatch()
        self.f.cost_state.clear();self.f.cost_state.update(initial)
        for value in (0,3,True,"2"):
            write(self.codes,{"scientific_exit_code":value,"report_exit_code":0,"plot_exit_code":0})
            with self.subTest(code=value), self.assertRaises(continuation.ContinuationRefused):self.f.dispatch()
        for raw in (b'{"scientific_exit_code":2}',b'{"scientific_exit_code":2,"report_exit_code":0,"plot_exit_code":0,"scientific_exit_code":2}',b'not json'):
            self.codes.write_bytes(raw)
            with self.subTest(raw=raw), self.assertRaises((continuation.ContinuationRefused,ValueError)):self.f.dispatch()
        self.assertEqual(self.f.queue.call_count,1)

    def test_core_oom_cancelled_and_unknown_stay_refused(self):
        self.actual_cli_pending_exit()
        for state in ("OUT_OF_MEMORY","CANCELLED"):
            self.terminal["state"]=state
            with self.subTest(state=state), self.assertRaises(continuation.ContinuationRefused):self.f.dispatch()
        self.terminal["status"]="UNKNOWN"
        with self.assertRaises(continuation.ContinuationRefused):self.f.dispatch()
        write(self.f.reports/self.f.run.name/"batch-exit-codes-100.json",
            {"scientific_exit_code":2,"report_exit_code":0,"plot_exit_code":0})
        guard=ops.phase_predecessor_guard(self.f.run,"100",{**self.terminal,"status":"VERIFIED_TERMINAL","job_id":"100","state":"FAILED"},
            self.f.launcher,phase="core",launch=json.loads(self.f.original_core),reports=self.f.reports)
        self.assertIsNotNone(guard["stop"]);self.assertIsNone(guard["delivery_exception"])

    def test_registered_current_cost_retry_is_idempotent_for_same_pid_job(self):
        self.actual_cli_pending_exit()
        validation_path=self.f.reports/self.f.run.name/"validation512/plan.json"
        validation_plan={"plan_hash":"plan-hash","protocol_hash":"protocol","source":self.plan["source"]}
        write(validation_path,validation_plan)
        self.plan["validation512_plan"]={"path":str(validation_path),"sha256":ops.checksum(validation_path.read_bytes()),"plan_hash":"plan-hash"}
        record=self.retry();f=self.f
        f.stack.enter_context(mock.patch.object(ops,"_study",return_value=SimpleNamespace(load_plan=lambda run:validation_plan,plan_path=lambda run:validation_path)))
        f.stack.enter_context(mock.patch.object(ops,"supplement_state",side_effect=lambda run:f.state))
        write(f.run/"protocol.json",{"source_hash":"source","protocol_hash":"protocol"})
        proc=f.root/"registered_proc"
        for name,data in (("self/stat",Path("/proc/self/stat").read_bytes()),("self/cgroup",record["cgroup"].encode()),
                          ("sys/kernel/random/boot_id",record["boot_id"].encode()),
                          ("self/cmdline",b"\0".join(x.encode() for x in record["command"])+b"\0")):
            path=proc/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        intent=Path(record["cpu_delivery_retry"]["path"]);before=intent.read_bytes()
        for _ in range(2):
            proof=ops.registered_cost_launch(f.run,self.plan,reports=f.reports,proc=proc,
                environment={"SLURM_JOB_ID":"300"},resume=f.launcher)
            self.assertEqual(proof["pid"],os.getpid())
        self.assertEqual(intent.read_bytes(),before)

    def test_retry_infrastructure_termination_without_batch_tail_does_not_reopen_budget(self):
        self.actual_cli_pending_exit();record=self.retry()
        self.f.args.previous_job_id="300"
        self.terminal["state"]="TIMEOUT"  # There is no batch-exit-codes-300.json.
        self.f.launcher.current_allocation.return_value={**self.f.allocation,"job_id":"400"}
        before=self.f.queue.call_count
        with self.assertRaises(continuation.ContinuationRefused):self.f.dispatch()
        self.assertEqual(self.f.queue.call_count,before)
        self.assertEqual(json.loads(ops.launch_path(self.f.run,self.f.reports,phase="observer_cost").read_text())["slurm_job_id"],"300")
        receipt=self.f.output/"delivery/delivery_receipt.json";write(receipt,{"delivery_hash":"fixed"})
        self.f.cost_state.update(status="COMPLETE",delivery_complete=True,delivery_proof={
            "verified":True,"delivery_complete":True,"status":"VERIFIED","plan_hash":"cost-plan",
            "receipt_path":str(receipt),"receipt_sha256":ops.checksum(receipt.read_bytes()),"delivery_hash":"fixed"})
        self.assertEqual(self.f.dispatch(),0)
        self.assertTrue(Path(record["cpu_delivery_retry"]["path"]).is_file())

    def test_actual_frozen_full512_cli2_MISSING_retry_registration_and_resolved_terminal(self):
        f=DispatchTests("test_core_complete_pending_plan_is_not_overall_completion")
        f.setUp();self.addCleanup(f.doCleanups)
        with self.assertRaises(continuation.ContinuationRefused):f.dispatch()
        f.state.update(status="RAW_COMPLETE",completion_proof={"complete":True},delivery_complete=False,
            delivery_proof={"verified":False,"delivery_complete":False,"status":"MISSING"})
        plan={"plan_hash":"plan-hash","protocol_hash":"protocol","source":{"frozen_source_hash":"source"}}
        write(f.output/"completion.json",{"completed_work":True});write(f.output/"raw_completion.json",{})
        with mock.patch.object(sys,"path",[str(REPO/"scripts"),*sys.path]):
            import state_study_validation512 as frozen
        with contextlib.ExitStack() as stack:
            for name,value in (("load_plan",plan),("inspect_status",{"status":"READY"}),("output_dir",f.output),
                ("registered_launch",None),("local_lock",SimpleNamespace(close=lambda:None)),
                ("source_snapshot",({}, {}, {})),("verify_raw_completion",None),("analyze",{"completed_work":True})):
                stack.enter_context(mock.patch.object(frozen,name,return_value=value))
            stack.enter_context(mock.patch.dict(sys.modules,{"state_study_validation512_report":SimpleNamespace(
                ensure_delivery=lambda *a,**k:{"verified":False,"delivery_complete":False,"status":"MISSING"})}))
            code=frozen.main(["run","--run-dir",str(f.run)])
        self.assertEqual(code,2)
        write(f.reports/f.run.name/"batch-exit-codes-200.json",{"scientific_exit_code":code,"report_exit_code":0,"plot_exit_code":0})
        module=SimpleNamespace(load_plan=lambda run:plan,status=lambda run:f.state,output_dir=lambda run,plan:f.output)
        f.stack.enter_context(mock.patch.object(ops,"_study",return_value=module))
        f.args.previous_job_id="200"
        f.launcher.predecessor_terminal.side_effect=lambda job:{"status":"VERIFIED_TERMINAL","job_id":str(job),
            "state":"TIMEOUT" if str(job)=="100" else "FAILED"}
        f.launcher.predecessor_stop.side_effect=lambda run,job,proof:continuation.predecessor_stop(run,job,proof,reports=f.reports)
        f.launcher.current_allocation.return_value={"job_id":"300","cgroup":"0::/jobs/job_300/step_batch"}
        with self.assertRaises(continuation.ContinuationRefused):f.dispatch()
        record=json.loads(ops.launch_path(f.run,f.reports).read_text());self.assertEqual(record["slurm_job_id"],"300")
        write(f.run/"protocol.json",{"source_hash":"source","protocol_hash":"protocol"})
        proc=f.root/"legacy_proc"
        for name,data in (("self/stat",Path("/proc/self/stat").read_bytes()),("self/cgroup",record["cgroup"].encode()),
                          ("sys/kernel/random/boot_id",record["boot_id"].encode()),
                          ("self/cmdline",b"\0".join(x.encode() for x in record["command"])+b"\0")):
            path=proc/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        actual=frozen.registered_launch(f.run,plan,reports=f.reports,proc=proc,
            environment={"SLURM_JOB_ID":"300"},resume=f.launcher)
        self.assertEqual(actual["pid"],os.getpid())
        pending_guard=ops.phase_predecessor_guard(f.run,"200",f.launcher.predecessor_terminal("200"),f.launcher,
            phase="validation512",launch=json.loads(Path(record["previous_launch_history"]).read_text()),reports=f.reports,current_launch=record)
        self.assertIsNone(pending_guard["stop"])
        # After a second Pending, operator success is actual delivery, not a reset.
        write(f.reports/f.run.name/"batch-exit-codes-300.json",{"scientific_exit_code":2,"report_exit_code":0,"plot_exit_code":0})
        receipt=f.output/"delivery/delivery_receipt.json";write(receipt,{"delivery_hash":"delivered"})
        f.state.update(status="COMPLETE",delivery_complete=True,delivery_proof={"verified":True,"delivery_complete":True,
            "status":"VERIFIED","plan_hash":"plan-hash","receipt_path":str(receipt),"receipt_sha256":ops.checksum(receipt.read_bytes()),"delivery_hash":"delivered"})
        self.assertEqual(ops.validation_terminal(f.run,f.reports,f.launcher)["job_id"],"300")


if __name__ == "__main__":
    unittest.main()
