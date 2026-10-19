"""CPU-only fake-Slurm checks: no sbatch, model, GPU, or live-run mutation."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from scripts import state_study_continuation as ops


REPO = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class FakeSlurm:
    def __init__(self, repo):
        self.repo = repo
        self.rows = {}
        self.submit_calls = []
        self.query_calls = []
        self.submit_code = 0
        self.submit_output = "200;local\n"
        self.create_on_submit = True
        self.child_changes = {}
        self.current_changes = {}

    def record(self, job, state, name, dependency, changes=None):
        values = {
            "JobId": job, "JobName": name, "UserId": f"jgu7({os.getuid()})",
            "JobState": state, "Partition": "contrib-B200", "Account": "kerenzhou",
            "QOS": "kzhou6", "Requeue": "0", "NumCPUs": "8", "CPUs/Task": "8",
            "NumNodes": "1", "BatchFlag": "1", "TimeLimit": "1-00:00:00",
            "ReqTRES": "cpu=8,mem=128G,node=1,billing=120,gres/gpu=1,gres/gpu:b200.180gb=1",
            "Dependency": dependency,
            "Command": str(self.repo / "scripts/state_study_resume.sbatch"),
            "WorkDir": str(self.repo),
        }
        values.update(changes or {})
        return " ".join(f"{key}={value}" for key, value in values.items()) + "\n"

    def query(self, command):
        self.query_calls.append(command)
        if command[0] == "squeue":
            output = "".join("|".join((job, *row[:3])) + "\n" for job, row in self.rows.items())
        elif command == ["scontrol", "-o", "show", "job", "100"]:
            output = self.record("100", "RUNNING", "moe-state-study", "(null)", self.current_changes)
        elif command[0] == "scontrol" and command[-1] in self.rows:
            job = command[-1]
            name, state, dependency, changes = self.rows[job]
            output = self.record(job, state, name, dependency, changes)
        else:
            return {"command": command, "returncode": 1, "stdout": "", "stderr": "unknown job"}
        return {"command": command, "returncode": 0, "stdout": output, "stderr": ""}

    def submit(self, command):
        self.submit_calls.append(command)
        name = next(part.split("=", 1)[1] for part in command if part.startswith("--job-name="))
        dependency = next(part.split("=", 1)[1] for part in command if part.startswith("--dependency="))
        if self.create_on_submit:
            self.rows["200"] = (name, "PENDING", dependency + "(unfulfilled)", self.child_changes)
        return {"command": command, "returncode": self.submit_code,
                "stdout": self.submit_output, "stderr": "simulated timeout" if self.submit_code is None else ""}


class ContinuationFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/scratch/jgu7/tmp")
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        self.run = self.root / "runs/study"
        self.reports = self.root / "reports"
        for index in range(15):
            path = self.repo / f"moe_research/m{index}.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# frozen toy module {index}\n")
        for name in ("scripts/_harness.py", "scripts/run_state_calibration_study.py",
                     "configs/state_calibration_study.yaml", "scripts/state_study_continuation.py",
                     "scripts/launch_state_study_resume.py", "scripts/state_study_resume.sbatch"):
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# tiny CPU source fixture\n")
        frozen_files = [*self.repo.glob("moe_research/*.py"), self.repo / "scripts/_harness.py",
                        self.repo / "scripts/run_state_calibration_study.py",
                        self.repo / "configs/state_calibration_study.yaml"]
        frozen = {str(path.relative_to(self.repo)): ops.checksum(path.read_bytes()) for path in frozen_files}
        self.protocol = {"source_hash": ops.canonical(frozen)}
        self.protocol["protocol_hash"] = ops.canonical(self.protocol)
        write_json(self.run / "source_hashes.json", frozen)
        write_json(self.run / "protocol.json", self.protocol)
        self.manifest = {"status": "running", "protocol_hash": self.protocol["protocol_hash"], "stages": {}}
        write_json(self.run / "manifest.json", self.manifest)
        self.history = self.run / "launch_history/previous.json"
        self.commit = self.run / "launch_history/boundaries/previous/complete.json"
        write_json(self.history, {"previous": True})
        write_json(self.commit, {"committed": True})
        self.command = [sys.executable, "-u", str(self.repo / "scripts/run_state_calibration_study.py"),
                        "auto", "--run-dir", str(self.run)]
        self.launch = {"pid": os.getpid(), "slurm_job_id": "100", "command": self.command,
                       "metadata_written_before_exec": True, "exec_preserves_pid": True,
                       "source_hash": self.protocol["source_hash"],
                       "protocol_hash": self.protocol["protocol_hash"],
                       "previous_slurm_job_id": "99", "previous_launch_history": str(self.history),
                       "predecessor_terminal_proof": {"status": "VERIFIED_TERMINAL", "state": "TIMEOUT"},
                       "current_allocation_proof": {"job_id": "100"},
                       "predecessor_snapshot": {"commit_path": str(self.commit)}}
        write_json(self.run / "launch.json", self.launch)
        self.slurm = FakeSlurm(self.repo)

    def tearDown(self):
        self.tmp.cleanup()

    def ensure(self):
        return ops.ensure_successor(self.run, "100", repo=self.repo, reports=self.reports,
                                    query_fn=self.slurm.query, submit_fn=self.slurm.submit)

    @property
    def directory(self):
        return self.reports / self.run.name / "continuation/100"


class ContinuationTests(ContinuationFixture):
    def test_actual_returned_id_resource_dependency_and_history_are_committed(self):
        result = self.ensure()
        self.assertEqual(result["status"], "QUEUED")
        self.assertEqual(result["job_id"], "200")
        self.assertEqual(result["cluster"], "local")
        self.assertEqual(result["dependency"], "afterany:100")
        self.assertEqual(len(self.slurm.submit_calls), 1)
        command = self.slurm.submit_calls[0]
        self.assertIn("--no-requeue", command)
        self.assertIn("--gres=gpu:B200.180gb:1", command)
        self.assertEqual(command[-2:], [str(self.run), "100"])
        committed = ops.read_json(self.directory / "committed.json")
        self.assertEqual(committed["job_id"], "200")
        self.assertEqual(committed["resources"], ops.RESOURCES)
        self.assertEqual(ops.read_json(self.directory / "submission.json")["stdout"], "200;local\n")
        outcome = ops.record_outcome(self.run, "100", result, reports=self.reports)
        self.assertEqual(ops.read_json(outcome), result)

    def test_same_job_repeated_call_does_not_submit_twice(self):
        first = self.ensure()
        intent_raw = (self.directory / "intent.json").read_bytes()
        committed_raw = (self.directory / "committed.json").read_bytes()
        second = self.ensure()
        self.assertEqual(first["status"], "QUEUED")
        self.assertEqual(second["status"], "ALREADY_QUEUED")
        self.assertEqual(len(self.slurm.submit_calls), 1)
        self.assertEqual((self.directory / "intent.json").read_bytes(), intent_raw)
        self.assertEqual((self.directory / "committed.json").read_bytes(), committed_raw)

    def test_unknown_submission_without_job_is_never_retried(self):
        self.slurm.submit_code = None
        self.slurm.submit_output = ""
        self.slurm.create_on_submit = False
        first = self.ensure()
        second = self.ensure()
        self.assertEqual(first["status"], "SUBMISSION_UNKNOWN")
        self.assertEqual(second["status"], "SUBMISSION_UNKNOWN")
        self.assertEqual(len(self.slurm.submit_calls), 1)
        self.assertFalse((self.directory / "committed.json").exists())

    def test_uncertain_submission_adopted_only_from_exact_controller_identity(self):
        self.slurm.submit_code = None
        self.slurm.submit_output = ""
        first = self.ensure()
        second = self.ensure()
        self.assertEqual(first["status"], "SUBMISSION_UNKNOWN")
        self.assertEqual(second["status"], "QUEUED")
        self.assertTrue(second["adopted_existing"])
        self.assertEqual(second["job_id"], "200")
        self.assertEqual(len(self.slurm.submit_calls), 1)

    def test_explicit_errors_even_hidden_by_running_status_do_not_claim_or_submit(self):
        for location in ("manifest", "stage", "preflight"):
            with self.subTest(location=location):
                self.manifest["status"] = "running"
                self.manifest["stages"] = {}
                (self.run / "preflight.json").unlink(missing_ok=True)
                if location == "manifest":
                    self.manifest["status"] = "error"
                elif location == "stage":
                    self.manifest["stages"]["P0"] = {"status": "FAILED_IMPLEMENTATION"}
                else:
                    write_json(self.run / "preflight.json", {"status": "FAILED_IMPLEMENTATION"})
                write_json(self.run / "manifest.json", self.manifest)
                self.assertEqual(self.ensure()["status"], "SCIENTIFIC_STOP")
                self.assertFalse(self.directory.exists())
                self.assertEqual(self.slurm.submit_calls, [])

    def test_core_completion_skips_even_if_scientific_disposition_inconclusive(self):
        self.manifest["stages"] = {"P5": {"status": "complete"},
                                    "P6_P7": {"status": "gated_pending"}}
        write_json(self.run / "manifest.json", self.manifest)
        (self.run / "RUN_REPORT.md").write_text("INCONCLUSIVE: complete core evidence\n")
        with mock.patch("scripts.state_study_supplement_ops.supplement_state",
                        return_value={"status": "NOT_REQUIRED", "delivery_complete": True}), \
             mock.patch("scripts.state_study_supplement_ops.observer_cost_state",
                        return_value={"status": "COMPLETE", "delivery_complete": True}):
            self.assertEqual(self.ensure()["status"], "ALREADY_COMPLETE_NO_SUBMISSION")
        self.assertFalse(self.directory.exists())
        self.assertEqual(self.slurm.submit_calls, [])

    def test_foreign_launch_pid_or_changed_source_never_submits(self):
        self.launch["pid"] = os.getpid() + 1
        write_json(self.run / "launch.json", self.launch)
        self.assertEqual(self.ensure()["status"], "REFUSED")
        self.launch["pid"] = os.getpid()
        write_json(self.run / "launch.json", self.launch)
        (self.repo / "moe_research/m0.py").write_text("changed\n")
        self.assertEqual(self.ensure()["status"], "REFUSED")
        self.assertFalse(self.directory.exists())
        self.assertEqual(self.slurm.submit_calls, [])

    def test_wrong_controller_resources_owner_or_dependency_are_not_accepted(self):
        changes = ({"Dependency": "afterany:999"}, {"NumCPUs": "16"}, {"Requeue": "1"},
                   {"UserId": f"other({os.getuid()+1})"}, {"Partition": "different"},
                   {"ReqTRES": "cpu=8,mem=128G,node=1,gres/gpu=1,gres/gpu:a100=1"})
        for index, change in enumerate(changes):
            with self.subTest(change=change):
                # New isolated current-job state for each actual fake submit.
                reports = self.reports / str(index)
                slurm = FakeSlurm(self.repo)
                slurm.child_changes = change
                result = ops.ensure_successor(self.run, "100", repo=self.repo, reports=reports,
                                               query_fn=slurm.query, submit_fn=slurm.submit)
                self.assertEqual(result["status"], "SUBMISSION_UNKNOWN")
                self.assertFalse((reports / self.run.name / "continuation/100/committed.json").exists())
                again = ops.ensure_successor(self.run, "100", repo=self.repo, reports=reports,
                                              query_fn=slurm.query, submit_fn=slurm.submit)
                self.assertNotIn(again["status"], ("QUEUED", "ALREADY_QUEUED"))
                self.assertEqual(len(slurm.submit_calls), 1)

    def test_existing_different_successor_blocks_another_submission(self):
        write_json(self.run / "slurm_continuation.json", {"job_id": "777"})
        for state in ("PENDING", "REQUEUE_HOLD"):
            with self.subTest(state=state):
                self.slurm.rows["777"] = ("moe-state-study", state, "afterany:50", {})
                result = self.ensure()
                self.assertEqual(result["status"], "REFUSED")
                self.assertEqual(self.slurm.submit_calls, [])
                self.assertFalse((self.directory / "intent.json").exists())

    def test_current_allocation_resource_error_is_rejected_before_intent(self):
        self.slurm.current_changes = {"Requeue": "1"}
        self.assertEqual(self.ensure()["status"], "REFUSED")
        self.assertFalse(self.directory.exists())
        self.assertEqual(self.slurm.submit_calls, [])

    def test_predecessor_known_scientific_exit_or_ambiguous_failure_stops_chain(self):
        path = self.reports / self.run.name / "batch-exit-codes-99.json"
        write_json(path, {"scientific_exit_code": 1})
        proof = {"state": "TIMEOUT"}
        self.assertIsNotNone(ops.predecessor_stop(self.run, "99", proof, reports=self.reports))
        path.unlink()
        self.assertIsNone(ops.predecessor_stop(self.run, "99", proof, reports=self.reports))
        for state in ("FAILED", "CANCELLED", "OUT_OF_MEMORY"):
            self.assertIsNotNone(ops.predecessor_stop(self.run, "99", {"state": state}, reports=self.reports))


class LauncherGuardTests(ContinuationFixture):
    """Exercise the real entrypoint with fake scheduler/process/model boundaries."""
    def setUp(self):
        super().setUp()
        with mock.patch.object(sys, "path", [str(REPO / "scripts"), *sys.path]):
            spec = importlib.util.spec_from_file_location("resume_ops_test_launcher", REPO / "scripts/launch_state_study_resume.py")
            self.launcher = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.launcher)

    def launch_main(self, manifest=None, live=True):
        module = self.launcher
        proof = {"file_count": 18, **self.protocol}
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(module, "RUNS", self.run.parent))
            stack.enter_context(mock.patch.object(module, "validate_sources", return_value=(proof, manifest or self.manifest)))
            stack.enter_context(mock.patch.object(module, "original_command", return_value=self.command))
            stack.enter_context(mock.patch.object(module, "dispatch_supplement", return_value=0))
            stack.enter_context(mock.patch.object(module, "verify_launch_process", return_value={"status": "VERIFIED_LIVE" if live else "PROCESS_GONE"}))
            queue = stack.enter_context(mock.patch.object(module, "ensure_successor"))
            claim = stack.enter_context(mock.patch.object(module, "archive_launch_once"))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            code = module.main(["--run-dir", str(self.run), "--previous-job-id", "100"])
        return code, queue, claim

    def test_live_predecessor_refuses_before_any_claim_or_ops_action(self):
        before = (self.run / "launch.json").read_bytes()
        code, queue, claim = self.launch_main(live=True)
        self.assertEqual(code, 3)
        queue.assert_not_called()
        claim.assert_not_called()
        self.assertEqual((self.run / "launch.json").read_bytes(), before)

    def test_launcher_scientific_error_or_complete_returns_before_ops_claim(self):
        broken = dict(self.manifest, stages={"auto": {"status": "error"}})
        code, queue, claim = self.launch_main(manifest=broken)
        self.assertEqual(code, 3)
        queue.assert_not_called()
        claim.assert_not_called()
        done = dict(self.manifest, stages={"P5": {"status": "complete"}, "P6_P7": {"status": "gated_pending"}})
        (self.run / "RUN_REPORT.md").write_text("complete\n")
        code, queue, claim = self.launch_main(manifest=done)
        self.assertEqual(code, 0)
        queue.assert_not_called()
        claim.assert_not_called()

    def test_wrong_new_allocation_resources_refuse_before_handoff_claim(self):
        module = self.launcher
        proof = {"file_count": 18, **self.protocol}
        allocation = {"job_id": "200", "scontrol": {"returncode": 0,
            "stdout": self.slurm.record("200", "RUNNING", "moe-state-study", "(null)", {"Requeue": "1"})}}
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(module, "RUNS", self.run.parent))
            stack.enter_context(mock.patch.object(module, "validate_sources", return_value=(proof, self.manifest)))
            stack.enter_context(mock.patch.object(module, "original_command", return_value=self.command))
            stack.enter_context(mock.patch.object(module, "verify_launch_process", return_value={"status": "PROCESS_GONE"}))
            stack.enter_context(mock.patch.object(module, "predecessor_terminal", return_value={"status": "VERIFIED_TERMINAL", "state": "TIMEOUT"}))
            stack.enter_context(mock.patch.object(module, "current_allocation", return_value=allocation))
            claim = stack.enter_context(mock.patch.object(module, "archive_launch_once"))
            queue = stack.enter_context(mock.patch.object(module, "ensure_successor"))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            code = module.main(["--run-dir", str(self.run), "--previous-job-id", "100"])
        self.assertEqual(code, 3)
        claim.assert_not_called()
        queue.assert_not_called()

    def test_valid_handoff_keeps_claim_boundary_pid_and_exec_despite_queue_failure(self):
        module = self.launcher
        proof = {"file_count": 18, **self.protocol}
        old_raw = (self.run / "launch.json").read_bytes()
        terminal = {"status": "VERIFIED_TERMINAL", "state": "TIMEOUT",
                    "end": "2026-10-20T03:05:52"}
        allocation = {"job_id": "200", "cgroup": "/job_200/step_batch",
                      "scontrol": {"returncode": 0, "stdout": self.slurm.record(
                          "200", "RUNNING", "moe-state-study", "(null)"), "stderr": ""}}
        def queue_after_handoff(run, job):
            self.assertEqual(job, "200")
            current = ops.read_json(run / "launch.json")
            self.assertEqual(current["pid"], os.getpid())
            self.assertEqual(current["previous_slurm_job_id"], "100")
            self.assertEqual(Path(current["previous_launch_history"]).read_bytes(), old_raw)
            self.assertTrue(Path(current["predecessor_snapshot"]["commit_path"]).is_file())
            return {"status": "SUBMISSION_UNKNOWN", "reason": "fake RPC timeout"}
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(module, "REPO", self.repo))
            stack.enter_context(mock.patch.object(module, "RUNS", self.run.parent))
            stack.enter_context(mock.patch.object(module, "validate_sources", return_value=(proof, self.manifest)))
            stack.enter_context(mock.patch.object(module, "original_command", return_value=self.command))
            stack.enter_context(mock.patch.object(module, "verify_launch_process", return_value={"status": "PROCESS_GONE"}))
            stack.enter_context(mock.patch.object(module, "predecessor_terminal", return_value=terminal))
            stack.enter_context(mock.patch.object(module, "current_allocation", return_value=allocation))
            queue = stack.enter_context(mock.patch.object(module, "ensure_successor", side_effect=queue_after_handoff))
            watch_start = stack.enter_context(mock.patch.object(module, "launch_watch_start", side_effect=OSError("fake watch start failure")))
            stack.enter_context(mock.patch.object(module, "record_outcome", side_effect=lambda run, job, result: ops.record_outcome(run, job, result, reports=self.reports)))
            stack.enter_context(mock.patch.object(module.os, "chdir"))
            execute = stack.enter_context(mock.patch.object(module.os, "execvpe", side_effect=OSError("fake exec boundary")))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            code = module.main(["--run-dir", str(self.run), "--previous-job-id", "100",
                                "--log", "/scratch/jgu7/moe-state-study/slurm/fake-200.log"])
        self.assertEqual(code, 3)  # Only the deliberately simulated exec failed.
        queue.assert_called_once()
        watch_start.assert_called_once_with(self.run)
        execute.assert_called_once()
        self.assertEqual(execute.call_args.args[:2], (self.command[0], self.command))
        self.assertTrue((self.run / "launch_exec_failure.200.json").is_file())
        outcomes = list((self.reports / self.run.name / "continuation/200").glob("outcome-*.json"))
        self.assertEqual(ops.read_json(outcomes[0])["status"], "SUBMISSION_UNKNOWN")


if __name__ == "__main__":
    unittest.main()
