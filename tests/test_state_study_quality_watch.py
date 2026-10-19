"""Fake-freezer/PID CPU tests; never starts a real supervisor or seals a plan."""
from pathlib import Path
import contextlib
import io
import json
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts import state_study_quality_watch as watch


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_proc(root, pid, command, *, ticks="123", group="0::/jobs/job_100/step_batch", state="S"):
    directory = root / str(pid)
    directory.mkdir(parents=True, exist_ok=True)
    fields = [state, *(["0"] * 18), str(ticks), "0", "0"]
    (directory / "stat").write_text(f"{pid} (fake main) " + " ".join(fields))
    (directory / "cmdline").write_bytes(b"\0".join(part.encode() for part in command) + b"\0")
    (directory / "cgroup").write_text(group + "\n")


class QualityWatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/scratch/jgu7/tmp")
        self.local_tmp = tempfile.TemporaryDirectory(dir="/raid/jgu7")
        self.root = Path(self.tmp.name)
        self.run = self.root / "run"
        self.reports = self.root / "reports"
        self.control = Path(self.local_tmp.name)
        self.manifest = {"status": "running", "stages": {"P2": {"status": "complete"}}}
        write_json(self.run / "manifest.json", self.manifest)
        self.final = self.reports / self.run.name / "quality_plan.json"
        self.draft_path = self.final.with_name("quality_plan.draft.json")
        self.draft = {"plan_hash": "draft-fixed", "protocol_hash": "protocol-fixed"}
        write_json(self.draft_path, self.draft)
        self.command = [sys.executable, "-u", str(watch.REPO / "scripts/run_state_calibration_study.py"),
                        "auto", "--run-dir", str(self.run)]
        self.launch = {"hostname": "node", "boot_id": "boot-test", "pid": 100,
                       "proc_start_ticks": "123", "slurm_job_id": "100", "command": self.command}
        write_json(self.run / "launch.json", self.launch)
        self.binding = {"run_dir": str(self.run), "launch": self.launch,
                        "launch_identity": watch._identity(self.launch),
                        "draft_hash": "draft-fixed", "draft_path": str(self.draft_path),
                        "final_path": str(self.final)}
        self.proc = self.root / "proc"
        (self.proc / "sys/kernel/random").mkdir(parents=True)
        (self.proc / "sys/kernel/random/boot_id").write_text("boot-test\n")
        (self.proc / "self").mkdir()
        (self.proc / "self/cgroup").write_text("0::/jobs/job_100/step_batch\n")
        write_proc(self.proc, 100, self.command)
        self.held_descriptors = []
        self.spawn_calls = []
        self.events = []

    def tearDown(self):
        for descriptor in self.held_descriptors:
            os.close(descriptor)
        self.local_tmp.cleanup()
        self.tmp.cleanup()

    def inspect(self, run, binding):
        return {"status": "VERIFIED_MAIN_AUTO"}

    def verify(self, run, final):
        self.assertEqual(final, self.final)
        return {"status": "VERIFIED_PROSPECTIVE_PLAN", "plan_hash": "final-fixed"}

    def freeze_success(self, run, draft, final):
        self.assertEqual(draft, self.draft_path)
        write_json(final, {"already_published_by_fake_freezer": True})
        return {"exit_code": 0, "stdout": "FROZEN_PROSPECTIVE\n", "stderr": ""}

    def monitor(self, **kwargs):
        arguments = {"inspect_fn": self.inspect, "freeze_fn": self.freeze_success,
                     "verify_fn": self.verify, "emit": self.events.append,
                     "sleep_fn": lambda seconds: self.fail("unexpected sleep")}
        arguments.update(kwargs)
        return watch.monitor(self.binding, **arguments)

    def fake_spawn(self, command, **kwargs):
        self.spawn_calls.append((command, kwargs))
        self.held_descriptors.append(os.dup(kwargs["pass_fds"][0]))
        write_proc(self.proc, 900, command, ticks="999")
        return SimpleNamespace(pid=900)

    def start(self, **kwargs):
        arguments = {"reports": self.reports, "control": self.control,
                     "draft_fn": lambda run, reports: (self.draft, self.draft_path),
                     "inspect_fn": self.inspect, "spawn_fn": self.fake_spawn,
                     "proc_root": self.proc, "hostname": "node"}
        arguments.update(kwargs)
        return watch.start_watch(self.run, **arguments)

    def test_pending_p2_never_invokes_freezer_and_then_seals_once(self):
        self.manifest["stages"]["P2"]["status"] = "running"
        write_json(self.run / "manifest.json", self.manifest)
        sleeps = []
        def finish_p2(seconds):
            sleeps.append(seconds)
            self.manifest["stages"]["P2"]["status"] = "complete"
            write_json(self.run / "manifest.json", self.manifest)
        freeze = mock.Mock(side_effect=self.freeze_success)
        result = self.monitor(sleep_fn=finish_p2, freeze_fn=freeze)
        self.assertEqual(result["status"], "FINAL_VERIFIED")
        self.assertEqual(sleeps, [30])
        freeze.assert_called_once()

    def test_freezer_pending_is_waiting_then_success_not_terminal_error(self):
        count, sleeps = [], []
        def freezer(run, draft, final):
            count.append(True)
            return ({"exit_code": 2, "stdout": "PENDING\n", "stderr": ""}
                    if len(count) == 1 else self.freeze_success(run, draft, final))
        result = self.monitor(freeze_fn=freezer, sleep_fn=sleeps.append)
        self.assertEqual(result["status"], "FINAL_VERIFIED")
        self.assertEqual(len(count), 2)
        self.assertEqual(sleeps, [30])
        self.assertIn("WAITING_FREEZER_PENDING", [event["status"] for event in self.events])

    def test_exact_helper_can_wait_without_timeout_before_same_pid_auto(self):
        states = iter(["WAITING_MAIN_EXEC", "WAITING_MAIN_EXEC", "VERIFIED_MAIN_AUTO", "VERIFIED_MAIN_AUTO"])
        sleeps = []
        result = self.monitor(inspect_fn=lambda run, binding: {"status": next(states)}, sleep_fn=sleeps.append)
        self.assertEqual(result["status"], "FINAL_VERIFIED")
        self.assertEqual(sleeps, [30, 30])

    def test_dead_or_changed_main_never_calls_freezer(self):
        for status in ("MAIN_EXITED", "MAIN_IDENTITY_CHANGED"):
            freeze = mock.Mock()
            result = self.monitor(inspect_fn=lambda run, binding: {"status": status}, freeze_fn=freeze)
            self.assertEqual(result["status"], status)
            freeze.assert_not_called()

    def test_unknown_main_proof_waits_without_calling_freezer(self):
        states = iter(["WAITING_MAIN_PROOF", "MAIN_EXITED"])
        sleeps, freeze = [], mock.Mock()
        result = self.monitor(inspect_fn=lambda run, binding: {"status": next(states)},
                              freeze_fn=freeze, sleep_fn=sleeps.append)
        self.assertEqual(result["status"], "MAIN_EXITED")
        self.assertEqual(sleeps, [30])
        freeze.assert_not_called()

    def test_explicit_scientific_stage_error_stops_even_top_status_running(self):
        self.manifest["stages"]["auto"] = {"status": "error"}
        write_json(self.run / "manifest.json", self.manifest)
        freeze = mock.Mock()
        self.assertEqual(self.monitor(freeze_fn=freeze)["status"], "SCIENTIFIC_STOP")
        freeze.assert_not_called()
        self.assertEqual(self.start()["status"], "SCIENTIFIC_STOP")
        self.assertFalse(list(self.control.iterdir()))

    def test_existing_final_only_verified_never_rewritten(self):
        write_json(self.final, {"frozen": "original"})
        original = self.final.read_bytes()
        freeze, verify = mock.Mock(), mock.Mock(side_effect=self.verify)
        self.assertEqual(self.monitor(freeze_fn=freeze, verify_fn=verify)["status"], "FINAL_VERIFIED")
        freeze.assert_not_called()
        verify.assert_called_once()
        self.assertEqual(self.final.read_bytes(), original)

    def test_validation_output_or_freezer_invalid_never_backfills(self):
        (self.run / "formal/validate").mkdir(parents=True)
        (self.run / "formal/validate/started.json").write_text("{}")
        freeze = mock.Mock()
        self.assertEqual(self.monitor(freeze_fn=freeze)["status"], "REFUSED")
        freeze.assert_not_called()
        (self.run / "formal/validate/started.json").unlink()
        freeze = mock.Mock(return_value={"exit_code": 3, "stderr": "REFUSED invalid source"})
        self.assertEqual(self.monitor(freeze_fn=freeze)["status"], "FREEZER_REFUSED")
        freeze.assert_called_once()

    def test_no_valid_draft_or_no_verified_main_never_spawns(self):
        def no_draft(run, reports):
            raise watch.freezer.Pending("missing")
        with self.assertRaises(watch.freezer.Pending):
            self.start(draft_fn=no_draft)
        result = self.start(inspect_fn=lambda run, binding: {"status": "MAIN_IDENTITY_CHANGED"})
        self.assertEqual(result["status"], "NO_VERIFIED_MAIN_NO_START")
        self.assertEqual(self.spawn_calls, [])
        self.assertFalse(list(self.control.iterdir()))

    def test_inherited_local_lock_and_concrete_pid_prevent_duplicate_watch(self):
        first = self.start()
        self.assertEqual(first["status"], "WATCH_STARTED_VERIFIED_PID")
        self.assertEqual(first["watch"]["pid"], 900)
        self.assertEqual(self.start()["status"], "ALREADY_WATCHING")
        self.assertEqual(len(self.spawn_calls), 1)
        command, options = self.spawn_calls[0]
        self.assertIn("--run-dir", command)
        self.assertTrue(options["start_new_session"])
        self.assertEqual(len(options["pass_fds"]), 1)
        # A held lock plus stale PID metadata is UNKNOWN, not a proof of life.
        record_path = next(self.control.rglob("watch.pid.json"))
        record = json.loads(record_path.read_text())
        record["proc_start_ticks"] = "wrong"
        write_json(record_path, record)
        self.assertEqual(self.start()["status"], "LOCK_HELD_WATCH_IDENTITY_UNKNOWN")
        self.assertEqual(len(self.spawn_calls), 1)

    def test_fake_proc_identity_and_exact_cgroup_are_required(self):
        self.assertEqual(watch.inspect_main(self.run, self.binding, proc_root=self.proc,
                                           hostname="node")["status"], "VERIFIED_MAIN_AUTO")
        write_proc(self.proc, 100, self.command, ticks="124")
        self.assertEqual(watch.inspect_main(self.run, self.binding, proc_root=self.proc,
                                           hostname="node")["status"], "MAIN_IDENTITY_CHANGED")
        write_proc(self.proc, 100, self.command, group="0::/jobs/job_999/step_batch")
        self.assertEqual(watch.inspect_main(self.run, self.binding, proc_root=self.proc,
                                           hostname="node")["status"], "MAIN_IDENTITY_CHANGED")

    def test_only_precise_trusted_helper_can_transition_to_auto(self):
        self.launch.update(metadata_written_before_exec=True, previous_slurm_job_id="99",
                           log="/scratch/jgu7/moe-state-study/slurm/test.log")
        write_json(self.run / "launch.json", self.launch)
        self.binding["launch_identity"] = watch._identity(self.launch)
        helper = watch._helper_command(self.launch)[0]
        write_proc(self.proc, 100, helper)
        self.assertEqual(watch.inspect_main(self.run, self.binding, proc_root=self.proc,
                                           hostname="node")["status"], "WAITING_MAIN_EXEC")
        relative_helper = watch._helper_command(self.launch)[2]
        write_proc(self.proc, 100, relative_helper)
        (self.proc / "100/cwd").symlink_to(watch.REPO, target_is_directory=True)
        self.assertEqual(watch.inspect_main(self.run, self.binding, proc_root=self.proc,
                                           hostname="node")["status"], "WAITING_MAIN_EXEC")
        (self.proc / "100/cwd").unlink()
        (self.proc / "100/cwd").symlink_to(self.run, target_is_directory=True)
        self.assertEqual(watch.inspect_main(self.run, self.binding, proc_root=self.proc,
                                           hostname="node")["status"], "MAIN_IDENTITY_CHANGED")
        write_proc(self.proc, 100, self.command)
        self.assertEqual(watch.inspect_main(self.run, self.binding, proc_root=self.proc,
                                           hostname="node")["status"], "VERIFIED_MAIN_AUTO")
        write_proc(self.proc, 100, [*helper[:-1], "wrong-log"])
        self.assertEqual(watch.inspect_main(self.run, self.binding, proc_root=self.proc,
                                           hostname="node")["status"], "MAIN_IDENTITY_CHANGED")

    def test_observation_oserror_waits_then_same_main_can_seal(self):
        inspect = mock.Mock(side_effect=[OSError("momentary proc read"),
            {"status": "VERIFIED_MAIN_AUTO"}, {"status": "VERIFIED_MAIN_AUTO"}])
        sleeps = []
        result = self.monitor(inspect_fn=inspect, sleep_fn=sleeps.append)
        self.assertEqual(result["status"], "FINAL_VERIFIED")
        self.assertEqual(sleeps, [30])
        self.assertIn("WAITING_OBSERVATION", [event["status"] for event in self.events])

    def test_manifest_oserror_waits_instead_of_losing_supervisor(self):
        original, reads = watch.read_json, []
        def read(path):
            if Path(path) == self.run / "manifest.json":
                reads.append(True)
                if len(reads) == 1:
                    raise OSError("momentary manifest read")
            return original(path)
        sleeps = []
        with mock.patch.object(watch, "read_json", side_effect=read):
            result = self.monitor(sleep_fn=sleeps.append)
        self.assertEqual(result["status"], "FINAL_VERIFIED")
        self.assertEqual(sleeps, [30])

    def test_publication_then_io_error_verifies_final_without_second_call(self):
        def publish_then_io(run, draft, final):
            self.freeze_success(run, draft, final)
            raise OSError("publication outcome interrupted")
        freeze, sleeps = mock.Mock(side_effect=publish_then_io), []
        result = self.monitor(freeze_fn=freeze, sleep_fn=sleeps.append)
        self.assertEqual(result["status"], "FINAL_VERIFIED")
        freeze.assert_called_once()
        self.assertEqual(sleeps, [30])

    def test_cli_three_after_publication_still_verifies_existing_final(self):
        def published_then_refused(run, draft, final):
            self.freeze_success(run, draft, final)
            return {"exit_code": 3, "stderr": "publication fsync failed"}
        freeze = mock.Mock(side_effect=published_then_refused)
        result = self.monitor(freeze_fn=freeze)
        self.assertEqual(result["status"], "FINAL_VERIFIED")
        freeze.assert_called_once()

    def test_initial_draft_io_or_pending_recovers_with_same_bound_main(self):
        validate = mock.Mock(side_effect=[OSError("draft read"), watch.freezer.Pending("archive read"),
                                           (self.draft, self.draft_path)])
        sleeps = []
        result = self.monitor(validate_fn=validate, sleep_fn=sleeps.append)
        self.assertEqual(result["status"], "FINAL_VERIFIED")
        self.assertEqual(sleeps, [30, 30])
        self.assertEqual(validate.call_count, 3)

    def test_evidence_io_failure_cannot_break_wait_or_publish_without_intent(self):
        emissions, sleeps = [], []
        def emit(value):
            emissions.append(value)
            if len(emissions) == 1:
                raise OSError("scratch unavailable")
        freeze = mock.Mock(side_effect=self.freeze_success)
        with contextlib.redirect_stderr(io.StringIO()):
            result = self.monitor(emit=emit, freeze_fn=freeze, sleep_fn=sleeps.append)
        self.assertEqual(result["status"], "FINAL_VERIFIED")
        self.assertEqual(sleeps, [30])
        freeze.assert_called_once()

    def test_locked_freezer_entrypoint_invoked_without_new_process(self):
        with mock.patch.object(watch.freezer, "main", return_value=2) as cli:
            result = watch.invoke_freezer(self.run, self.draft_path, self.final)
        self.assertEqual(result["exit_code"], 2)
        cli.assert_called_once_with(["--run-dir", str(self.run), "--draft", str(self.draft_path),
                                     "--output", str(self.final)])


if __name__ == "__main__":
    unittest.main()
