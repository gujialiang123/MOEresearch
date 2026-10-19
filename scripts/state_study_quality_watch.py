#!/usr/bin/env python3
"""CPU supervisor for one concrete launch's prospective quality-plan seal.

No service, scheduler submission, model import, or analysis-rule implementation.
The existing locked freezer owns every draft/final scientific check.  Locks and
PID metadata live on node-local /raid; immutable evidence and logs use scratch.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

if __package__:
    from . import freeze_state_study_quality_plan as freezer
    from .state_study_continuation import ContinuationRefused, _publish, canonical, read_json, scientific_error
    from .summarize_state_study_results import _proc_stat, verify_launch_process
else:
    import freeze_state_study_quality_plan as freezer
    from state_study_continuation import ContinuationRefused, _publish, canonical, read_json, scientific_error
    from summarize_state_study_results import _proc_stat, verify_launch_process


REPO = Path(__file__).resolve().parents[1]
REPORTS = Path("/scratch/jgu7/moe-state-study/reports")
CONTROL = Path("/raid/jgu7/state-study-quality-watch")
IDENTITY_FIELDS = ("hostname", "boot_id", "pid", "proc_start_ticks", "slurm_job_id", "command")


def utc():
    return datetime.now(timezone.utc).isoformat()


def validate_draft(run, reports=REPORTS):
    """Reuse locked source/archive/formula checks without recreating a draft."""
    path = Path(reports) / Path(run).name / "quality_plan.draft.json"
    if not path.is_file():
        raise freezer.Pending("No prospective draft exists; no watcher started")
    draft = freezer.load(path)
    freezer.verify_self(draft, "DRAFT_NOT_FROZEN")
    freezer.verify_source_archive(draft)
    protocol, source = freezer.sources(run)
    if (draft.get("protocol_hash") != protocol["protocol_hash"] or draft.get("source") != source
            or draft.get("run_dir") != str(Path(run).resolve())
            or draft.get("formal_selection_output_absent_at_draft") is not True
            or draft.get("analysis_supplement") != freezer.supplement(protocol)):
        raise freezer.Invalid("Existing draft source/protocol/run/precedence identity differs")
    return draft, path


def verify_final(run, path):
    return freezer.verify_plan(run, freezer.load(path))


def invoke_freezer(run, draft_path, final_path):
    """Execute the locked freezer CLI's main in this one lock-owning process.

    Avoids an orphan freezer child if the watch is interrupted. Exit 2 is a
    local pending observation, not an uncertain external job submission.
    """
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = freezer.main(["--run-dir", str(run), "--draft", str(draft_path),
                             "--output", str(final_path)])
    return {"exit_code": code, "stdout": stdout.getvalue(), "stderr": stderr.getvalue()}


def _identity(launch):
    return {field: launch.get(field) for field in IDENTITY_FIELDS}


def _helper_command(launch):
    if launch.get("metadata_written_before_exec") is not True:
        return []
    previous, log = launch.get("previous_slurm_job_id"), launch.get("log")
    if not isinstance(previous, str) or not previous.isdigit() or not isinstance(log, str):
        return []
    tail = [str(REPO / "scripts/launch_state_study_resume.py"), "--run-dir", launch["command"][5],
            "--previous-job-id", previous, "--log", log]
    relative = ["scripts/launch_state_study_resume.py", *tail[1:]]
    return [[launch["command"][0], "-u", *tail], [launch["command"][0], *tail],
            [launch["command"][0], "-u", *relative], [launch["command"][0], *relative]]


def inspect_main(run, binding, *, proc_root=Path("/proc"), hostname=None):
    """Verify PID/boot/start/command and the supervisor's same Slurm cgroup."""
    launch = binding["launch"]
    current = read_json(Path(run) / "launch.json")
    if _identity(current) != binding["launch_identity"]:
        return {"status": "MAIN_IDENTITY_CHANGED", "reason": "current_launch_binding_changed"}
    proof = verify_launch_process(run, launch, proc_root=proc_root, hostname=hostname)
    main_directory = Path(proc_root) / str(launch["pid"])
    if proof["status"] != "VERIFIED_LIVE":
        if proof.get("reason") == "cmdline_does_not_match_launch_run_identity":
            if proof.get("observed_cmdline") not in _helper_command(launch):
                return {"status": "MAIN_IDENTITY_CHANGED", "proof": proof}
            command = proof["observed_cmdline"]
            script = command[2] if command[1] == "-u" else command[1]
            if not Path(script).is_absolute() and (main_directory / "cwd").resolve(strict=True) != REPO:
                return {"status": "MAIN_IDENTITY_CHANGED", "reason": "relative_helper_cwd_differs",
                        "proof": proof}
            # The locked verifier stops before its second stat on a command
            # mismatch; do that recheck for the exact trusted helper ourselves.
            after = _proc_stat((main_directory / "stat").read_text())
            if (after["pid"] != launch["pid"] or after["start_ticks"] != str(launch["proc_start_ticks"])
                    or after["state"] in ("Z", "X", "x")):
                return {"status": "MAIN_IDENTITY_CHANGED", "proof": proof}
            state = "WAITING_MAIN_EXEC"
        elif not main_directory.exists():
            return {"status": "MAIN_EXITED", "proof": proof}
        elif proof.get("reason") in {"local_proc_unreadable_or_identity_invalid"}:
            return {"status": "WAITING_MAIN_PROOF", "proof": proof}
        else:
            return {"status": "MAIN_IDENTITY_CHANGED", "proof": proof}
    else:
        state = "VERIFIED_MAIN_AUTO"
    main_group = (main_directory / "cgroup").read_text().strip()
    own_group = (Path(proc_root) / "self/cgroup").read_text().strip()
    if not main_group or main_group != own_group:
        return {"status": "MAIN_IDENTITY_CHANGED", "reason": "watch_not_in_exact_main_cgroup"}
    job = str(launch.get("slurm_job_id", ""))
    if not job.isdigit() or not any("job_" + job in part.split("/") for part in own_group.split("\n")):
        return {"status": "MAIN_IDENTITY_CHANGED", "reason": "cgroup_has_no_bound_slurm_job"}
    return {"status": state, "proof": proof, "cgroup": own_group}


def monitor(binding, *, inspect_fn=inspect_main, freeze_fn=invoke_freezer,
            verify_fn=verify_final, sleep_fn=time.sleep, emit=lambda value: None,
            validate_fn=None):
    """Wait for P2; a pending freezer stays attached to the same main launch."""
    run = Path(binding["run_dir"])
    final = Path(binding["final_path"])
    previous_status = None
    draft_validated = validate_fn is None
    def event(status, **details):
        nonlocal previous_status
        signature = canonical({"status": status, "reason": details.get("reason")})
        if signature != previous_status or status.startswith("FREEZER_"):
            try:
                emit({"status": status, "utc": utc(), **details})
            except OSError as exc:
                try:
                    print(json.dumps({"status": "WATCH_EVIDENCE_IO_WAIT", "error": str(exc),
                                      "intended_status": status}), file=sys.stderr, flush=True)
                except OSError:
                    pass
                return False
            previous_status = signature
        return True
    while True:
        try:
            main = inspect_fn(run, binding)
            if main["status"] in {"MAIN_EXITED", "MAIN_IDENTITY_CHANGED"}:
                event(main["status"], identity=main)
                return main
            manifest = read_json(run / "manifest.json")
            errors = scientific_error(manifest, run)
            if errors:
                result = {"status": "SCIENTIFIC_STOP", "errors": errors}
                event(result["status"], errors=errors)
                return result
            if not draft_validated:
                draft, _ = validate_fn(run)
                if draft["plan_hash"] != binding["draft_hash"]:
                    raise freezer.Invalid("Draft changed after watch start")
                draft_validated = True
            if final.is_file():
                proof = verify_fn(run, final)
                event("FINAL_VERIFIED", proof=proof)
                return {"status": "FINAL_VERIFIED", "proof": proof}
            freezer.validation_absence(run)
            if main["status"] != "VERIFIED_MAIN_AUTO":
                event(main["status"], identity=main)
                sleep_fn(30)
                continue
            if manifest.get("stages", {}).get("P2", {}).get("status") != "complete":
                event("WAITING_P2")
                sleep_fn(30)
                continue
            # Recheck the exact live identity and current error before the
            # potentially publishing CPU call. The freezer alone enforces all
            # complete six-arm and prevalidation-absence scientific checks.
            if inspect_fn(run, binding)["status"] != "VERIFIED_MAIN_AUTO":
                continue
            if scientific_error(read_json(run / "manifest.json"), run):
                continue
            if not event("FREEZER_CALL_INTENT", draft_hash=binding["draft_hash"]):
                sleep_fn(30)
                continue
            result = freeze_fn(run, Path(binding["draft_path"]), final)
            event("FREEZER_RESULT", result=result)
            if final.is_file():
                proof = verify_fn(run, final)
                event("FINAL_VERIFIED", proof=proof)
                return {"status": "FINAL_VERIFIED", "proof": proof}
            if result.get("exit_code") == 2:
                event("WAITING_FREEZER_PENDING")
                sleep_fn(30)
                continue
            if result.get("exit_code") != 0:
                event("FREEZER_REFUSED", result=result)
                return {"status": "FREEZER_REFUSED", "result": result}
            if not final.is_file():
                event("FREEZER_FAILED", reason="successful_cli_created_no_final_plan")
                return {"status": "FREEZER_FAILED", "result": result}
            proof = verify_fn(run, final)
            event("FINAL_VERIFIED", proof=proof)
            return {"status": "FINAL_VERIFIED", "proof": proof}
        except OSError as exc:
            event("WAITING_OBSERVATION", reason=str(exc))
            sleep_fn(30)
        except freezer.Pending as exc:
            event("WAITING_FREEZER_PENDING", reason=str(exc))
            sleep_fn(30)
        except (ValueError, KeyError, ContinuationRefused, freezer.Invalid) as exc:
            event("REFUSED", reason=str(exc))
            return {"status": "REFUSED", "reason": str(exc)}


def start_watch(run, *, reports=REPORTS, control=CONTROL, draft_fn=validate_draft,
                inspect_fn=inspect_main, spawn_fn=subprocess.Popen, proc_root=Path("/proc"),
                hostname=None):
    """Spawn only after draft/main proof; inherited flock serializes this launch."""
    run = Path(run).resolve()
    draft, draft_path = draft_fn(run, reports)
    errors = scientific_error(read_json(run / "manifest.json"), run)
    if errors:
        return {"status": "SCIENTIFIC_STOP", "errors": errors}
    final_path = Path(reports) / run.name / "quality_plan.json"
    if final_path.is_file():
        return {"status": "FINAL_VERIFIED", "proof": verify_final(run, final_path)}
    launch = read_json(run / "launch.json")
    binding_key = canonical({"run_dir": str(run), "identity": _identity(launch),
                             "draft_hash": draft["plan_hash"]})
    binding = {"schema_version": "state_study_quality_watch_binding.v1", "run_dir": str(run),
               "launch": launch, "launch_identity": _identity(launch), "binding_key": binding_key,
               "draft_path": str(draft_path), "final_path": str(final_path),
               "draft_hash": draft["plan_hash"], "protocol_hash": draft["protocol_hash"]}
    identity = inspect_fn(run, binding)
    if identity["status"] not in {"VERIFIED_MAIN_AUTO", "WAITING_MAIN_EXEC"}:
        return {"status": "NO_VERIFIED_MAIN_NO_START", "identity": identity}
    node = socket.gethostname() if hostname is None else hostname
    local = Path(control) / node / launch["boot_id"] / binding_key
    local.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = (local / "watch.lock").open("a+")
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pid_path = local / "watch.pid.json"
            record = read_json(pid_path) if pid_path.is_file() else None
            expected_prefix = [sys.executable, "-u", str(Path(__file__).resolve()), "run"]
            proof = (verify_launch_process(run, record, proc_root=proc_root, hostname=hostname)
                     if record and record.get("binding_key") == binding_key
                     and record.get("command", [])[:4] == expected_prefix else None)
            return {"status": "ALREADY_WATCHING" if proof and proof["status"] == "VERIFIED_LIVE"
                    else "LOCK_HELD_WATCH_IDENTITY_UNKNOWN", "watch_proof": proof}
        binding_path = local / "binding.json"
        if binding_path.exists():
            previous_binding = read_json(binding_path)
            if any(previous_binding.get(field) != binding[field] for field in (
                    "binding_key", "run_dir", "launch_identity", "draft_hash", "protocol_hash",
                    "draft_path", "final_path")):
                return {"status": "REFUSED", "reason": "Existing local binding differs"}
            binding = previous_binding
        else:
            _publish(binding_path, binding)
        attempt = Path(reports) / run.name / "quality_watch" / binding_key / uuid.uuid4().hex
        attempt.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "run", "--run-dir", str(run),
                   "--binding", str(binding_path), "--lock-fd", str(lock.fileno()),
                   "--attempt-dir", str(attempt)]
        with (attempt / "watch.log").open("xb") as output:
            child = spawn_fn(command, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                             pass_fds=(lock.fileno(),), start_new_session=True, close_fds=True)
        stat = _proc_stat((Path(proc_root) / str(child.pid) / "stat").read_text())
        record = {"pid": child.pid, "proc_start_ticks": stat["start_ticks"],
                  "boot_id": launch["boot_id"], "hostname": node, "command": command,
                  "slurm_job_id": launch["slurm_job_id"], "binding_key": binding_key,
                  "attempt_dir": str(attempt), "started_at_utc": utc()}
        proof = verify_launch_process(run, record, proc_root=proc_root, hostname=hostname)
        if proof["status"] != "VERIFIED_LIVE":
            return {"status": "SPAWNED_WATCH_IDENTITY_UNKNOWN", "pid": child.pid,
                    "watch_proof": proof, "attempt_dir": str(attempt)}
        # This is active local control metadata, not immutable evidence. The
        # actual record/proof is independently preserved under scratch.
        temporary = local / (".pid." + uuid.uuid4().hex)
        with temporary.open("x") as handle:
            json.dump(record, handle); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, local / "watch.pid.json")
        result = {"status": "WATCH_STARTED_VERIFIED_PID", "watch": record, "watch_proof": proof}
        _publish(attempt / "start.json", result)
        return result
    finally:
        # Do not LOCK_UN: the child holds the inherited open-file description.
        lock.close()


def launch_watch_start(run):
    """Short start command exits so the watch is reparented, never the model."""
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "start", "--run-dir", str(run)]
    result = subprocess.run(command, text=True, capture_output=True, timeout=20, check=False)
    return {"command": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("start", "run"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--binding"); parser.add_argument("--lock-fd", type=int)
    parser.add_argument("--attempt-dir")
    args = parser.parse_args(argv)
    run = Path(args.run_dir).resolve()
    if not run.is_dir() or not run.is_relative_to(Path("/scratch/jgu7/moe-state-study/runs")):
        parser.error("Only an existing scratch study run is allowed")
    try:
        if args.mode == "start":
            result = start_watch(run)
        else:
            if args.lock_fd is None or not args.binding or not args.attempt_dir:
                parser.error("run requires binding, inherited lock fd, and scratch attempt directory")
            binding_path, attempt = Path(args.binding).resolve(), Path(args.attempt_dir).resolve()
            if not binding_path.is_relative_to(CONTROL) or not attempt.is_relative_to(REPORTS / run.name / "quality_watch"):
                parser.error("Control must be node-local /raid; evidence must be scratch quality_watch")
            binding = read_json(binding_path)
            if binding.get("run_dir") != str(run):
                raise freezer.Invalid("Watch binding run differs")
            descriptor_stat, lock_stat = os.fstat(args.lock_fd), (binding_path.parent / "watch.lock").stat()
            if (descriptor_stat.st_dev, descriptor_stat.st_ino) != (lock_stat.st_dev, lock_stat.st_ino):
                raise freezer.Invalid("Inherited fd is not the bound local lock")
            fcntl.flock(args.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            def emit(value):
                _publish(attempt / ("event-" + uuid.uuid4().hex + ".json"), value)
                print(json.dumps(value, ensure_ascii=False), flush=True)
            result = monitor(binding, emit=emit, validate_fn=validate_draft)
            _publish(attempt / "outcome.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0 if result["status"] in {"FINAL_VERIFIED", "WATCH_STARTED_VERIFIED_PID", "ALREADY_WATCHING", "MAIN_EXITED"} else 3
    except freezer.Pending as exc:
        print(json.dumps({"status": "NO_VALID_DRAFT_NO_START", "reason": str(exc)}), flush=True)
        return 2
    except (OSError, ValueError, KeyError, ContinuationRefused, freezer.Invalid) as exc:
        print(json.dumps({"status": "REFUSED", "reason": str(exc)}), file=sys.stderr, flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
