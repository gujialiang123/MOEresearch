"""One-ahead Slurm scheduling for the unchanged frozen study.

Called once by the verified handoff launcher, immediately before its exec.
There is no daemon, model import, requeue, raw repair, or execution entrypoint.
Every allocation has one durable submission intent.  An uncertain submission
may only be adopted from an exact scheduler identity, never resubmitted.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid


REPO = Path(__file__).resolve().parents[1]
REPORTS = Path("/scratch/jgu7/moe-state-study/reports")
RESOURCES = {"partition": "contrib-B200", "account": "kerenzhou", "qos": "kzhou6",
             "gres": "gpu:B200.180gb:1", "cpus_per_task": 8, "memory": "128G",
             "time": "1-00:00:00", "requeue": 0}
ERROR_STATES = {"ERROR", "FAILED_IMPLEMENTATION", "FAILED", "FAIL"}


class ContinuationRefused(RuntimeError):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def checksum(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return checksum(json.dumps(value, sort_keys=True, ensure_ascii=False,
                               separators=(",", ":"), allow_nan=False).encode())


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContinuationRefused(f"Expected a JSON object: {path}")
    return value


def scientific_error(manifest, run=None):
    """Inspect current explicit errors, including stages hidden by top status."""
    errors = []
    if str(manifest.get("status", "")).upper() in ERROR_STATES:
        errors.append({"stage": "manifest", "status": manifest["status"]})
    stages = manifest.get("stages", {})
    if not isinstance(stages, dict):
        raise ContinuationRefused("manifest.stages is not an object")
    for stage, item in stages.items():
        if not isinstance(item, dict):
            raise ContinuationRefused("A manifest stage is not an object")
        if str(item.get("status", "")).upper() in ERROR_STATES:
            errors.append({"stage": stage, "status": item["status"],
                           "detail": item.get("detail")})
    if run is not None:
        preflight = Path(run) / "preflight.json"
        if preflight.is_file():
            status = read_json(preflight).get("status")
            if str(status).upper() in ERROR_STATES:
                errors.append({"stage": "current_preflight", "status": status,
                               "path": str(preflight)})
    return errors


def core_completed(run, manifest):
    stages = manifest.get("stages", {})
    report = Path(run) / "RUN_REPORT.md"
    return (stages.get("P5", {}).get("status") == "complete"
            and stages.get("P6_P7", {}).get("status") == "gated_pending"
            and report.is_file() and report.stat().st_size > 0)


def predecessor_stop(run, job_id, terminal_proof, *, reports=REPORTS):
    """Do not turn a scientific failure or an unknown failed job into a loop."""
    codes = Path(reports) / Path(run).name / f"batch-exit-codes-{job_id}.json"
    if codes.exists():
        value = read_json(codes).get("scientific_exit_code")
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            return {"reason": "predecessor_scientific_exit_not_zero", "path": str(codes),
                    "scientific_exit_code": value}
    # Infrastructure interruptions leave durable progress and may continue.
    # FAILED/OOM or explicit cancellation without a scientific exit proof are
    # ambiguous; stop for review instead of allocating identical endless jobs.
    if terminal_proof.get("state") in {"FAILED", "OUT_OF_MEMORY", "CANCELLED"}:
        return {"reason": "predecessor_failure_or_cancellation_requires_review",
                "state": terminal_proof.get("state")}
    return None


def query(command):
    record = {"command": command, "checked_at_utc": utc()}
    try:
        result = subprocess.run(command, text=True, capture_output=True,
                                timeout=20, check=False)
        record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        record.update(returncode=None, error=str(exc))
    return record


def _record(response, job_id):
    if response.get("returncode") != 0:
        raise ContinuationRefused("Scheduler lookup failed or is unknown")
    lines = [line for line in response.get("stdout", "").split("\n")
             if re.search(r"(?:^|\s)JobId=" + re.escape(str(job_id)) + r"(?:\s|$)", line)]
    if len(lines) != 1:
        raise ContinuationRefused("Exact scheduler job record is missing or ambiguous")
    return dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_:/-]*)=([^\s]+)", lines[0]))


def _memory_mib(value):
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGT]?)", value)
    if not match:
        raise ContinuationRefused("Unknown Slurm memory representation")
    return Decimal(match[1]) * {"K": Decimal(1)/1024, "": 1, "M": 1,
                                "G": 1024, "T": 1024**2}[match[2]]


def _verify_resources(record, *, state, name=None, dependency=None, repo=REPO):
    owner = re.search(r"\(([0-9]+)\)$", record.get("UserId", ""))
    if not owner or int(owner.group(1)) != os.getuid():
        raise ContinuationRefused("Scheduler job owner differs from current user")
    expected = {"JobState": state, "Partition": RESOURCES["partition"],
                "Account": RESOURCES["account"], "QOS": RESOURCES["qos"],
                "Requeue": "0", "NumCPUs": "8", "CPUs/Task": "8", "NumNodes": "1"}
    for key, value in expected.items():
        if record.get(key) != value:
            raise ContinuationRefused(f"Scheduler {key} differs: {record.get(key)!r}")
    if (not record.get("BatchFlag", "").isdigit() or int(record["BatchFlag"]) < 1
            or record.get("TimeLimit") not in {"1-00:00:00", "24:00:00"}):
        raise ContinuationRefused("Job must be a batch allocation with exactly 24h time limit")
    tres = dict(part.split("=", 1) for part in record.get("ReqTRES", "").split(",") if "=" in part)
    gpu_tres = {key.lower(): value for key, value in tres.items() if key.startswith("gres/gpu")}
    if (tres.get("cpu") != "8" or tres.get("node") != "1"
            or _memory_mib(tres.get("mem", "")) != 128 * 1024
            or gpu_tres != {"gres/gpu": "1", "gres/gpu:b200.180gb": "1"}):
        raise ContinuationRefused("Job resources differ from 1 B200/8 CPU/128G")
    if name is not None and record.get("JobName") != name:
        raise ContinuationRefused("Successor job name differs from unique run/parent identity")
    if dependency is not None and record.get("Dependency") not in {
            dependency, dependency + "(unfulfilled)"}:
        raise ContinuationRefused("Successor dependency is not the exact current allocation")
    if name is not None:
        if record.get("Command") != str(Path(repo) / "scripts/state_study_resume.sbatch"):
            raise ContinuationRefused("Successor command differs from the resume batch script")
        if record.get("WorkDir") != str(Path(repo).resolve()):
            raise ContinuationRefused("Successor working directory differs")


def _source_context(run, repo):
    frozen = read_json(run / "source_hashes.json")
    sources = [*repo.glob("moe_research/*.py"), repo / "scripts/_harness.py",
               repo / "scripts/run_state_calibration_study.py", repo / "configs/state_calibration_study.yaml"]
    actual = {str(path.relative_to(repo)): checksum(path.read_bytes()) for path in sources}
    if len(frozen) != 18 or actual != frozen:
        raise ContinuationRefused("Frozen source inventory/bytes changed")
    protocol = read_json(run / "protocol.json")
    if protocol.get("source_hash") != canonical(frozen):
        raise ContinuationRefused("Protocol/source hash mismatch")
    if protocol.get("protocol_hash") != canonical({k: v for k, v in protocol.items() if k != "protocol_hash"}):
        raise ContinuationRefused("Protocol self hash mismatch")
    return protocol


def validate_allocation_resources(allocation, *, repo=REPO):
    """Check the original launcher's controller proof before any handoff claim."""
    job = str(allocation.get("job_id", ""))
    _verify_resources(_record(allocation.get("scontrol", {}), job),
                      state="RUNNING", repo=repo)


def _publish(path, value):
    """Exclusive, fsynced publication; a surviving intent is never overwritten."""
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".pending")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _active_successors(query_fn, prefix, current_job, legacy_job):
    response = query_fn(["squeue", "--noheader", f"--user={os.getuid()}",
                         "--format=%i|%.128j|%T|%E"])
    if response.get("returncode") != 0:
        raise ContinuationRefused("Cannot establish the active successor inventory")
    matches = []
    for line in response.get("stdout", "").split("\n"):
        if not line.strip():
            continue
        fields = [field.strip() for field in line.strip().split("|")]
        if len(fields) != 4 or not fields[0].isdigit():
            raise ContinuationRefused("Unrecognized scheduler inventory row")
        job, name, state, dependency = fields
        if job == current_job:
            continue
        # squeue's active inventory is authoritative. An unfamiliar state is
        # a blocker, never evidence that a matching successor does not exist.
        if name.startswith(prefix) or job == legacy_job:
            matches.append({"job_id": job, "name": name, "state": state, "dependency": dependency})
    if len(matches) > 1:
        raise ContinuationRefused("Multiple active successors exist; no further submission")
    return matches, response


def ensure_successor(run, current_job_id, *, repo=REPO, reports=REPORTS,
                     query_fn=query, submit_fn=query, phase="core"):
    """Submit/adopt at most one successor; all ambiguous cases remain explicit.

    Only the same-PID post-handoff launcher may call this.  Return values never
    imply QUEUED unless an exact PENDING controller record has been verified.
    Errors are reported to the caller; they must not kill its scientific auto.
    No CLI invokes this function and no submit occurs on module import.
    """
    run, repo = Path(run).resolve(), Path(repo).resolve()
    current_job = str(current_job_id)
    directory = None
    result = {"schema_version": "state_study_continuation.v1", "checked_at_utc": utc(),
              "run_dir": str(run), "current_job_id": current_job, "phase": phase,
              "status": "REFUSED"}
    try:
        manifest = read_json(run / "manifest.json")
        errors = scientific_error(manifest, run)
        if errors:
            return {**result, "status": "SCIENTIFIC_STOP", "errors": errors}
        if phase not in {"core", "validation512", "observer_cost"}:
            raise ContinuationRefused("Unknown continuation phase")
        supplement = None
        resolution = None
        if phase != "core" or core_completed(run, manifest):
            if __package__:
                from . import state_study_supplement_ops as phase_ops
            else:
                import state_study_supplement_ops as phase_ops
            resolution = phase_ops.required_phase_state(run)
            supplement = resolution["validation512"]
            active = resolution["state"]
            if phase_ops.overall_done(resolution):
                return {**result, "status": "ALREADY_COMPLETE_NO_SUBMISSION",
                        "required_phases": resolution}
            if supplement["status"] == "NOT_REQUIRED" and not phase_ops.phase_done(supplement):
                return {**result, "status": "SUPPLEMENT_DELIVERY_REQUIRED", "supplement": supplement}
            if active["status"] == "SCIENTIFIC_ERROR":
                return {**result, "status": "SCIENTIFIC_STOP", "required_phases": resolution}
            if phase == "core":
                return {**result, "status": "SUPPLEMENT_DISPATCH_REQUIRED",
                        "required_phases": resolution}
            if active["status"] not in {"READY", "RAW_COMPLETE"}:
                raise ContinuationRefused("Required phase is not ready for continuation: " + active["status"])
            if phase == "observer_cost" and resolution["phase"] != "observer_cost":
                raise ContinuationRefused("Observer-cost cannot precede verified validation512 delivery")
        if phase != "core" and not core_completed(run, manifest):
            raise ContinuationRefused("Supplement cannot continue before core completion")
        if not current_job.isdigit():
            raise ContinuationRefused("Current job ID must be exact numeric")
        protocol = _source_context(run, repo)
        if manifest.get("protocol_hash") != protocol["protocol_hash"]:
            raise ContinuationRefused("Manifest/protocol hash mismatch")
        launch_path = (Path(reports) / run.name / phase / "ops/launch.json"
                       if phase != "core" else run / "launch.json")
        launch_raw = launch_path.read_bytes()
        launch = json.loads(launch_raw)
        expected_command = [sys.executable, "-u", str(repo / "scripts/run_state_calibration_study.py"),
                            "auto", "--run-dir", str(run)]
        if phase != "core":
            expected_command = phase_ops.command(run, repo=repo, phase=phase)
            claim = launch.get("shared_handoff_claim", {})
            claim_path = Path(claim.get("path", "/nonexistent"))
            current_state = supplement if phase == "validation512" else resolution["state"]
            if (launch.get("schema_version") != phase_ops.schema_prefix(phase) + "_launch.v1"
                    or launch.get("task") != phase or launch.get("phase") != phase
                    or launch.get("plan_hash") != current_state.get("plan_hash")
                    or launch.get(phase_ops.plan_key(phase)) != current_state.get("plan_hash")
                    or launch.get("core_launch_sha256") != checksum((run / "launch.json").read_bytes())
                    or not claim_path.is_file() or checksum(claim_path.read_bytes()) != claim.get("sha256")):
                raise ContinuationRefused("Independent supplement launch/plan/core/shared-claim binding differs")
        if (str(launch.get("slurm_job_id")) != current_job or launch.get("pid") != os.getpid()
                or launch.get("metadata_written_before_exec") is not True
                or launch.get("exec_preserves_pid") is not True
                or launch.get("source_hash") != protocol["source_hash"]
                or launch.get("protocol_hash") != protocol["protocol_hash"]
                or launch.get("command") not in ((expected_command,) if phase != "core"
                    else (expected_command, expected_command + [
                        "--config", str(repo / "configs/state_calibration_study.yaml")]))):
            raise ContinuationRefused("Not the same-PID verified handoff into the exact frozen auto")
        previous = launch.get("previous_slurm_job_id")
        if not isinstance(previous, str) or not previous.isdigit() or previous == current_job:
            raise ContinuationRefused("Missing distinct predecessor binding")
        if (launch.get("predecessor_terminal_proof", {}).get("status") != "VERIFIED_TERMINAL"
                or launch.get("current_allocation_proof", {}).get("job_id") != current_job
                or not Path(launch.get("previous_launch_history", "")).is_file()
                or not Path(launch.get("predecessor_snapshot", {}).get("commit_path", "")).is_file()):
            raise ContinuationRefused("Verified handoff history/boundary/allocation proof missing")
        active_response = query_fn(["scontrol", "-o", "show", "job", current_job])
        _verify_resources(_record(active_response, current_job), state="RUNNING", repo=repo)
        key = checksum((str(run) + "\n" + protocol["protocol_hash"]).encode())[:16]
        prefix = "moe-cont-" + key + "-"
        name = prefix + current_job
        dependency = "afterany:" + current_job
        directory = Path(reports) / run.name / "continuation" / current_job
        directory.mkdir(parents=True, exist_ok=True)
        for parent in (directory.parent, directory.parent.parent):
            _sync_directory(parent)
        legacy_path = run / "slurm_continuation.json"
        legacy_job = str(read_json(legacy_path).get("job_id", "")) if legacy_path.exists() else ""
        matches, inventory = _active_successors(query_fn, prefix, current_job, legacy_job)
        if matches and (matches[0]["name"] != name or matches[0]["state"] != "PENDING"):
            raise ContinuationRefused("Another active/unknown successor prevents submission")
        intent_path = directory / "intent.json"
        commit_path = directory / "committed.json"
        command = ["sbatch", "--parsable", "--job-name=" + name, "--dependency=" + dependency,
                   "--no-requeue", "--partition=" + RESOURCES["partition"],
                   "--account=" + RESOURCES["account"], "--qos=" + RESOURCES["qos"],
                   "--gres=" + RESOURCES["gres"], "--cpus-per-task=8", "--mem=128G",
                   "--time=1-00:00:00", "--chdir=" + str(repo),
                   "--output=/scratch/jgu7/moe-state-study/slurm/%x-%j.log",
                   str(repo / "scripts/state_study_resume.sbatch"), str(run), current_job]
        intent = {**result, "status": "SUBMISSION_INTENT", "job_name": name,
                  "dependency": dependency, "command": command, "resources": RESOURCES,
                  "protocol_hash": protocol["protocol_hash"], "source_hash": protocol["source_hash"],
                  "launch_sha256": checksum(launch_raw), "current_allocation": active_response,
                  "supplement_plan_hash": supplement.get("plan_hash") if supplement else None,
                  "observer_cost_plan_hash": (resolution["state"].get("plan_hash")
                      if phase == "observer_cost" else None),
                  "inventory": inventory, "owner_uid": os.getuid(),
                  "source_files": {str(path.relative_to(repo)): checksum(path.read_bytes()) for path in (
                      repo / "scripts/state_study_continuation.py", repo / "scripts/launch_state_study_resume.py",
                      repo / "scripts/state_study_resume.sbatch")}}
        existing = intent_path.exists()
        if existing:
            old_intent = read_json(intent_path)
            for field in ("run_dir", "current_job_id", "job_name", "dependency", "command",
                          "protocol_hash", "source_hash", "launch_sha256", "owner_uid", "resources",
                          "phase", "supplement_plan_hash", "observer_cost_plan_hash"):
                if old_intent.get(field) != intent[field]:
                    raise ContinuationRefused("Existing intent identity differs; never resubmit")
        else:
            try:
                _publish(intent_path, intent)
            except FileExistsError:
                # Another caller owns submission, even if its record has not
                # arrived yet.  Never race it by starting our own sbatch.
                raise ContinuationRefused("Another caller published this job's submission intent")
        result.update(intent_path=str(intent_path), job_name=name, dependency=dependency)
        if matches:
            job_id = matches[0]["job_id"]
            response = query_fn(["scontrol", "-o", "show", "job", job_id])
            _verify_resources(_record(response, job_id), state="PENDING", name=name,
                              dependency=dependency, repo=repo)
            committed = {**result, "status": "QUEUED", "job_id": job_id,
                         "adopted_existing": True, "verification": response,
                         "resources": RESOURCES}
            if commit_path.exists():
                if read_json(commit_path).get("job_id") != job_id:
                    raise ContinuationRefused("Committed successor ID conflicts with controller")
                return {**committed, "status": "ALREADY_QUEUED"}
            _publish(commit_path, committed)
            return committed
        if existing or commit_path.exists():
            return {**result, "status": "SUBMISSION_UNKNOWN",
                    "reason": "durable_intent_has_no_exact_active_successor; no resubmission"}
        # Recheck immediately before the sole non-read-only scheduler action.
        fresh = read_json(run / "manifest.json")
        fresh_errors = scientific_error(fresh, run)
        if fresh_errors:
            return {**result, "status": "SCIENTIFIC_STOP", "errors": fresh_errors}
        if phase != "core":
            latest = phase_ops.required_phase_state(run)
            if phase_ops.overall_done(latest):
                return {**result, "status": "ALREADY_COMPLETE_NO_SUBMISSION", "required_phases": latest}
            if latest["validation512"]["status"] == "NOT_REQUIRED" and not phase_ops.phase_done(latest["validation512"]):
                return {**result, "status": "SUPPLEMENT_DELIVERY_REQUIRED", "required_phases": latest}
            if latest["state"]["status"] == "SCIENTIFIC_ERROR":
                return {**result, "status": "SCIENTIFIC_STOP", "required_phases": latest}
            if (latest["state"]["status"] not in {"READY", "RAW_COMPLETE"}
                    or latest["phase"] != resolution["phase"]
                    or latest["state"].get("plan_hash") != resolution["state"].get("plan_hash")
                    or latest["validation512"].get("plan_hash") != supplement.get("plan_hash")):
                raise ContinuationRefused("Required phase/status/plan changed before submission")
        elif core_completed(run, fresh):
            return {**result, "status": "SUPPLEMENT_DISPATCH_REQUIRED"}
        if launch_path.read_bytes() != launch_raw:
            raise ContinuationRefused("Launch identity changed before submission")
        submission = submit_fn(command)
        _publish(directory / "submission.json", submission)
        output = submission.get("stdout", "").strip()
        match = re.fullmatch(r"([0-9]+)(?:;([A-Za-z0-9_.-]+))?", output)
        if submission.get("returncode") != 0 or not match:
            return {**result, "status": "SUBMISSION_UNKNOWN",
                    "reason": "sbatch_failed_timed_out_or_returned_no_unambiguous_id",
                    "submission": submission}
        job_id = match[1]
        verification = query_fn(["scontrol", "-o", "show", "job", job_id])
        try:
            _verify_resources(_record(verification, job_id), state="PENDING", name=name,
                              dependency=dependency, repo=repo)
        except ContinuationRefused as exc:
            return {**result, "status": "SUBMISSION_UNKNOWN", "job_id_returned": job_id,
                    "reason": str(exc), "submission": submission, "verification": verification}
        committed = {**result, "status": "QUEUED", "job_id": job_id,
                     "cluster": match[2], "adopted_existing": False,
                     "submission": submission, "verification": verification, "resources": RESOURCES}
        _publish(commit_path, committed)
        return committed
    except (ContinuationRefused, OSError, ValueError, KeyError, TypeError) as exc:
        return {**result, "status": "REFUSED", "reason": str(exc)}
def record_outcome(run, job_id, outcome, *, reports=REPORTS):
    """Archive the exact returned status without replacing earlier evidence."""
    directory = Path(reports) / Path(run).name / "continuation" / str(job_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("outcome-" + uuid.uuid4().hex + ".json")
    _publish(path, outcome)
    return str(path)
