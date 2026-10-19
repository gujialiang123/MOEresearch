"""CPU handoffs through independently sealed validation and engineering phases.

The original run is read-only in this phase.  Launch/history/boundaries are
independent report artifacts; only the original core handoff updates run/launch.
No scheduler submission or model execution occurs on import.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

if __package__:
    from .state_study_continuation import ContinuationRefused, _publish, canonical, read_json
else:
    from state_study_continuation import ContinuationRefused, _publish, canonical, read_json


REPORTS = Path("/scratch/jgu7/moe-state-study/reports")
REPO = Path(__file__).resolve().parents[1]
PHASE = "validation512"
COST_PHASE = "observer_cost"
PHASES = (PHASE, COST_PHASE)
DONE = {"COMPLETE", "NOT_REQUIRED"}
READY = {"READY", "RAW_COMPLETE"}
KNOWN = DONE | READY | {"PLAN_PENDING", "INVALID_PLAN", "SCIENTIFIC_ERROR",
                        "WAITING_P4", "WAITING_CORE"}
COST_KNOWN = READY | {"COMPLETE", "PLAN_PENDING", "INVALID_PLAN", "SCIENTIFIC_ERROR",
                     "WAITING_CORE", "WAITING_VALIDATION512"}
OPS_FILES = ("scripts/state_study_supplement_ops.py", "scripts/launch_state_study_resume.py",
             "scripts/state_study_continuation.py", "scripts/state_study_quality_watch.py",
             "scripts/state_study_resume.sbatch",
             "tests/test_state_study_supplement_ops.py", "tests/test_state_study_continuation.py",
             "tests/test_state_study_quality_watch.py")


def utc():
    return datetime.now(timezone.utc).isoformat()


def checksum(raw):
    return hashlib.sha256(raw).hexdigest()


def ops_dir(run, reports=REPORTS, *, phase=PHASE):
    if phase not in PHASES:
        raise ContinuationRefused("Unknown independent phase")
    return Path(reports) / Path(run).name / phase / "ops"


def launch_path(run, reports=REPORTS, *, phase=PHASE):
    return ops_dir(run, reports, phase=phase) / "launch.json"


def command(run, *, repo=REPO, python=None, phase=PHASE):
    if phase not in PHASES:
        raise ContinuationRefused("Unknown independent phase")
    script = "state_study_validation512.py" if phase == PHASE else "state_study_observer_cost.py"
    return [python or sys.executable, "-u", str(Path(repo) / "scripts" / script),
            "run", "--run-dir", str(Path(run).resolve())]


def schema_prefix(phase):
    if phase not in PHASES:
        raise ContinuationRefused("Unknown independent phase")
    return "state_validation512" if phase == PHASE else "state_observer_cost"


def plan_key(phase):
    return "supplement_plan_hash" if phase == PHASE else "observer_cost_plan_hash"


def effective_launch(run, reports=REPORTS):
    """Phases advance once; the latest registered phase is the predecessor."""
    for phase in reversed(PHASES):
        path = launch_path(run, reports, phase=phase)
        if path.is_file():
            return phase, path
    return "core", Path(run) / "launch.json"


def publish_ops_sources(run, *, reports=REPORTS, repo=REPO):
    """Preserve actual mutable ops sources separately from the science seal."""
    repo = Path(repo)
    raw = {name: (repo / name).read_bytes() for name in OPS_FILES}
    files = {name: checksum(value) for name, value in raw.items()}
    key = canonical(files)
    directory = Path(reports) / Path(run).name / "validation512_ops_sources" / ("runtime_" + key)
    manifest = directory / "MANIFEST.json"
    if directory.exists():
        if not manifest.is_file() or read_json(manifest).get("files") != files:
            raise ContinuationRefused("Existing ops source archive is incomplete or differs")
        if any(not (directory / name).is_file() or checksum((directory / name).read_bytes()) != digest
               for name, digest in files.items()):
            raise ContinuationRefused("Archived actual ops source bytes differ")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        for name, value in raw.items():
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(value); handle.flush(); os.fsync(handle.fileno())
        _publish(manifest, {"schema_version": "state_validation512_ops_sources.v1",
                           "captured_at_utc": utc(), "files": files, "source_inventory_hash": key,
                           "scope": "Operational sources/tests; separate from immutable scientific plan"})
    return {"directory": str(directory.resolve()), "manifest_sha256": checksum(manifest.read_bytes()),
            "source_inventory_hash": key, "files": files}


def _study():
    if __package__:
        from . import state_study_validation512
    else:
        import state_study_validation512
    return state_study_validation512


def _cost():
    if __package__:
        from . import state_study_observer_cost
    else:
        import state_study_observer_cost
    return state_study_observer_cost


def phase_module(phase):
    return _study() if phase == PHASE else _cost()


def supplement_state(run):
    """Use the supplement's CPU verifier; missing/invalid plans never mean done."""
    try:
        value = _study().status(Path(run).resolve())
    except Exception as exc:
        raise ContinuationRefused("Full-validation supplement CPU status is unavailable: " + str(exc)) from exc
    if not isinstance(value, dict) or value.get("status") not in KNOWN:
        raise ContinuationRefused("Unknown full-validation supplement status")
    return value


def observer_cost_state(run):
    try:
        value = _cost().status(Path(run).resolve())
    except Exception as exc:
        raise ContinuationRefused("Observer-cost CPU status is unavailable: " + str(exc)) from exc
    if not isinstance(value, dict) or value.get("status") not in COST_KNOWN:
        raise ContinuationRefused("Unknown observer-cost status")
    return value


def _report():
    if __package__:
        from . import state_study_validation512_report
    else:
        import state_study_validation512_report
    return state_study_validation512_report


def verify_delivery(run):
    return _report().verify_delivery(run)


def ensure_not_required_delivery(run):
    """The sole report owner writes; ops only demands its explicit receipt."""
    created = _report().ensure_delivery(run)
    if not isinstance(created, dict) or created.get("verified") is not True or created.get("delivery_complete") is not True:
        raise ContinuationRefused("NOT_REQUIRED CPU delivery is not verified")
    proof = verify_delivery(run)
    if not isinstance(proof, dict) or proof.get("verified") is not True or proof.get("delivery_complete") is not True:
        raise ContinuationRefused("NOT_REQUIRED delivery receipt did not verify after creation")
    return proof


def phase_done(state):
    return state.get("status") in DONE and state.get("delivery_complete") is True


def required_phase_state(run, *, validation_state=None, cost_state_fn=None):
    """A verified validation completion is a prerequisite, never overall done."""
    validation = supplement_state(run) if validation_state is None else validation_state
    if not phase_done(validation):
        return {"phase": PHASE, "state": validation, "validation512": validation,
                "overall_complete": False}
    cost = (cost_state_fn or observer_cost_state)(run)
    return {"phase": COST_PHASE, "state": cost, "validation512": validation,
            "overall_complete": cost.get("status") == "COMPLETE" and cost.get("delivery_complete") is True}


def overall_done(resolution):
    return resolution.get("overall_complete") is True


def ready_context(run, state, *, reports=REPORTS, phase=PHASE):
    if state.get("status") not in READY:
        raise ContinuationRefused("Supplement is not ready: " + str(state.get("status")))
    module = phase_module(phase)
    try:
        plan = module.load_plan(run)
    except Exception as exc:
        raise ContinuationRefused("Verified supplement plan is unavailable: " + str(exc)) from exc
    if not isinstance(plan, dict) or not isinstance(plan.get("plan_hash"), str):
        raise ContinuationRefused("Verified supplement plan hash is missing")
    output = Path(module.output_dir(run, plan)).resolve()
    parent = (Path(reports) / Path(run).name / phase).resolve()
    if output.parent != parent or output.name != plan["plan_hash"]:
        raise ContinuationRefused("Supplement output does not match the sealed plan hash")
    if state.get("plan_hash") != plan["plan_hash"] or Path(state.get("output_dir", "/nonexistent")).resolve() != output:
        raise ContinuationRefused("Supplement status and verified plan/output disagree")
    return plan, output


def _snapshot_file(path, parent):
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    identity = lambda s: (s.st_ino, s.st_size, s.st_mtime_ns)
    if identity(before) != identity(after) or len(raw) != before.st_size:
        raise ContinuationRefused("Supplement checkpoint changed during handoff")
    result = {"bytes": before.st_size, "inode": before.st_ino,
              "mtime_ns": before.st_mtime_ns, "sha256": checksum(raw)}
    if path.suffix == ".jsonl":
        if raw and not raw.endswith(b"\n"):
            raise ContinuationRefused("Incomplete supplement JSONL checkpoint; preserve for review")
        # LF is the physical commit delimiter. CR is valid JSON whitespace,
        # and U+2028 inside strings is not a record boundary.
        lines = raw.split(b"\n")[:-1] if raw else []
        for line in lines:
            if not isinstance(json.loads(line), dict):
                raise ContinuationRefused("Supplement JSONL checkpoint is not an object")
        result["physical_lines"] = len(lines)
    return str(path.relative_to(parent)), result


def checkpoint_snapshot(output, *, all_json=False):
    """Bind committed progress without rereading large activation/token arrays."""
    output = Path(output)
    if not output.exists():
        return {"directory": str(output), "files": {}, "physical_lines": 0, "all_json": all_json}
    def inventory():
        receipts = (output / name for name in ("source_snapshot.json", "raw_completion.json",
                    "completion.json", "summaries/diagnostic_summary.json"))
        return sorted({*output.rglob("*.complete.json"), *output.rglob("*.empty.json"),
                       *(p for p in output.rglob("*.jsonl") if not p.name.endswith(".tokens.jsonl")),
                       *(p for p in receipts if p.is_file()),
                       *(output.rglob("*.json") if all_json else ())})
    paths = inventory()
    files = {}
    for path in paths:
        if not path.resolve().is_relative_to(output.resolve()):
            raise ContinuationRefused("Supplement checkpoint escapes output directory")
        name, item = _snapshot_file(path, output)
        files[name] = item
    if paths != inventory():
        raise ContinuationRefused("Supplement checkpoint inventory changed during handoff")
    return {"directory": str(output), "files": files,
            "all_json": all_json,
            "physical_lines": sum(x.get("physical_lines", 0) for x in files.values()),
            "large_arrays_not_reread": True,
            "array_integrity_owner": "sealed supplement collector/status and per-prompt commit hashes"}


def verify_checkpoint_snapshot(snapshot):
    parent = Path(snapshot["directory"])
    current = checkpoint_snapshot(parent, all_json=snapshot.get("all_json", False))
    if current["files"] != snapshot["files"]:
        raise ContinuationRefused("Supplement progress changed after boundary capture")


def _strict_batch_exit(path):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ContinuationRefused("Duplicate batch exit-code field")
            value[key] = item
        return value
    raw = Path(path).read_bytes()
    value = json.loads(raw, object_pairs_hook=pairs,
                       parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    keys = {"scientific_exit_code", "report_exit_code", "plot_exit_code"}
    if (not isinstance(value, dict) or set(value) != keys
            or any(type(value[k]) is not int or not 0 <= value[k] <= 255 for k in keys)):
        raise ContinuationRefused("Malformed exact batch exit-code record")
    return value, checksum(raw)


def _registered_predecessor(run, phase, launch, plan, sources, reports):
    """Identity of the old noncore launch, including its exclusive provenance."""
    expected = {"schema_version": schema_prefix(phase) + "_launch.v1", "phase": phase, "task": phase,
        "run_dir": str(run), "plan_hash": plan["plan_hash"], plan_key(phase): plan["plan_hash"],
        "protocol_hash": sources["protocol_hash"], "source_hash": sources["source_hash"],
        "core_launch_sha256": checksum((run / "launch.json").read_bytes()),
        "metadata_written_before_exec": True, "exec_preserves_pid": True}
    if (any(launch.get(k) != v for k, v in expected.items())
            or type(launch.get("pid")) is not int or launch["pid"] <= 0
            or any(not isinstance(launch.get(k), str) or not launch[k]
                   for k in ("hostname", "boot_id", "proc_start_ticks", "cgroup"))):
        raise ContinuationRefused("Unregistered delivery predecessor identity")
    job = str(launch.get("slurm_job_id", ""))
    allocation = launch.get("current_allocation_proof", {})
    if (not job.isdigit() or str(allocation.get("job_id")) != job
            or allocation.get("cgroup") != launch["cgroup"]):
        raise ContinuationRefused("Delivery predecessor allocation binding differs")
    local = ops_dir(run, reports, phase=phase)
    previous_hash = launch.get("previous_launch_sha256", "")
    history = Path(launch.get("previous_launch_history", "/missing"))
    reference = launch.get("shared_handoff_claim", {})
    claim_path = Path(reference.get("path", "/missing"))
    if (not history.resolve().is_relative_to((local / "launch_history").resolve())
            or not history.is_file() or checksum(history.read_bytes()) != previous_hash
            or claim_path.resolve() != (local / "claims" / (previous_hash + ".json")).resolve()
            or not claim_path.is_file() or checksum(claim_path.read_bytes()) != reference.get("sha256")):
        raise ContinuationRefused("Delivery predecessor shared claim/history differs")
    claim = read_json(claim_path)
    expected_claim = {"schema_version": schema_prefix(phase) + "_handoff_claim.v1", "task": phase,
        "run_dir": str(run), "plan_hash": plan["plan_hash"], "protocol_hash": sources["protocol_hash"],
        "source_hash": sources["source_hash"], "owner_uid": os.getuid(), "pid": launch["pid"],
        "hostname": launch["hostname"], "boot_id": launch["boot_id"],
        "proc_start_ticks": launch["proc_start_ticks"], "slurm_job_id": launch["slurm_job_id"],
        "previous_launch_sha256": previous_hash}
    if any(claim.get(k) != v for k, v in expected_claim.items()):
        raise ContinuationRefused("Delivery predecessor actual claim identity differs")
    archive = launch.get("ops_source_archive", {})
    archived = Path(archive.get("directory", "/missing"))
    files = launch.get("ops_source_sha256", {})
    if (set(files) != set(OPS_FILES) or not (archived / "MANIFEST.json").is_file()
            or checksum((archived / "MANIFEST.json").read_bytes()) != archive.get("manifest_sha256")
            or read_json(archived / "MANIFEST.json").get("files") != files
            or any(checksum((archived / name).read_bytes()) != digest for name, digest in files.items())):
        raise ContinuationRefused("Delivery predecessor actual ops source archive differs")


def delivery_exit2_proof(run, job, terminal, phase, launch, resume, *, reports=REPORTS,
                         current_launch=None):
    """Read-only narrow proof; inspecting/check-only never spends a retry.

    A resolved genuine receipt permits later phases without reopening the retry
    budget. Otherwise precisely one committed-work delivery reentry is allowed.
    """
    run = Path(run).resolve(); job = str(job)
    if (phase not in PHASES or not isinstance(launch, dict)
            or terminal.get("status") != "VERIFIED_TERMINAL"
            or str(terminal.get("job_id")) != job or str(launch.get("slurm_job_id")) != job
            or terminal.get("state") not in {"FAILED", "COMPLETED"}):
        raise ContinuationRefused("CLI2 exception requires an exact registered noncore terminal job")
    codes = Path(reports) / run.name / ("batch-exit-codes-" + job + ".json")
    values, codes_hash = _strict_batch_exit(codes)
    if values["scientific_exit_code"] != 2:
        raise ContinuationRefused("Only the actual noncore CLI Pending exit2 can reenter")
    module = phase_module(phase)
    plan = module.load_plan(run)
    sources, manifest = resume.validate_sources(run)
    if (resume.scientific_error(manifest, run)
            or plan.get("protocol_hash") != sources["protocol_hash"]
            or plan.get("source", {}).get("frozen_source_hash") != sources["source_hash"]
            or launch.get("command") != command(run, repo=resume.REPO, phase=phase)):
        raise ContinuationRefused("Delivery predecessor science/source/command differs")
    _registered_predecessor(run, phase, launch, plan, sources, reports)
    state = module.status(run)
    if state.get("plan_hash") != plan["plan_hash"]:
        raise ContinuationRefused("Delivery status and sealed predecessor plan differ")
    directory = Path(module.output_dir(run, plan)).resolve()
    if directory != (Path(reports) / run.name / phase / plan["plan_hash"]).resolve():
        raise ContinuationRefused("Delivery output differs from the sealed plan")
    result = {"phase": phase, "plan_hash": plan["plan_hash"], "previous_job_id": job,
        "previous_launch_identity_hash": canonical(launch), "batch_exit_path": str(codes),
        "batch_exit_sha256": codes_hash, "batch_exit": values, "native_status": state.get("status")}
    delivery = state.get("delivery_proof", {})
    resolved_status = {"COMPLETE", "NOT_REQUIRED"} if phase == PHASE else {"COMPLETE"}
    if state.get("status") in resolved_status and phase_done(state) and delivery.get("verified") is True and delivery.get("delivery_complete") is True and delivery.get("status") == "VERIFIED":
        receipt = directory / "delivery/delivery_receipt.json"
        if (Path(delivery.get("receipt_path", "/missing")).resolve() != receipt
                or delivery.get("plan_hash") != plan["plan_hash"]
                or not receipt.is_file() or checksum(receipt.read_bytes()) != delivery.get("receipt_sha256")
                or read_json(receipt).get("delivery_hash") != delivery.get("delivery_hash")):
            raise ContinuationRefused("Resolved delivery actual receipt identity differs")
        return {**result, "mode": "resolved_delivery", "delivery_identity": {
            "receipt_path": str(receipt), "receipt_sha256": delivery["receipt_sha256"],
            "delivery_hash": delivery["delivery_hash"]}}
    work = state.get("completion_proof" if phase == PHASE else "work_proof", {})
    pending_status = {"MISSING", "PENDING"} if phase == PHASE else {"PENDING"}
    if (state.get("status") != "RAW_COMPLETE" or work.get("complete") is not True
            or delivery.get("status") not in pending_status or delivery.get("verified") is not False
            or delivery.get("delivery_complete") is not False):
        raise ContinuationRefused("CLI2 lacks completed work and a genuine pending delivery")
    complete = directory / "completion.json"
    completion_hash = checksum(complete.read_bytes())
    key = canonical([str(run), phase, plan["plan_hash"], completion_hash])
    intent_path = ops_dir(run, reports, phase=phase) / "delivery_retries" / key / "intent.json"
    result.update(mode="cpu_delivery_retry", completion_sha256=completion_hash,
                  retry_key=key, retry_intent_path=str(intent_path), attempt=1,
                  native_delivery_status=delivery["status"])
    if intent_path.exists():
        intent = read_json(intent_path)
        reference = (current_launch or {}).get("cpu_delivery_retry", {})
        if (not current_launch or reference.get("path") != str(intent_path)
                or reference.get("sha256") != checksum(intent_path.read_bytes())
                or any(intent.get(k) != v for k, v in {
                    "schema_version": "state_cpu_delivery_retry.v1", "run_dir": str(run), "phase": phase,
                    "plan_hash": plan["plan_hash"], "completion_sha256": completion_hash,
                    "previous_launch_identity_hash": canonical(launch), "previous_job_id": job,
                    "job_id": current_launch.get("slurm_job_id"), "pid": current_launch.get("pid"),
                    "boot_id": current_launch.get("boot_id"), "proc_start_ticks": current_launch.get("proc_start_ticks"),
                    "hostname": current_launch.get("hostname"), "attempt": 1}.items())):
            raise ContinuationRefused("STOP_FOR_DELIVERY_REVIEW: one automatic CPU delivery retry already used")
        result["registered_current_retry"] = True
    elif current_launch is not None:
        raise ContinuationRefused("Registered CPU delivery retry intent is absent")
    return result


def phase_predecessor_guard(run, job, terminal, resume, *, phase="core", launch=None,
                            reports=REPORTS, current_launch=None):
    """Keep the legacy failure rule except for the proven delivery-only case."""
    stop = resume.predecessor_stop(run, job, terminal)
    if (not stop and phase in PHASES and isinstance(launch, dict)
            and launch.get("cpu_delivery_retry") and current_launch is None):
        # Spending one delivery retry covers this allocation's infrastructure
        # failure too. A missing batch tail must not reopen an endless chain.
        try:
            if terminal.get("status") != "VERIFIED_TERMINAL" or str(terminal.get("job_id")) != str(job):
                raise ContinuationRefused("Delivery retry allocation is not positively terminal")
            module = phase_module(phase); plan = module.load_plan(Path(run).resolve())
            sources, manifest = resume.validate_sources(run)
            if resume.scientific_error(manifest, run):
                raise ContinuationRefused("Scientific error after delivery retry")
            _registered_predecessor(Path(run).resolve(), phase, launch, plan, sources, reports)
            state = module.status(run); delivery = state.get("delivery_proof", {})
            if (not phase_done(state) or state.get("plan_hash") != plan["plan_hash"]
                    or delivery.get("verified") is not True or delivery.get("delivery_complete") is not True
                    or delivery.get("status") != "VERIFIED"):
                raise ContinuationRefused("One delivery retry allocation already consumed; delivery is still incomplete")
            receipt = Path(module.output_dir(run, plan)).resolve() / "delivery/delivery_receipt.json"
            if (Path(delivery.get("receipt_path", "/missing")).resolve() != receipt or not receipt.is_file()
                    or checksum(receipt.read_bytes()) != delivery.get("receipt_sha256")
                    or read_json(receipt).get("delivery_hash") != delivery.get("delivery_hash")):
                raise ContinuationRefused("Resolved retry actual delivery receipt differs")
            proof = {"phase": phase, "plan_hash": plan["plan_hash"], "previous_job_id": str(job),
                "previous_launch_identity_hash": canonical(launch), "mode": "resolved_delivery",
                "resolution_origin": "delivery_retry_allocation_terminal_without_nonzero_scientific_exit",
                "native_status": state["status"], "delivery_identity": {"receipt_path": str(receipt),
                    "receipt_sha256": delivery["receipt_sha256"], "delivery_hash": delivery["delivery_hash"]}}
            return {"stop": None, "delivery_exception": proof}
        except (ContinuationRefused, OSError, ValueError, KeyError, TypeError) as exc:
            return {"stop": {"reason": "STOP_FOR_DELIVERY_REVIEW", "detail": str(exc)},
                    "delivery_exception": None}
    if not stop:
        return {"stop": None, "delivery_exception": None}
    if phase not in PHASES:
        return {"stop": stop, "delivery_exception": None}
    try:
        proof = delivery_exit2_proof(run, job, terminal, phase, launch, resume,
                                     reports=reports, current_launch=current_launch)
    except (ContinuationRefused, OSError, ValueError, KeyError, TypeError) as exc:
        return {"stop": {**stop, "delivery_exception_refused": str(exc)}, "delivery_exception": None}
    return {"stop": None, "delivery_exception": proof}


def record_resolved_delivery(run, proof, old_raw, *, reports=REPORTS):
    identity = {**proof, "previous_launch_sha256": checksum(old_raw)}
    key = canonical(identity)
    path = ops_dir(run, reports, phase=proof["phase"]) / "resolved_delivery" / (key + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        try:
            _publish(path, {**identity, "recorded_at_utc": utc()})
        except FileExistsError:
            pass
    if {k:v for k,v in read_json(path).items() if k != "recorded_at_utc"} != identity:
        raise ContinuationRefused("Immutable resolved-delivery evidence differs")
    return {"path": str(path), "sha256": checksum(path.read_bytes())}


def validation_terminal(run, reports, resume):
    """A completed receipt does not itself prove its GPU process terminated."""
    path = launch_path(run, reports)
    if not path.is_file():
        return None
    value = read_json(path)
    job = str(value.get("slurm_job_id", ""))
    if not job.isdigit() or resume.verify_launch_process(run, value).get("status") == "VERIFIED_LIVE":
        raise ContinuationRefused("Exact validation512 process is live or its allocation is invalid")
    proof = resume.predecessor_terminal(job)
    if proof.get("status") != "VERIFIED_TERMINAL" or str(proof.get("job_id")) != job:
        raise ContinuationRefused("Validation512 Slurm state is unknown/nonterminal")
    guarded = phase_predecessor_guard(run, job, proof, resume, phase=PHASE,
                                      launch=value, reports=reports)
    if guarded["stop"]:
        raise ContinuationRefused("Validation512 failure/cancellation requires review")
    return proof


def registered_cost_launch(run, plan, *, reports=REPORTS, proc=Path("/proc"),
                           environment=None, resume=None):
    """CPU-only validation of the exact same-PID registered engineering run.

    The cost module calls this before importing torch or loading a model. Shared
    claims and positive scheduler evidence establish cross-node exclusivity;
    the later local-disk lock is additional protection only.
    """
    if resume is None:
        if __package__:
            from . import launch_state_study_resume as resume
        else:
            import launch_state_study_resume as resume
    run = Path(run).resolve()
    environment = os.environ if environment is None else environment
    proc = Path(proc)
    verified_plan = _cost().load_plan(run)
    if plan != verified_plan:
        raise ContinuationRefused("Observer-cost plan is not the actual verified sealed plan")
    validation_plan = _study().load_plan(run)
    validation = supplement_state(run)
    if (not phase_done(validation) or validation.get("plan_hash") != validation_plan["plan_hash"]
            or plan.get("source") != validation_plan.get("source")
            or plan.get("protocol_hash") != validation_plan.get("protocol_hash")):
        raise ContinuationRefused("Observer-cost requires verified validation512 work and delivery")
    reference = plan.get("validation512_plan", {})
    validation_plan_path = Path(_study().plan_path(run))
    if (Path(reference.get("path", "/missing")).resolve() != validation_plan_path.resolve()
            or reference.get("sha256") != checksum(validation_plan_path.read_bytes())
            or reference.get("plan_hash") != validation_plan["plan_hash"]):
        raise ContinuationRefused("Observer-cost sealed validation512 plan reference differs")
    local = ops_dir(run, reports, phase=COST_PHASE)
    path = launch_path(run, reports, phase=COST_PHASE)
    launch = read_json(path)
    core_path = run / "launch.json"
    core = read_json(core_path)
    protocol = read_json(run / "protocol.json")
    expected_command = command(run, phase=COST_PHASE)
    expected = {"schema_version": "state_observer_cost_launch.v1", "task": COST_PHASE,
        "phase": COST_PHASE, "run_dir": str(run), "plan_hash": plan["plan_hash"],
        "observer_cost_plan_hash": plan["plan_hash"], "protocol_hash": plan["protocol_hash"],
        "source_hash": protocol["source_hash"], "pid": os.getpid(),
        "hostname": socket.gethostname(), "command": expected_command,
        "core_launch_sha256": checksum(core_path.read_bytes()),
        "validation512_plan_hash": validation_plan["plan_hash"],
        "metadata_written_before_exec": True, "exec_preserves_pid": True}
    if any(launch.get(k) != value for k, value in expected.items()):
        raise ContinuationRefused("Registered observer-cost launch/source/command differs")
    observed = resume._proc_stat((proc / "self/stat").read_text())
    boot = (proc / "sys/kernel/random/boot_id").read_text().strip()
    cgroup = (proc / "self/cgroup").read_text()
    actual_command = (proc / "self/cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
    if (str(observed["start_ticks"]) != str(launch.get("proc_start_ticks"))
            or boot != launch.get("boot_id") or cgroup != launch.get("cgroup")
            or actual_command != expected_command
            or str(launch.get("slurm_job_id")) != environment.get("SLURM_JOB_ID")):
        raise ContinuationRefused("Registered observer-cost exact PID/boot/ticks/cgroup/argv differs")
    previous = str(launch.get("previous_slurm_job_id", ""))
    core_job = str(core.get("slurm_job_id", ""))
    terminals = {}
    for job, key in ((previous, "predecessor_terminal_proof"), (core_job, "core_terminal_proof")):
        proof = launch.get(key, {})
        actual_terminal = resume.predecessor_terminal(job)
        if (not job.isdigit() or proof.get("status") != "VERIFIED_TERMINAL"
                or str(proof.get("job_id")) != job
                or actual_terminal.get("status") != "VERIFIED_TERMINAL"):
            raise ContinuationRefused("Observer-cost predecessor/core terminal proof differs or is unknown")
        terminals[job] = actual_terminal
    if resume.predecessor_stop(run, core_job, terminals[core_job]):
        raise ContinuationRefused("Core nonzero/failure cannot be a delivery-only exception")
    if resume.verify_launch_process(run, core).get("status") == "VERIFIED_LIVE":
        raise ContinuationRefused("Original exact core PID is still live")
    if not resume.completed(run, read_json(run / "manifest.json")):
        raise ContinuationRefused("Observer-cost core is incomplete")
    if resume.scientific_error(read_json(run / "manifest.json"), run):
        raise ContinuationRefused("Core scientific error forbids observer-cost")
    resume.validate_sources(run)
    allocation = resume.current_allocation(previous, environment=environment, proc_root=proc)
    resume.validate_allocation_resources(allocation)
    if (allocation.get("cgroup") != cgroup
            or str(allocation.get("job_id")) != str(launch.get("slurm_job_id"))):
        raise ContinuationRefused("Observer-cost actual allocation differs")
    history = Path(launch.get("previous_launch_history", "/missing"))
    previous_hash = launch.get("previous_launch_sha256", "")
    if (not history.resolve().is_relative_to((local / "launch_history").resolve())
            or not history.is_file() or checksum(history.read_bytes()) != previous_hash):
        raise ContinuationRefused("Observer-cost predecessor history bytes differ")
    predecessor = read_json(history)
    if (str(predecessor.get("slurm_job_id")) != previous
            or resume.verify_launch_process(run, predecessor).get("status") == "VERIFIED_LIVE"):
        raise ContinuationRefused("Exact observer-cost predecessor is live or differs")
    previous_phase = predecessor.get("phase", "core")
    guarded = phase_predecessor_guard(run, previous, terminals[previous], resume,
        phase=previous_phase, launch=predecessor, reports=reports, current_launch=launch)
    if guarded["stop"]:
        raise ContinuationRefused("Registered observer-cost predecessor failure requires review: " + str(guarded["stop"]))
    validation_launch = launch_path(run, reports)
    expected_validation_launch_hash = (checksum(validation_launch.read_bytes()) if validation_launch.is_file() else None)
    if launch.get("validation512_launch_sha256") != expected_validation_launch_hash:
        raise ContinuationRefused("Observer-cost validation512 launch changed")
    actual_validation_terminal = validation_terminal(run, reports, resume)
    saved_validation_terminal = launch.get("validation512_terminal_proof")
    if validation.get("status") == "COMPLETE" and actual_validation_terminal is None:
        raise ContinuationRefused("Completed validation512 lacks its actual independent launch")
    if actual_validation_terminal is not None and (not isinstance(saved_validation_terminal, dict)
            or saved_validation_terminal.get("status") != "VERIFIED_TERMINAL"
            or str(saved_validation_terminal.get("job_id")) != str(actual_validation_terminal.get("job_id"))):
        raise ContinuationRefused("Observer-cost validation512 terminal proof differs")
    ref = launch.get("shared_handoff_claim", {})
    claim_path = Path(ref.get("path", "/missing"))
    if (claim_path.resolve() != (local / "claims" / (previous_hash + ".json")).resolve()
            or not claim_path.is_file() or checksum(claim_path.read_bytes()) != ref.get("sha256")):
        raise ContinuationRefused("Observer-cost exclusive shared claim is missing/changed")
    claim = read_json(claim_path)
    expected_claim = {"schema_version": "state_observer_cost_handoff_claim.v1", "task": COST_PHASE,
        "run_dir": str(run), "plan_hash": plan["plan_hash"], "protocol_hash": plan["protocol_hash"],
        "source_hash": protocol["source_hash"], "owner_uid": os.getuid(), "pid": os.getpid(),
        "hostname": socket.gethostname(), "boot_id": boot, "proc_start_ticks": observed["start_ticks"],
        "slurm_job_id": allocation["job_id"], "previous_launch_sha256": previous_hash}
    if any(claim.get(k) != value for k, value in expected_claim.items()):
        raise ContinuationRefused("Observer-cost shared claim identity differs")
    archive = launch.get("ops_source_archive", {})
    directory = Path(archive.get("directory", "/missing"))
    files = launch.get("ops_source_sha256", {})
    if (set(files) != set(OPS_FILES) or not (directory / "MANIFEST.json").is_file()
            or checksum((directory / "MANIFEST.json").read_bytes()) != archive.get("manifest_sha256")
            or read_json(directory / "MANIFEST.json").get("files") != files
            or any(checksum((directory / name).read_bytes()) != digest
                   or checksum((REPO / name).read_bytes()) != digest for name, digest in files.items())):
        raise ContinuationRefused("Observer-cost actual ops source archive differs")
    return {"phase": COST_PHASE, "plan_hash": plan["plan_hash"], "pid": os.getpid(),
        "launch_path": str(path), "launch_sha256": checksum(path.read_bytes()),
        "claim_sha256": checksum(claim_path.read_bytes()), "allocation": allocation,
        "core_terminal_proof": launch["core_terminal_proof"],
        "validation512_plan_hash": validation_plan["plan_hash"], "ops_source_archive": archive}


def dispatch(args, run, audit, manifest, resume, *, reports=REPORTS,
             state_fn=supplement_state, context_fn=None, cost_state_fn=None):
    """Called only after the original core completion check in the shared helper.

    ``resume`` supplies the unchanged original guards/publication primitives.
    This entrypoint must execute in the real helper PID, never a second model
    process or background controller.
    """
    run = Path(run).resolve()
    if not resume.completed(run, manifest):
        raise ContinuationRefused("Independent supplement dispatch requires completed core work")
    if resume.scientific_error(manifest, run):
        raise ContinuationRefused("Core scientific error forbids supplement dispatch")
    validation = state_fn(run)
    audit["supplement"] = validation
    previous_phase, effective = effective_launch(run, reports)
    old_raw = effective.read_bytes()
    old = json.loads(old_raw)
    if not isinstance(old, dict) or str(old.get("slurm_job_id")) != args.previous_job_id:
        raise ContinuationRefused("Previous job differs from the latest effective supplement/core launch")
    core_raw = (run / "launch.json").read_bytes()
    if previous_phase != "core":
        if (old.get("schema_version") != schema_prefix(previous_phase) + "_launch.v1"
                or old.get("phase") != previous_phase or old.get("task") != previous_phase
                or old.get("run_dir") != str(run)
                or old.get("core_launch_sha256") != checksum(core_raw)
                or old.get("protocol_hash") != audit["sources"]["protocol_hash"]
                or old.get("source_hash") != audit["sources"]["source_hash"]
                or old.get("command") != command(run, repo=resume.REPO, phase=previous_phase)):
            raise ContinuationRefused("Independent supplement predecessor identity differs")
    else:
        resume.original_command(old, run)
    audit["effective_previous_launch"] = str(effective)
    audit["old_process"] = resume.verify_launch_process(run, old)
    if audit["old_process"]["status"] == "VERIFIED_LIVE":
        raise ContinuationRefused("Exact effective predecessor PID is still VERIFIED_LIVE")
    terminal = resume.predecessor_terminal(args.previous_job_id)
    audit["previous_job"] = terminal
    if terminal.get("status") != "VERIFIED_TERMINAL":
        raise ContinuationRefused("Effective predecessor Slurm state is unknown/nonterminal")
    guarded = phase_predecessor_guard(run, args.previous_job_id, terminal, resume,
        phase=previous_phase, launch=old, reports=reports)
    audit["predecessor_delivery_exception"] = guarded["delivery_exception"]
    if guarded["stop"]:
        raise ContinuationRefused("Effective predecessor scientific failure/cancellation requires review: " + str(guarded["stop"]))
    if guarded["delivery_exception"] and guarded["delivery_exception"]["mode"] == "resolved_delivery" and not args.check_only:
        audit["resolved_delivery_record"] = record_resolved_delivery(run, guarded["delivery_exception"], old_raw, reports=reports)
    if validation["status"] == "NOT_REQUIRED" and not phase_done(validation):
        if args.check_only:
            delivery = verify_delivery(run)
            delivered = delivery.get("verified") is True and delivery.get("delivery_complete") is True
            if not delivered:
                print(json.dumps({**audit, "status": "READY_FOR_NOT_REQUIRED_CPU_DELIVERY",
                                  "delivery": delivery}, ensure_ascii=False), flush=True)
                return 0
            validation = {**validation, "delivery_complete": True, "delivery_proof": delivery}
        else:
            allocation = resume.current_allocation(args.previous_job_id)
            resume.validate_allocation_resources(allocation)
            audit["ops_source_archive"] = publish_ops_sources(run, reports=reports, repo=resume.REPO)
            delivery = ensure_not_required_delivery(run)
            validation = {**validation, "delivery_complete": True, "delivery_proof": delivery}
    if validation["status"] == "COMPLETE" and not phase_done(validation):
        raise ContinuationRefused("COMPLETE supplement status lacks verified delivery")
    resolution = required_phase_state(run, validation_state=validation, cost_state_fn=cost_state_fn)
    audit["required_phases"] = resolution
    if overall_done(resolution):
        print(json.dumps({**audit, "status": "ALL_REQUIRED_PHASES_COMPLETE_NO_MODEL_LOAD"},
                         ensure_ascii=False), flush=True)
        return 0
    phase, state = resolution["phase"], resolution["state"]
    if previous_phase == COST_PHASE and phase != COST_PHASE:
        raise ContinuationRefused("Required-phase state regressed behind registered observer-cost launch")
    if state["status"] == "COMPLETE":
        raise ContinuationRefused("COMPLETE observer-cost status lacks verified delivery")
    if state["status"] not in READY:
        raise ContinuationRefused("Core is complete but " + phase + " is not complete/ready: " + state["status"])
    def context(value):
        if context_fn is not None:
            return context_fn(run, value, reports=reports)
        return ready_context(run, value, reports=reports, phase=phase)
    plan, output = context(state)
    independent = launch_path(run, reports, phase=phase)
    expected_command = command(run, repo=resume.REPO, phase=phase)
    if previous_phase == phase and old.get(plan_key(phase)) != plan["plan_hash"]:
        raise ContinuationRefused("Supplement predecessor belongs to another sealed plan")
    if previous_phase == PHASE and phase == COST_PHASE and old.get("supplement_plan_hash") != validation.get("plan_hash"):
        raise ContinuationRefused("Observer-cost predecessor differs from verified validation plan")
    if args.check_only:
        print(json.dumps({**audit, "status": "READY_FOR_" + phase.upper() + "_HANDOFF",
                          "command": expected_command}, ensure_ascii=False), flush=True)
        return 0
    if not args.log:
        raise ContinuationRefused("Supplement handoff requires a scratch Slurm log")
    log = Path(args.log).resolve()
    if not log.is_relative_to(Path("/scratch/jgu7/moe-state-study/slurm")):
        raise ContinuationRefused("Supplement Slurm log must stay in scratch Slurm directory")
    allocation = resume.current_allocation(args.previous_job_id)
    resume.validate_allocation_resources(allocation)
    resume.validate_sources(run)
    if effective.read_bytes() != old_raw or (run / "launch.json").read_bytes() != core_raw:
        raise ContinuationRefused("Effective/core launch changed during supplement checks")
    if resume.verify_launch_process(run, old)["status"] == "VERIFIED_LIVE":
        raise ContinuationRefused("Effective predecessor PID became live during checks")
    terminal = resume.predecessor_terminal(args.previous_job_id)
    if terminal.get("status") != "VERIFIED_TERMINAL":
        raise ContinuationRefused("Effective predecessor is no longer verified terminal")
    guarded = phase_predecessor_guard(run, args.previous_job_id, terminal, resume,
        phase=previous_phase, launch=old, reports=reports)
    if guarded["stop"]:
        raise ContinuationRefused("Predecessor delivery retry no longer eligible: " + str(guarded["stop"]))
    fresh_resolution = required_phase_state(run, validation_state=state_fn(run), cost_state_fn=cost_state_fn)
    if overall_done(fresh_resolution):
        print(json.dumps({"status": "ALL_REQUIRED_PHASES_COMPLETE_NO_MODEL_LOAD",
                          "required_phases": fresh_resolution}), flush=True)
        return 0
    if fresh_resolution["phase"] != phase:
        raise ContinuationRefused("Required phase changed before claim; retry through its validated handoff")
    fresh_plan, fresh_output = context(fresh_resolution["state"])
    if fresh_plan["plan_hash"] != plan["plan_hash"] or fresh_output != output:
        raise ContinuationRefused("Supplement plan/status changed before claim")
    validation_terminal_proof = validation_terminal(run, reports, resume) if phase == COST_PHASE else None
    if phase == COST_PHASE and fresh_resolution["validation512"].get("status") == "COMPLETE" and validation_terminal_proof is None:
        raise ContinuationRefused("Completed validation512 lacks its actual independent launch")
    validation_launch_hash = (checksum(launch_path(run, reports).read_bytes())
        if phase == COST_PHASE and launch_path(run, reports).is_file() else None)
    core_boundary, manifest_raw = resume.capture_predecessor_boundary(run, audit["sources"]["protocol_hash"])
    supplement_boundary = checkpoint_snapshot(output, all_json=phase == COST_PHASE)
    ops_archive = publish_ops_sources(run, reports=reports, repo=resume.REPO)
    boundary = {**core_boundary, "phase": phase, "core_run_dir": str(run),
                plan_key(phase): plan["plan_hash"], "supplement_checkpoint": supplement_boundary}
    local = ops_dir(run, reports, phase=phase)
    local.mkdir(parents=True, exist_ok=True)
    history = resume.archive_launch_once(local, old_raw)
    proof = resume.publish_predecessor_boundary(local, old_raw, boundary, manifest_raw)
    if effective.read_bytes() != old_raw or (run / "launch.json").read_bytes() != core_raw:
        raise ContinuationRefused("Launch changed after supplement handoff claim")
    if (run / "manifest.json").read_bytes() != manifest_raw:
        raise ContinuationRefused("Core manifest changed after supplement boundary")
    if set(core_boundary["generation_files"]) != {
            str(p.relative_to(run)) for p in run.glob("**/generations/*.jsonl")}:
        raise ContinuationRefused("Core generation inventory changed after supplement boundary")
    for name, item in core_boundary["generation_files"].items():
        s = (run / name).stat()
        if (s.st_ino, s.st_size, s.st_mtime_ns) != (item["inode"], item["bytes"], item["mtime_ns"]):
            raise ContinuationRefused("Core generation bytes changed after supplement boundary")
    verify_checkpoint_snapshot(supplement_boundary)
    if phase == COST_PHASE:
        current_validation_hash = (checksum(launch_path(run, reports).read_bytes())
            if launch_path(run, reports).is_file() else None)
        if current_validation_hash != validation_launch_hash:
            raise ContinuationRefused("Validation512 launch changed after observer-cost claim")
    self_stat = resume._proc_stat(Path("/proc/self/stat").read_text())
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    claim = {"schema_version": schema_prefix(phase) + "_handoff_claim.v1", "task": phase,
             "run_dir": str(run), "plan_hash": plan["plan_hash"],
             "protocol_hash": audit["sources"]["protocol_hash"],
             "source_hash": audit["sources"]["source_hash"],
             "slurm_job_id": allocation["job_id"], "pid": os.getpid(),
             "owner_uid": os.getuid(),
             "proc_start_ticks": self_stat["start_ticks"], "boot_id": boot_id,
             "hostname": socket.gethostname(), "previous_launch_sha256": checksum(old_raw)}
    claim_path = local / "claims" / (checksum(old_raw) + ".json")
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    _publish(claim_path, claim)
    value = {"schema_version": schema_prefix(phase) + "_launch.v1", "phase": phase, "task": phase,
             "run_dir": str(run), "pid": os.getpid(), "proc_start_ticks": self_stat["start_ticks"],
             "boot_id": boot_id,
             "hostname": socket.gethostname(), "started_utc": utc(),
             "slurm_job_id": allocation["job_id"], "command": expected_command,
             "log": str(log), "cgroup": allocation["cgroup"],
             "source_hash": audit["sources"]["source_hash"],
             "protocol_hash": audit["sources"]["protocol_hash"],
             "plan_hash": plan["plan_hash"], plan_key(phase): plan["plan_hash"],
             "supplement_output_dir": str(output),
             "core_launch_sha256": checksum(core_raw), "previous_launch_sha256": checksum(old_raw),
             "previous_slurm_job_id": args.previous_job_id,
             "previous_effective_launch": str(effective), "previous_launch_history": str(history),
             "predecessor_snapshot": proof, "predecessor_terminal_proof": terminal,
             "core_terminal_proof": old.get("core_terminal_proof", terminal),
             "shared_handoff_claim": {"path": str(claim_path),
                                      "sha256": checksum(claim_path.read_bytes())},
             "current_allocation_proof": allocation, "metadata_written_before_exec": True,
             "exec_preserves_pid": True}
    value["ops_source_sha256"] = ops_archive["files"]
    value["ops_source_archive"] = ops_archive
    exception = guarded["delivery_exception"]
    if exception and exception["mode"] == "cpu_delivery_retry":
        if phase != previous_phase:
            raise ContinuationRefused("CPU delivery retry cannot change scientific phase")
        intent_path = Path(exception["retry_intent_path"])
        intent_path.parent.mkdir(parents=True, exist_ok=True)
        intent = {**exception, "schema_version": "state_cpu_delivery_retry.v1", "run_dir": str(run),
            "created_at_utc": utc(), "previous_launch_sha256": checksum(old_raw),
            "job_id": value["slurm_job_id"], "pid": value["pid"], "owner_uid": os.getuid(),
            "boot_id": value["boot_id"], "proc_start_ticks": value["proc_start_ticks"],
            "hostname": value["hostname"], "fixed_command": expected_command}
        _publish(intent_path, intent)
        value["cpu_delivery_retry"] = {"path": str(intent_path), "sha256": checksum(intent_path.read_bytes()),
                                       "retry_key": exception["retry_key"], "attempt": 1}
    elif exception:
        value["resolved_delivery_predecessor"] = exception
    if phase == COST_PHASE:
        value["observer_cost_output_dir"] = str(output)
        value["validation512_prerequisite"] = fresh_resolution["validation512"]
        value["validation512_plan_hash"] = fresh_resolution["validation512"].get("plan_hash")
        value["validation512_launch_sha256"] = validation_launch_hash
        value["validation512_terminal_proof"] = validation_terminal_proof
    resume.atomic_json(independent, value)
    continuation = None
    try:
        continuation = resume.ensure_successor(run, allocation["job_id"], phase=phase)
        outcome = resume.record_outcome(run, allocation["job_id"], continuation)
        print(json.dumps({"continuation": continuation, "outcome_path": outcome}), flush=True)
    except Exception as exc:
        print(json.dumps({"status": "CONTINUATION_FAILED_NOT_QUEUED", "error": repr(exc),
                          "supplement_continues": True}), file=sys.stderr, flush=True)
    if continuation and continuation.get("status") == "SCIENTIFIC_STOP":
        raise ContinuationRefused("Supplement scientific error appeared before exec")
    if continuation and continuation.get("status") == "ALREADY_COMPLETE_NO_SUBMISSION":
        return 0
    print(json.dumps({"status": "EXEC_" + phase.upper(), "pid": os.getpid(),
                      "launch": str(independent), "command": expected_command}), flush=True)
    os.chdir(resume.REPO)
    try:
        os.execvpe(expected_command[0], expected_command, dict(os.environ))
    except OSError as exc:
        resume.atomic_json(local / ("launch_exec_failure." + allocation["job_id"] + ".json"),
                           {"utc": utc(), "error": repr(exc), "launch": value})
        raise ContinuationRefused("Supplement exec failed after preserved independent handoff") from exc
