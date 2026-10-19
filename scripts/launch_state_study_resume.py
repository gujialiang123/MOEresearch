#!/usr/bin/env python3
"""Checked Slurm handoff into the unchanged, frozen state-study auto runner.

No model is imported here. --check-only never writes launch metadata or execs.
Actual handoff requires the predecessor's positively verified terminal Slurm
record, a different live Slurm allocation, and unchanged frozen source bytes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import uuid

from summarize_state_study_results import _proc_stat, verify_launch_process
from state_study_continuation import (ContinuationRefused, ensure_successor,
                                      predecessor_stop, record_outcome,
                                      scientific_error, validate_allocation_resources)
from state_study_quality_watch import launch_watch_start


REPO = Path(__file__).resolve().parents[1]
RUNS = Path("/scratch/jgu7/moe-state-study/runs")
TERMINAL = {"BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL",
            "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT"}


class Refused(RuntimeError):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def checksum(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return checksum(json.dumps(value,sort_keys=True,ensure_ascii=False,
        separators=(",",":"),allow_nan=False).encode())


def json_read(path):
    with Path(path).open(encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result,dict):
        raise Refused(f"Expected a JSON object: {path}")
    return result


def validate_sources(run, repo=REPO):
    run, repo = Path(run).resolve(), Path(repo).resolve()
    frozen = json_read(run/"source_hashes.json")
    sources = [*repo.glob("moe_research/*.py"), repo/"scripts/_harness.py",
               repo/"scripts/run_state_calibration_study.py", repo/"configs/state_calibration_study.yaml"]
    actual = {str(path.relative_to(repo)):checksum(path.read_bytes()) for path in sorted(sources)}
    if len(frozen) != 18 or set(frozen) != set(actual):
        raise Refused("Frozen source inventory differs from the required 18 files; do not resume this run")
    mismatches = [name for name in frozen if frozen[name] != actual[name]]
    if mismatches:
        raise Refused("Frozen source bytes changed: "+", ".join(mismatches))
    protocol = json_read(run/"protocol.json")
    if protocol.get("source_hash") != canonical(frozen):
        raise Refused("protocol source_hash does not match source_hashes.json")
    body = {key:value for key,value in protocol.items() if key != "protocol_hash"}
    if protocol.get("protocol_hash") != canonical(body):
        raise Refused("protocol.json self hash is inconsistent")
    manifest = json_read(run/"manifest.json")
    if manifest.get("protocol_hash") != protocol["protocol_hash"]:
        raise Refused("manifest/protocol identity mismatch")
    return {"file_count":len(frozen),"source_hash":canonical(frozen),
            "protocol_hash":protocol["protocol_hash"]}, manifest


def completed(run, manifest):
    stages = manifest.get("stages",{})
    report = Path(run)/"RUN_REPORT.md"
    return (stages.get("P5",{}).get("status") == "complete" and
            stages.get("P6_P7",{}).get("status") == "gated_pending" and
            report.is_file() and report.stat().st_size > 0)


def terminal_state(value):
    # A trailing '+' can mean truncation/flags; never infer a terminal base from
    # such an ambiguous value. sacct may append a cancellation UID explicitly.
    if value in TERMINAL:
        return value
    if re.fullmatch(r"CANCELLED by [0-9]+",value):
        return "CANCELLED"
    return None


def valid_end(value):
    if not isinstance(value,str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:[+-][0-9]{2}:[0-9]{2}|Z)?",value):
        return False
    try:
        return datetime.fromisoformat(value.replace("Z","+00:00")).year >= 1970
    except ValueError:
        return False


def query(command):
    record = {"command":command,"checked_at_utc":utc()}
    try:
        response = subprocess.run(command,text=True,capture_output=True,timeout=20,check=False)
        record.update(returncode=response.returncode,stdout=response.stdout,stderr=response.stderr)
    except (OSError,subprocess.TimeoutExpired) as exc:
        record.update(returncode=None,error=str(exc))
    return record


def job_record(output, job_id):
    records = [line for line in output.split("\n") if re.search(r"(?:^|\s)JobId="+re.escape(str(job_id))+r"(?:\s|$)",line)]
    if len(records) != 1:
        return None
    return dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_:/-]*)=([^\s]+)",records[0]))


def record_absent(response):
    if response.get("returncode") in (None,0):
        return False
    lines = [line.strip() for key in ("stdout","stderr") for line in response.get(key,"").split("\n") if line.strip()]
    pattern = re.compile(r"(?:(?:scontrol:\s*)?(?:error:\s*)?)?(?:slurm_load_jobs error:\s*)?Invalid job id specified",re.I)
    return bool(lines) and all(pattern.fullmatch(line) for line in lines)


def predecessor_terminal(job_id, query_fn=query):
    """Current controller state wins; accounting fallback only on explicit miss."""
    control = query_fn(["scontrol","-o","show","job",str(job_id)])
    proof = {"status":"UNKNOWN","job_id":str(job_id),"scontrol":control}
    if control.get("returncode") == 0:
        record = job_record(control.get("stdout",""),job_id)
        if record is None:
            return {**proof,"reason":"scontrol_exact_job_record_missing_or_ambiguous"}
        state = terminal_state(record.get("JobState",""))
        if not state:
            return {**proof,"reason":"scontrol_current_state_not_terminal","observed_state":record.get("JobState")}
        if not valid_end(record.get("EndTime")):
            return {**proof,"reason":"scontrol_terminal_record_has_invalid_EndTime"}
        return {**proof,"status":"VERIFIED_TERMINAL","source":"scontrol",
                "state":state,"end":record["EndTime"],"nodes":record.get("NodeList")}
    if not record_absent(control):
        return {**proof,"reason":"scontrol_unknown_error_no_accounting_fallback"}
    accounting = query_fn(["sacct","-j",str(job_id),"--noheader","--parsable2",
                           "--format=JobIDRaw,State%64,NodeList%256,End%64"])
    proof["sacct"] = accounting
    if accounting.get("returncode") != 0:
        return {**proof,"reason":"sacct_lookup_failed"}
    records = []
    for line in accounting.get("stdout","").split("\n"):
        fields = line.strip().split("|")
        if fields and fields[0] == str(job_id):
            if len(fields) != 4:
                return {**proof,"reason":"sacct_exact_record_has_unexpected_fields"}
            records.append(fields)
    if len(records) != 1:
        return {**proof,"reason":"sacct_exact_job_record_missing_or_conflicting"}
    _, value, nodes, end = records[0]
    state = terminal_state(value)
    if not state or not valid_end(end):
        return {**proof,"reason":"sacct_exact_record_not_unambiguously_terminal_with_valid_End",
                "observed_state":value,"observed_end":end}
    return {**proof,"status":"VERIFIED_TERMINAL","source":"sacct_after_explicit_scontrol_absence",
            "state":state,"nodes":nodes,"end":end}


def original_command(launch, run, repo=REPO):
    command = launch.get("command")
    if not isinstance(command,list) or not command or not all(isinstance(part,str) for part in command):
        raise Refused("Missing original launch.command")
    expected = [sys.executable,"-u",str(Path(repo)/"scripts/run_state_calibration_study.py"),
                "auto","--run-dir",str(Path(run).resolve())]
    if command not in (expected,expected+["--config",str(Path(repo)/"configs/state_calibration_study.yaml")]):
        raise Refused("Original command differs from the exact frozen auto/interpreter/run/config contract")
    return command


def fsync_directory(directory):
    descriptor = os.open(directory,os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name("."+path.name+"."+uuid.uuid4().hex+".pending")
    try:
        with temporary.open("x",encoding="utf-8") as handle:
            json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)
            handle.write("\n");handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def archive_launch_once(run, raw):
    """Immutable history publication also claims exactly one old-launch handoff.

    Exclusive hard-link publication, rather than an advisory NFS lock, prevents
    two separately queued successors from replacing the same launch record.
    A crash after this claim requires explicit review; claims are never deleted.
    """
    directory = Path(run)/"launch_history"
    directory.mkdir(exist_ok=True)
    destination = directory/(checksum(raw)+".json")
    temporary = directory/("."+uuid.uuid4().hex+".pending")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw);handle.flush();os.fsync(handle.fileno())
        try:
            os.link(temporary,destination)
        except FileExistsError as exc:
            raise Refused("This predecessor launch was already claimed; inspect launch_history/current launch before any retry") from exc
        fsync_directory(directory)
        fsync_directory(Path(run))
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def capture_predecessor_boundary(run, protocol_hash):
    """Read an unchanged append boundary after the old allocation has ended."""
    run = Path(run).resolve()
    manifest_path = run/"manifest.json"
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    if not isinstance(manifest,dict) or manifest.get("protocol_hash") != protocol_hash:
        raise Refused("Predecessor manifest is invalid or has a different protocol")
    files = sorted(run.glob("**/generations/*.jsonl"))
    boundaries = {}
    for path in files:
        if not path.resolve().is_relative_to(run):
            raise Refused(f"Generation evidence escapes the run: {path}")
        digest = hashlib.sha256()
        lines, count_bytes, last_id, last_line_hash, seen = 0, 0, None, None, set()
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            for raw in handle:
                lines += 1
                if not raw.endswith(b"\n"):
                    raise Refused(f"Incomplete physical JSONL record at {path}:{lines}; preserve raw for explicit review")
                try:
                    def reject_constant(value):
                        raise ValueError(f"non-finite JSON constant {value}")
                    row = json.loads(raw,parse_constant=reject_constant)
                except (ValueError,UnicodeError) as exc:
                    raise Refused(f"Invalid physical JSONL record at {path}:{lines}: {exc}") from exc
                if not isinstance(row,dict) or isinstance(row.get("sample_id"),bool) or not isinstance(row.get("sample_id"),(int,str)):
                    raise Refused(f"Missing/invalid sample_id at {path}:{lines}")
                if row["sample_id"] in seen:
                    raise Refused(f"Duplicate sample_id at {path}:{lines}; no raw repair is performed")
                seen.add(row["sample_id"])
                last_id = row["sample_id"]
                last_line_hash = checksum(raw)
                digest.update(raw)
                count_bytes += len(raw)
            after = os.fstat(handle.fileno())
        current = path.stat()
        identity = lambda stat:(stat.st_ino,stat.st_size,stat.st_mtime_ns)
        if identity(before) != identity(after) or identity(before) != identity(current) or count_bytes != before.st_size:
            raise Refused(f"Generation file changed while reading predecessor boundary: {path}")
        boundaries[str(path.relative_to(run))] = {
            "bytes":count_bytes,"physical_lines":lines,"record_count":lines,
            "last_sample_id":last_id,"sha256":digest.hexdigest(),
            "last_record_sha256":last_line_hash,"mtime_ns":before.st_mtime_ns,
            "inode":before.st_ino}
    if files != sorted(run.glob("**/generations/*.jsonl")) or manifest_raw != manifest_path.read_bytes():
        raise Refused("Predecessor manifest/generation inventory changed during boundary capture")
    snapshot = {"schema_version":"state_study_handoff_boundary.v1","captured_at_utc":utc(),
        "protocol_hash":protocol_hash,"manifest_sha256":checksum(manifest_raw),
        "manifest_bytes":len(manifest_raw),"generation_files":boundaries,
        "generation_file_count":len(boundaries),"total_physical_lines":sum(item["physical_lines"] for item in boundaries.values()),
        "append_origin_rule":"Rows before each physical_lines/bytes boundary already existed before this handoff. Only later appended segments may originate in the next allocation; follow the launch-history chain for repeated handoffs. No per-row timestamp is invented.",
        "timing_caveat":"The old manifest runtime is preserved verbatim. Warmups, replay/fitting, and unrecorded failed attempts are not inferred from generation-row boundaries."}
    return snapshot,manifest_raw


def publish_predecessor_boundary(run, old_launch_raw, snapshot, manifest_raw):
    """Publish immutable boundary files; never replace prior provenance."""
    run = Path(run)
    parent = run/"launch_history/boundaries"
    parent.mkdir(exist_ok=True)
    directory = parent/checksum(old_launch_raw)
    try:
        directory.mkdir()
    except FileExistsError as exc:
        raise Refused("Predecessor boundary directory already exists; preserve it for explicit review") from exc
    evidence = {**snapshot,"previous_launch_sha256":checksum(old_launch_raw),
                "manifest_snapshot":"manifest.json"}
    boundary_raw = (json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()
    for name,raw in (("manifest.json",manifest_raw),("boundary.json",boundary_raw)):
        with (directory/name).open("xb") as handle:
            handle.write(raw);handle.flush();os.fsync(handle.fileno())
    commit = {"schema_version":"state_study_handoff_boundary_commit.v1",
              "manifest_sha256":checksum(manifest_raw),"boundary_sha256":checksum(boundary_raw)}
    with (directory/"complete.json").open("x",encoding="utf-8") as handle:
        json.dump(commit,handle,sort_keys=True);handle.write("\n");handle.flush();os.fsync(handle.fileno())
    fsync_directory(directory);fsync_directory(parent)
    return {"manifest_path":str(directory/"manifest.json"),"manifest_sha256":checksum(manifest_raw),
            "boundary_path":str(directory/"boundary.json"),"boundary_sha256":checksum(boundary_raw),
            "commit_path":str(directory/"complete.json"),
            "generation_file_count":snapshot["generation_file_count"],
            "total_physical_lines":snapshot["total_physical_lines"]}


def current_allocation(previous_job, environment=None, proc_root=Path("/proc"), query_fn=query):
    environment = os.environ if environment is None else environment
    job = environment.get("SLURM_JOB_ID","")
    if not job.isdigit() or job == str(previous_job):
        raise Refused("Actual resume needs a different Slurm allocation; submit with --dependency=afterany:OLD_JOB")
    cgroup = (Path(proc_root)/"self/cgroup").read_text()
    if not re.search(r"(?:^|/)job_"+re.escape(job)+r"(?:/|$)",cgroup,re.M):
        raise Refused("Current /proc cgroup does not belong to SLURM_JOB_ID")
    response = query_fn(["scontrol","-o","show","job",job])
    record = job_record(response.get("stdout",""),job) if response.get("returncode") == 0 else None
    if not record or record.get("JobState") != "RUNNING" or not record.get("BatchFlag","").isdigit() or int(record["BatchFlag"]) < 1:
        raise Refused("Current allocation is not a verified RUNNING Slurm batch job")
    owner = re.search(r"\(([0-9]+)\)$",record.get("UserId",""))
    if not owner or int(owner.group(1)) != os.getuid():
        raise Refused("Current batch allocation owner does not match this user")
    return {"job_id":job,"cgroup":cgroup,"scontrol":response,
            "dependency_environment":environment.get("SLURM_JOB_DEPENDENCY"),
            "dependency_policy":"submit only as afterany of the exact previous launch job; satisfied dependencies may be removed by Slurm"}


def dispatch_supplement(args, run, audit, manifest):
    # Shared launcher dispatch also reaches batch bodies queued before this
    # supplement existed. Required phases are validation512 then observer_cost;
    # neither a core nor a validation-only completion ends the entire chain.
    # The scientific core keeps its original same-PID exec contract.
    from state_study_supplement_ops import dispatch
    return dispatch(args, run, audit, manifest, sys.modules[__name__])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",required=True)
    parser.add_argument("--previous-job-id",required=True)
    parser.add_argument("--log",help="Slurm output path recorded in the new launch metadata")
    parser.add_argument("--check-only",action="store_true",help="Read-only source/process/scheduler checks; never updates metadata or execs")
    args = parser.parse_args(argv)
    run = Path(args.run_dir).resolve()
    audit = {"checked_at_utc":utc(),"run_dir":str(run),"check_only":args.check_only}
    try:
        if not run.is_dir() or run == RUNS.resolve() or not run.is_relative_to(RUNS.resolve()):
            raise Refused("Only an existing run below /scratch/jgu7/moe-state-study/runs is allowed")
        if not args.previous_job_id.isdigit():
            raise Refused("previous-job-id must be one exact numeric Slurm job ID")
        audit["sources"], manifest = validate_sources(run)
        errors = scientific_error(manifest,run)
        if errors:
            audit["scientific_errors"] = errors
            raise Refused("Explicit scientific error in current manifest/stage; no claim, queue, or model load")
        if completed(run,manifest):
            return dispatch_supplement(args,run,audit,manifest)
        launch_path = run/"launch.json"
        old_raw = launch_path.read_bytes()
        old = json.loads(old_raw)
        if not isinstance(old,dict):
            raise Refused("launch.json must be an object")
        if str(old.get("slurm_job_id")) != args.previous_job_id:
            raise Refused("previous-job-id differs from current launch.json; do not resume a stale predecessor")
        command = original_command(old,run)
        audit["old_process"] = verify_launch_process(run,old)
        if audit["old_process"]["status"] == "VERIFIED_LIVE":
            raise Refused("Exact predecessor PID is still VERIFIED_LIVE; refusing duplicate launch")
        audit["previous_job"] = predecessor_terminal(args.previous_job_id)
        if audit["previous_job"]["status"] != "VERIFIED_TERMINAL":
            raise Refused("Predecessor Slurm state is UNKNOWN/nonterminal; refusing to replace launch metadata")
        stop = predecessor_stop(run,args.previous_job_id,audit["previous_job"])
        if stop:
            audit["predecessor_stop"] = stop
            raise Refused("Predecessor failure/cancellation needs review; no automatic retry")
        if args.check_only:
            print(json.dumps({**audit,"status":"READY_FOR_NEW_AFTERANY_BATCH_ALLOCATION","command":command},ensure_ascii=False,indent=2),flush=True)
            return 0
        allocation = current_allocation(args.previous_job_id)
        validate_allocation_resources(allocation)
        # Recheck the frozen bytes and predecessor snapshot immediately before
        # the one-time publication; no change to study.lock is ever made here.
        validate_sources(run)
        if launch_path.read_bytes() != old_raw:
            raise Refused("launch.json changed during checks")
        if verify_launch_process(run,old)["status"] == "VERIFIED_LIVE":
            raise Refused("Predecessor PID became live during checks")
        second_job = predecessor_terminal(args.previous_job_id)
        if second_job["status"] != "VERIFIED_TERMINAL":
            raise Refused("Predecessor job no longer verified terminal at handoff")
        snapshot, old_manifest_raw = capture_predecessor_boundary(run,audit["sources"]["protocol_hash"])
        self_stat = _proc_stat(Path("/proc/self/stat").read_text())
        launch = {"pid":os.getpid(),"proc_start_ticks":self_stat["start_ticks"],
            "boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "hostname":socket.gethostname(),"started_utc":utc(),"slurm_job_id":allocation["job_id"],
            "command":command,"log":args.log,"cgroup":allocation["cgroup"],
            "source_hash":audit["sources"]["source_hash"],"protocol_hash":audit["sources"]["protocol_hash"],
            "previous_launch_sha256":checksum(old_raw),"previous_slurm_job_id":args.previous_job_id,
            "predecessor_terminal_proof":second_job,"current_allocation_proof":allocation,
            "launcher":str(Path(__file__).resolve()),"launcher_sha256":checksum(Path(__file__).read_bytes()),
            "metadata_written_before_exec":True,"exec_preserves_pid":True}
        if not args.log:
            raise Refused("Actual handoff requires --log pointing at its Slurm output file")
        log = Path(args.log).resolve()
        if not log.is_relative_to(Path("/scratch/jgu7/moe-state-study/slurm").resolve()):
            raise Refused("Slurm log must be under /scratch/jgu7/moe-state-study/slurm")
        launch["log"] = str(log)
        history = archive_launch_once(run,old_raw)
        launch["previous_launch_history"] = str(history)
        launch["predecessor_snapshot"] = publish_predecessor_boundary(run,old_raw,snapshot,old_manifest_raw)
        # Publication may take some I/O; an unexpected writer must never be
        # silently attributed to the next allocation.
        if (run/"manifest.json").read_bytes() != old_manifest_raw:
            raise Refused("Predecessor manifest changed after boundary publication")
        if set(snapshot["generation_files"]) != {str(path.relative_to(run)) for path in run.glob("**/generations/*.jsonl")}:
            raise Refused("Generation inventory changed after boundary publication")
        for name,item in snapshot["generation_files"].items():
            stat = (run/name).stat()
            if (stat.st_ino,stat.st_size,stat.st_mtime_ns) != (item["inode"],item["bytes"],item["mtime_ns"]):
                raise Refused("Generation bytes changed after boundary publication")
        atomic_json(launch_path,launch)
        # One short ops action, after the original verified handoff and before
        # the same-PID exec. Submission failure is explicit and cannot kill the
        # scientific process. The pending child rechecks completion/errors.
        continuation = None
        try:
            continuation = ensure_successor(run,allocation["job_id"])
            outcome_path = record_outcome(run,allocation["job_id"],continuation)
            print(json.dumps({"continuation":continuation,"outcome_path":outcome_path},
                             ensure_ascii=False),flush=True)
        except Exception as exc:
            print(json.dumps({"status":"CONTINUATION_FAILED_NOT_QUEUED",
                              "error":repr(exc),"scientific_auto_continues":True},
                             ensure_ascii=False),file=sys.stderr,flush=True)
        if continuation and continuation.get("status") == "SCIENTIFIC_STOP":
            raise Refused("Scientific error appeared before exec; no model load")
        if continuation and continuation.get("status") == "ALREADY_COMPLETE_NO_SUBMISSION":
            return 0
        try:
            print(json.dumps({"quality_watch_start":launch_watch_start(run)},
                             ensure_ascii=False),flush=True)
        except Exception as exc:
            print(json.dumps({"status":"QUALITY_WATCH_START_FAILED",
                              "error":repr(exc),"scientific_auto_continues":True},
                             ensure_ascii=False),file=sys.stderr,flush=True)
        print(json.dumps({"status":"EXEC_FROZEN_AUTO","pid":os.getpid(),"launch":str(launch_path),
                          "previous_launch_history":str(history),"command":command},ensure_ascii=False),flush=True)
        os.chdir(REPO)
        try:
            os.execvpe(command[0],command,dict(os.environ))
        except OSError as exc:
            atomic_json(run/("launch_exec_failure."+allocation["job_id"]+".json"),
                        {"utc":utc(),"pid":os.getpid(),"error":repr(exc),"launch":launch})
            raise Refused(f"exec failed after preserved handoff: {exc}") from exc
    except (Refused,ContinuationRefused,OSError,ValueError,KeyError) as exc:
        print(json.dumps({**audit,"status":"REFUSED_NO_MODEL_LOAD","error":str(exc)},ensure_ascii=False,indent=2),file=sys.stderr,flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
