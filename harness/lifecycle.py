"""harness/lifecycle.py — ServerLifecycle: port pre-check, launch via existing
scripts/launch_server.py, wait for /health, force-kill process group on exit.

User decisions (2026-06-11):
  - If target port is already in use → hard error, don't reuse.
  - On harness exit (success/fail/Ctrl-C) → force-kill the spawned server.
  - --keep-server flag suppresses cleanup (debug only).
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCH_SERVER = REPO_ROOT / "scripts" / "launch_server.py"


class LifecycleError(RuntimeError):
    pass


def _port_in_use(host: str, port: int, timeout_s: float = 1.0) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout_s)
    try:
        s.connect((host, port))
        s.close()
        return True
    except (ConnectionRefusedError, OSError):
        return False


def _parse_host_port(url: str, fallback_port: int = 30000) -> tuple[str, int]:
    p = urlparse(url)
    host = p.hostname or "127.0.0.1"
    port = p.port or fallback_port
    return host, port


class ServerLifecycle:
    """Manages one sglang/vllm server process for the duration of a harness run.

    Use as a context manager so cleanup runs even on exceptions / Ctrl-C:

        with ServerLifecycle(...) as lifecycle:
            lifecycle.start()
            lifecycle.wait_healthy()
            ...do bench work...
        # __exit__ kills the process group unless --keep-server.
    """

    def __init__(
        self,
        *,
        resolved_server_config: Mapping[str, Any],
        conda_env: str,
        base_url: str,
        health_url: str,
        startup_timeout_s: int,
        out_dir: Path,
        keep_server: bool = False,
    ):
        self.resolved = dict(resolved_server_config)
        self.conda_env = conda_env
        self.base_url = base_url
        self.health_url = health_url
        self.startup_timeout_s = startup_timeout_s
        self.out_dir = Path(out_dir)
        self.keep_server = keep_server

        self.server_log_path = self.out_dir / "server.log"
        self.server_config_path = self.out_dir / "server_config_used.yaml"
        self.pidfile = self.out_dir / "server.pid"
        self.proc: subprocess.Popen | None = None
        self._start_time: float | None = None
        self._healthy_time: float | None = None

    # -----------------------------------------------------------------
    # Context manager
    # -----------------------------------------------------------------

    def __enter__(self) -> "ServerLifecycle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.keep_server:
            if self.proc:
                print(
                    f"[lifecycle] --keep-server set; leaving pid={self.proc.pid} "
                    f"running (kill manually via: kill -- -{os.getpgid(self.proc.pid)})",
                    flush=True,
                )
            return
        self.stop()

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def preflight_port_check(self) -> None:
        """User decision: if port already in use, hard error (no reuse)."""
        host, port = _parse_host_port(
            self.base_url,
            fallback_port=int(self.resolved.get("port", 30000)),
        )
        if _port_in_use(host, port):
            raise LifecycleError(
                f"Port {host}:{port} is already in use. Harness will not start "
                f"a competing server. Identify the existing process with "
                f"`lsof -i :{port}` and stop it (if yours) or change the spec's "
                f"server.overrides.port to a free port."
            )

    def start(self) -> None:
        """Write the resolved server config to disk, then exec scripts/launch_server.py."""
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.preflight_port_check()

        # Persist the EXACT config used to launch the server.
        self.server_config_path.write_text(yaml.safe_dump(self.resolved, sort_keys=False))

        if not LAUNCH_SERVER.exists():
            raise LifecycleError(f"launch_server.py not found at {LAUNCH_SERVER}")

        # Use the project's `python` directly so this works even when called
        # from inside a different conda env. launch_server.py internally uses
        # `conda run -n <env>` to dispatch to the sglang env.
        cmd = [
            sys.executable, str(LAUNCH_SERVER),
            "--config", str(self.server_config_path),
            "--log", str(self.server_log_path),
            "--pidfile", str(self.pidfile),
            "--conda-env", self.conda_env,
        ]
        print(f"[lifecycle] launching: {' '.join(cmd)}", flush=True)

        self._start_time = time.perf_counter()
        # NOTE: launch_server.py itself uses start_new_session=True for the
        # SGLang process. Our wrapper here is short-lived (it just spawns and
        # exits), so we don't need start_new_session on this layer.
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise LifecycleError(
                f"launch_server.py exited {proc.returncode}: "
                f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
            )
        if proc.stdout:
            print(proc.stdout, end="", flush=True)

        # The actual sglang process is the pid written to pidfile.
        if not self.pidfile.exists():
            raise LifecycleError(
                f"launch_server.py succeeded but pidfile {self.pidfile} missing"
            )
        sglang_pid = int(self.pidfile.read_text().strip())
        # Wrap in a fake Popen so __exit__ can kill it.
        self.proc = _FakePopen(sglang_pid)

    def wait_healthy(self) -> None:
        """Poll health_url until 200 or timeout."""
        if self._start_time is None:
            raise LifecycleError("start() must be called before wait_healthy()")
        deadline = self._start_time + self.startup_timeout_s
        last_err = "no probes yet"
        attempt = 0
        while time.perf_counter() < deadline:
            attempt += 1
            try:
                r = requests.get(self.health_url, timeout=5)
                if r.status_code == 200:
                    self._healthy_time = time.perf_counter()
                    elapsed = self._healthy_time - self._start_time
                    print(
                        f"[lifecycle] healthy after {elapsed:.1f}s (attempt {attempt})",
                        flush=True,
                    )
                    return
                last_err = f"HTTP {r.status_code}"
            except requests.RequestException as e:
                last_err = str(e)
            # Check the underlying process didn't die.
            if self.proc and not self._proc_alive():
                raise LifecycleError(
                    f"Server process pid={self.proc.pid} exited before /health "
                    f"became ready. Inspect {self.server_log_path}"
                )
            time.sleep(3)
        raise LifecycleError(
            f"Server did not become healthy within {self.startup_timeout_s}s. "
            f"Last probe error: {last_err}. Inspect {self.server_log_path}"
        )

    def stop(self) -> None:
        """Force-kill the server process group (SIGTERM, then SIGKILL after grace)."""
        if not self.proc:
            return
        pid = self.proc.pid
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            print(f"[lifecycle] pid={pid} already exited", flush=True)
            self.proc = None
            return
        print(f"[lifecycle] killing pgid={pgid} (SIGTERM)...", flush=True)
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        # Grace period.
        grace = 15.0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < grace:
            if not self._proc_alive():
                break
            time.sleep(0.5)
        else:
            print(f"[lifecycle] pgid={pgid} still alive after {grace}s; SIGKILL", flush=True)
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        self.proc = None

    # -----------------------------------------------------------------
    # Telemetry
    # -----------------------------------------------------------------

    def startup_wall_s(self) -> float:
        if self._start_time is None or self._healthy_time is None:
            return 0.0
        return self._healthy_time - self._start_time

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _proc_alive(self) -> bool:
        if not self.proc:
            return False
        try:
            os.kill(self.proc.pid, 0)
            return True
        except (ProcessLookupError, OSError):
            return False


class _FakePopen:
    """Minimal Popen-like object holding just the pid. We use this because
    scripts/launch_server.py spawns the actual server in a separate session
    and writes its pid to a file; we don't have a real Popen handle here."""

    def __init__(self, pid: int):
        self.pid = pid
