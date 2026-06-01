"""Shared utilities for Stage 1 scripts.

Keep this module tiny and dependency-light: stdlib + PyYAML only.
"""
from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


# ---------- paths ----------

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


# ---------- yaml / json ----------

def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def save_yaml(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def append_jsonl(record: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------- time ----------

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def now_compact() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------- networking ----------

def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def free_port_or_die(host: str, port: int) -> None:
    if port_in_use(host, port):
        sys.exit(f"[fatal] port {host}:{port} already in use; aborting to avoid collision.")


# ---------- subprocess ----------

def kill_process_group(pid: int | None, sig: int = signal.SIGTERM, wait_s: float = 10.0) -> None:
    """Kill a process group rooted at pid. Safe to call with None or dead PID."""
    if pid is None:
        return
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return
    deadline = time.time() + wait_s
    while time.time() < deadline:
        try:
            os.killpg(pgid, 0)  # signal 0 = poll
        except ProcessLookupError:
            return
        time.sleep(0.2)
    # Hard kill
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return


# ---------- log analysis ----------

_OOM_PATTERNS = [
    r"out of memory",
    r"CUDA out of memory",
    r"OutOfMemoryError",
    r"KV cache pool is full",
    r"RuntimeError: CUDA error: out of memory",
]
_CRASH_PATTERNS = [
    r"Traceback \(most recent call last\)",
    r"Segmentation fault",
    r"FATAL",
    r"core dumped",
    r"Aborted \(core dumped\)",
    r"RuntimeError: CUDA error",
    r"NCCL error",
]


def _scan(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def detect_oom(text: str) -> bool:
    return _scan(text, _OOM_PATTERNS)


def detect_server_crash(text: str) -> bool:
    return _scan(text, _CRASH_PATTERNS)


def read_text_safe(path: str | Path, max_bytes: int = 2_000_000) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    try:
        size = p.stat().st_size
        with open(p, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            return f.read().decode("utf-8", errors="ignore")
    except OSError:
        return ""


# ---------- sglang config ----------

# Keys allowed in configs/*.yaml whose names should NOT be passed to sglang launch_server.
PRIVATE_CONFIG_KEYS = {"_gpu_id", "_comment"}

# Bool flags that translate to "include flag if true, omit if false" (no value).
BOOL_FLAG_KEYS = {
    "disable-radix-cache",
    "disable-cuda-graph",
    "trust-remote-code",
    "log-requests",
    "enable-metrics",
    "enable-torch-compile",
    "enable-piecewise-cuda-graph",
    "skip-tokenizer-init",
}


def yaml_config_to_argv(config: dict) -> list[str]:
    """Convert a YAML config dict into sglang.launch_server argv tail.

    Returns a list like ['--model-path', '/x', '--port', '30000', '--disable-radix-cache'].
    Sentinel values like -1 (chunked-prefill-size etc.) are passed through as strings;
    callers can override by stripping these keys before calling.
    """
    argv: list[str] = []
    for key, val in config.items():
        if key in PRIVATE_CONFIG_KEYS:
            continue
        if val is None:
            continue
        flag = f"--{key}"
        if key in BOOL_FLAG_KEYS or isinstance(val, bool):
            if bool(val):
                argv.append(flag)
            continue
        if isinstance(val, list):
            # nargs='+' style: --flag v1 v2 v3
            argv.append(flag)
            argv.extend(str(v) for v in val)
            continue
        argv.extend([flag, str(val)])
    return argv


SGLANG_PY = "/home/t-jialianggu/.conda/envs/sglang-dev/bin/python"
SGLANG_CONDA_ENV = "sglang-dev"


def conda_run_argv(python_module_argv: list[str], conda_env: str = SGLANG_CONDA_ENV) -> list[str]:
    """Wrap a `python -m ...` argv into `conda run -n <env> --no-capture-output ...`.

    This is the cleanest way to inherit all conda activate hooks (LDFLAGS,
    PATH, LIBRARY_PATH for libcudart, etc.) without re-implementing them.
    `--no-capture-output` lets us redirect stdout/stderr to our log files.
    """
    return ["conda", "run", "-n", conda_env, "--no-capture-output",
            "python", *python_module_argv]


def env_for_config(config: dict, base_env: dict | None = None,
                   conda_env: str = SGLANG_CONDA_ENV) -> dict:
    """Build env dict for the sglang server / bench_serving process.

    - Honors `_gpu_id` from YAML → CUDA_VISIBLE_DEVICES.
    - Sets CUDA_HOME explicitly. `conda run` activates the cuda-nvcc hook
      (which fixes LDFLAGS / LIBRARY_PATH for libcudart) but does NOT set
      CUDA_HOME. sglang's cudagraph_runner explicitly checks CUDA_HOME.
    - Redirects HF cache to /data/hf/gujialiang123/hf_cache because the
      shared /data/hf/hub is owned by another user and not writable.
      bench_serving downloads ShareGPT for the `random` dataset and needs
      a writable cache. This redirect ONLY affects spawned subprocesses;
      it does not modify the user's shell env.
    """
    env = dict(base_env if base_env is not None else os.environ)
    if "_gpu_id" in config and config["_gpu_id"] is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(config["_gpu_id"])
    home = Path.home()
    envdir = home / ".conda" / "envs" / conda_env
    if envdir.exists():
        env["CUDA_HOME"] = str(envdir)

    # HF cache: always pin to the user-owned location under /data/hf.
    user_hf = Path("/data/hf/gujialiang123/hf_cache")
    user_hf.mkdir(parents=True, exist_ok=True)
    (user_hf / "hub").mkdir(parents=True, exist_ok=True)
    env["HF_HOME"] = str(user_hf)
    env["HF_HUB_CACHE"] = str(user_hf / "hub")
    return env


# ---------- experiment dir ----------

def next_run_dir(root: Path, prefix: str = "run") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(int(p.name.split("_")[-1])
                      for p in root.glob(f"{prefix}_*")
                      if p.is_dir() and p.name.split("_")[-1].isdigit())
    n = (existing[-1] + 1) if existing else 1
    d = root / f"{prefix}_{n:04d}"
    d.mkdir()
    return d
