"""harness/env_snapshot.py — capture hostname, GPU, driver, CUDA, library
versions, git state. Written into summary.json["environment"] so that a result
from 6 months ago is self-explanatory.
"""
from __future__ import annotations

import socket
import subprocess
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, check=False, cwd=cwd
        )
        return out.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def _gpu_info(gpu_id: int) -> dict[str, Any]:
    """Query nvidia-smi for one GPU's name/uuid/sm. All failures → empty strings."""
    raw = _run([
        "nvidia-smi",
        f"--id={gpu_id}",
        "--query-gpu=name,uuid,compute_cap",
        "--format=csv,noheader",
    ])
    if not raw:
        return {"name": "", "uuid": "", "id": gpu_id, "sm": ""}
    parts = [p.strip() for p in raw.split(",")]
    name = parts[0] if len(parts) > 0 else ""
    uuid = parts[1] if len(parts) > 1 else ""
    cc = parts[2] if len(parts) > 2 else ""
    # "9.0" → "90"
    sm = cc.replace(".", "") if cc else ""
    return {"name": name, "uuid": uuid, "id": gpu_id, "sm": sm}


def _driver_version() -> str:
    raw = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if not raw:
        return ""
    return raw.splitlines()[0].strip()


def _cuda_version(conda_env: str | None) -> str:
    """Try nvcc from the conda env, then system PATH."""
    candidates = []
    if conda_env:
        candidates.append(Path.home() / ".conda" / "envs" / conda_env / "bin" / "nvcc")
    candidates.append(Path("nvcc"))
    for nvcc in candidates:
        out = _run([str(nvcc), "--version"])
        if out:
            # Parse "release 12.4," → "12.4"
            for line in out.splitlines():
                if "release" in line:
                    chunks = line.split("release")
                    if len(chunks) > 1:
                        tail = chunks[1].strip()
                        return tail.split(",")[0].strip()
    return ""


def _lib_versions(conda_env: str) -> dict[str, str]:
    """pip show for sglang/flashinfer/vllm/torch/triton in the given conda env."""
    py = Path.home() / ".conda" / "envs" / conda_env / "bin" / "python"
    if not py.exists():
        py = Path("python")
    versions: dict[str, str] = {}
    pkgs = ["sglang", "flashinfer-python", "flashinfer", "vllm", "torch", "triton"]
    for pkg in pkgs:
        out = _run([
            str(py), "-c",
            f"import importlib.metadata as m;"
            f"\nprint(m.version({pkg!r}))",
        ])
        if out:
            # Drop dup keys: "flashinfer-python" vs "flashinfer" — first hit wins
            key = pkg.replace("-python", "")
            versions.setdefault(key, out.strip())
    return versions


def _git_info() -> dict[str, Any]:
    commit = _run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    if not commit:
        return {"commit": "", "dirty": False}
    status = _run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"])
    return {"commit": commit, "dirty": bool(status)}


def snapshot(gpu_id: int, conda_env: str) -> dict[str, Any]:
    """One-shot environment snapshot for summary.json["environment"]."""
    return {
        "hostname": socket.gethostname(),
        "gpu": _gpu_info(gpu_id),
        "driver": _driver_version(),
        "cuda": _cuda_version(conda_env),
        "engine_version": _lib_versions(conda_env),
        "git": _git_info(),
    }
