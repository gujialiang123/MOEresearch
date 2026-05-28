#!/usr/bin/env python3
"""Launch a SGLang server from a YAML config.

Implements DESIGN.md §0.G B1: YAML → argv translation. Does NOT use --config
(no such flag exists in sglang). The server is started in its own process group
so it can be killed cleanly even if it spawns subprocesses.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from utils import (
    PROJECT_ROOT,
    SGLANG_CONDA_ENV,
    conda_run_argv,
    env_for_config,
    free_port_or_die,
    load_yaml,
    yaml_config_to_argv,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to sglang server YAML config.")
    ap.add_argument("--log", required=True, help="Path to write server stdout+stderr.")
    ap.add_argument("--pidfile", default=None, help="Optional file to write spawned PID.")
    ap.add_argument("--conda-env", default=SGLANG_CONDA_ENV)
    ap.add_argument("--dry-run", action="store_true", help="Print argv and exit.")
    args = ap.parse_args()

    config = load_yaml(args.config)
    host = config.get("host", "127.0.0.1")
    port = int(config.get("port", 30000))

    argv = conda_run_argv(
        ["-m", "sglang.launch_server", *yaml_config_to_argv(config)],
        conda_env=args.conda_env,
    )
    if args.dry_run:
        print(" ".join(argv))
        return 0

    free_port_or_die(host, port)

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = env_for_config(config, conda_env=args.conda_env)

    print(f"[launch_server] argv: {' '.join(argv)}", flush=True)
    print(f"[launch_server] CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES','(unset)')}",
          flush=True)
    print(f"[launch_server] log: {log_path}", flush=True)

    with open(log_path, "ab") as logf:
        proc = subprocess.Popen(
            argv,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(PROJECT_ROOT),
            start_new_session=True,  # become process group leader
        )

    if args.pidfile:
        Path(args.pidfile).write_text(str(proc.pid))

    print(f"[launch_server] pid={proc.pid} pgid={os.getpgid(proc.pid)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
