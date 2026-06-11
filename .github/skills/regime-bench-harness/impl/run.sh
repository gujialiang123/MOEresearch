#!/usr/bin/env bash
# Thin wrapper around harness/run_bench.py so the skill machinery can call it.
# All real logic lives in harness/. This file is intentionally a one-liner so
# that hand-running `python harness/run_bench.py ...` and skill-invocation are
# the same.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." >/dev/null 2>&1 && pwd)"

exec "$(command -v python3 || command -v python)" "$REPO_ROOT/harness/run_bench.py" "$@"
