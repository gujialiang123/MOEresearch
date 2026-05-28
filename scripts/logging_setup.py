"""Logging helper: file + stdout simultaneously, with structured fields.

Usage:
    from logging_setup import setup_logger
    log = setup_logger("regime_suite", logfile="logs/regime_suite_20260528.log")
    log.info("starting", extra={"workload": "smoke"})
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logger(name: str, logfile: str | Path, level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(level)
    # Idempotent: clear handlers if called twice in the same process
    log.handlers.clear()
    log.propagate = False

    fmt = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FMT)

    Path(logfile).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)

    return log
