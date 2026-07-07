#!/usr/bin/env python3
"""Add-on to v4 sweep: run only LFM2.5-8B-A1B (same 3 configs × 5 regimes)."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from run_v4_decode_sweep import (
    CONFIGS, run_one, OUT_ROOT, PORT_BASE, GPU_ID
)
import time

LFM_MODEL = {
    "name": "lfm2.5-8b-a1b",
    "server_config": "configs/lfm2.5_8b_a1b_v4.yaml",
    "mfu_model": "configs/models/lfm2.5-8b-a1b.yaml",
}

def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    port_counter = 100  # avoid conflict with earlier v4 ports
    for i, cfg in enumerate(CONFIGS):
        port = PORT_BASE + port_counter + i
        print(f"\n>>> [{i+1}/{len(CONFIGS)}] {LFM_MODEL['name']} × {cfg['name']} at "
              f"{time.time()-t_start:.0f}s <<<", flush=True)
        try:
            run_one(LFM_MODEL, cfg, port)
        except Exception as e:
            print(f"[ERROR] {cfg['name']}: {e}")
            import traceback; traceback.print_exc()
    print(f"\n=== LFM2.5 sweep done in {(time.time()-t_start)/60:.1f} min ===",
          flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
