#!/usr/bin/env python3
"""Build v4 cross-model spreadsheet.

Rows: (model, config, regime)
Cols: workload params, tokens/s, req/s, MFU, MBU, HBM peak, decode step time, VRAM peak
"""
import csv, json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
V4 = REPO / "results/2026-07-07_v4_decode_sweep"

REGIMES_ORDER = [
    'R_decode_c1_out2k',
    'R_conc_ref',
    'R_decode_c32_out1k',
    'R_decode_c64_out512',
    'R_decode_c128_out256',
]

REGIME_INFO = {
    'R_decode_c1_out2k':   {'prompt_words': 100,  'out_tok': 2048, 'concurrency': 1},
    'R_conc_ref':          {'prompt_words': 200,  'out_tok': 256,  'concurrency': 32},
    'R_decode_c32_out1k':  {'prompt_words': 200,  'out_tok': 1024, 'concurrency': 32},
    'R_decode_c64_out512': {'prompt_words': 200,  'out_tok': 512,  'concurrency': 64},
    'R_decode_c128_out256':{'prompt_words': 200,  'out_tok': 256,  'concurrency': 128},
}

MODELS = ['qwen3-30b-a3b-bf16', 'qwen3-30b-a3b-fp8', 'qwen3-0.6b']
CONFIGS = ['cookbook_baseline', 'v3_best_chunk8k', 'big_batch_cap128']

CONFIG_FLAGS_DESC = {
    'cookbook_baseline':   'cap=32 chunk=-1 lpm mem=0.85',
    'v3_best_chunk8k':     'cap=32 chunk=8192 fcfs mem=0.75',
    'big_batch_cap128':    'cap=128 chunk=2048 fcfs mem=0.90',
}

COLS = ['model', 'config', 'config_flags', 'regime',
        'prompt_words', 'out_tok', 'concurrency',
        'req_per_s', 'tokens_per_s',
        'MFU_simple_pct', 'MFU_amortized_pct', 'MBU_pct',
        'HBM_BW_peak_pct', 'VRAM_peak_GB',
        'decode_step_ms_at_this_batch', 'gen_throughput_at_this_batch',
        'reliable']

out_csv = REPO / 'results/consolidated_v4_decode_sweep.csv'
with out_csv.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()

    for m in MODELS:
        for c in CONFIGS:
            sum_p = V4 / m / c / 'summary.json'
            hw_p = V4 / m / c / 'hw_stats.json'
            if not sum_p.exists() or not hw_p.exists():
                continue
            s = json.load(open(sum_p))
            hw = json.load(open(hw_p))
            
            bw_peak = hw.get('gpu_stats', {}).get('memory_util_pct', {}).get('max', 0)
            vram_peak_gb = hw.get('gpu_stats', {}).get('memory_used_mib', {}).get('max', 0) / 1024
            timings = hw.get('sglang_timings', {})

            for reg_id in REGIMES_ORDER:
                r = s.get('regimes', {}).get(reg_id, {})
                if not r:
                    continue
                info = REGIME_INFO[reg_id]
                B = info['concurrency']
                # Find decode step time at THIS batch size
                bkey = f"batch_{B}"
                step_info = timings.get(bkey, {})
                step_ms = step_info.get('median_step_ms', '')
                gen_thr = step_info.get('median_gen_throughput_tps', '')
                mfu = r.get('mfu', {})
                w.writerow({
                    'model': m,
                    'config': c,
                    'config_flags': CONFIG_FLAGS_DESC.get(c, ''),
                    'regime': reg_id,
                    'prompt_words': info['prompt_words'],
                    'out_tok': info['out_tok'],
                    'concurrency': B,
                    'req_per_s': round(r['req_per_s']['mean'], 4),
                    'tokens_per_s': round(r['tokens_per_s']['mean'], 1),
                    'MFU_simple_pct': mfu.get('mfu_pct_simple', ''),
                    'MFU_amortized_pct': mfu.get('mfu_pct_amortized', ''),
                    'MBU_pct': mfu.get('mbu_pct', ''),
                    'HBM_BW_peak_pct': bw_peak,
                    'VRAM_peak_GB': round(vram_peak_gb, 1),
                    'decode_step_ms_at_this_batch': step_ms,
                    'gen_throughput_at_this_batch': gen_thr,
                    'reliable': r.get('reliable', ''),
                })
print(f"Wrote {out_csv}")
