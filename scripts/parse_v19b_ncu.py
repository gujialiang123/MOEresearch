#!/usr/bin/env python3
"""Parse v19b NCU CSVs into a per-regime, per-kernel-family summary of the 11
metrics, and compute decode roofline signals (DRAM %, occupancy limiter, bytes).
"""
import csv, glob, os, json
from collections import defaultdict, OrderedDict

ROOT = "/home/t-jialianggu/work/EndtoEnd-auto-optimization/results/2026-07-15_v19b_ncu_decode/qwen3-30b-a3b-bf16"
METRICS = [
    "gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "l1tex__t_sector_hit_rate.pct", "lts__t_sector_hit_rate.pct",
    "launch__occupancy_limit_registers", "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
]

def fam(n):
    nl = n.lower()
    if "fused_moe" in nl: return "fused_moe (expert GEMM)"
    if "flashattnfwdsm90" in nl or ("flashattn" in nl and "combine" not in nl): return "FlashAttention"
    if "flashattnfwdcombine" in nl: return "FlashAttn combine"
    if "rmsnorm" in nl: return "RMSNorm"
    if "rotary" in nl or "batchqkapply" in nl: return "RoPE"
    if "nvjet" in nl: return "dense GEMM (nvjet)"
    if "topkgating" in nl: return "MoE router topk"
    if "act_and_mul" in nl: return "SiLU act_and_mul"
    if "prepare_varlen" in nl: return "FlashAttn setup"
    return n.split("(")[0][:28]

def to_bytes(v, unit):
    try: x = float(v.replace(",", ""))
    except: return 0.0
    u = (unit or "").lower()
    return x * {"byte":1,"kbyte":1e3,"mbyte":1e6,"gbyte":1e9}.get(u, 1)

def parse(path):
    # id -> {name, metric->(val,unit)}
    ids = OrderedDict()
    for r in csv.DictReader(open(path)):
        i = r["ID"]
        ids.setdefault(i, {"name": r["Kernel Name"], "m": {}})
        ids[i]["m"][r["Metric Name"]] = (r["Metric Value"], r["Metric Unit"])
    return ids

def agg(path):
    ids = parse(path)
    fam_dur = defaultdict(float); fam_rd = defaultdict(float); fam_wr = defaultdict(float)
    fam_n = defaultdict(int)
    fam_dram = defaultdict(list); fam_sm = defaultdict(list); fam_warp = defaultdict(list)
    fam_l1 = defaultdict(list); fam_l2 = defaultdict(list)
    fam_lim = defaultdict(lambda: defaultdict(int))
    for d in ids.values():
        f = fam(d["name"]); m = d["m"]
        def g(k):
            return m.get(k, ("", ""))
        dur = g("gpu__time_duration.sum")
        try: durv = float(dur[0].replace(",","")); 
        except: durv = 0.0
        # dur unit usually ns/us; normalize to us
        du = (dur[1] or "").lower()
        durv_us = durv * {"ns":1e-3,"us":1,"msecond":1e3,"ms":1e3,"second":1e6}.get(du,1e-3 if du=="ns" else 1)
        fam_dur[f] += durv_us; fam_n[f] += 1
        fam_rd[f] += to_bytes(*g("dram__bytes_read.sum"))
        fam_wr[f] += to_bytes(*g("dram__bytes_write.sum"))
        def fv(k):
            try: return float(g(k)[0].replace(",",""))
            except: return None
        for lst,k in [(fam_dram,"dram__throughput.avg.pct_of_peak_sustained_elapsed"),
                      (fam_sm,"sm__throughput.avg.pct_of_peak_sustained_elapsed"),
                      (fam_warp,"sm__warps_active.avg.pct_of_peak_sustained_active"),
                      (fam_l1,"l1tex__t_sector_hit_rate.pct"),
                      (fam_l2,"lts__t_sector_hit_rate.pct")]:
            val = fv(k)
            if val is not None: lst[f].append((val, durv_us))
        for k,short in [("launch__occupancy_limit_registers","reg"),
                        ("launch__occupancy_limit_shared_mem","smem"),
                        ("launch__occupancy_limit_warps","warp")]:
            v = g(k)[0]
            if v not in ("","n/a"): fam_lim[f][short]=v
    def wavg(lst):
        num = sum(v*w for v,w in lst); den = sum(w for _,w in lst)
        return round(num/den,1) if den else None
    out = []
    for f in sorted(fam_dur, key=lambda x:-fam_dur[x]):
        out.append(OrderedDict([
            ("kernel", f), ("n", fam_n[f]), ("dur_us", round(fam_dur[f],1)),
            ("dram_rd_MB", round(fam_rd[f]/1e6,2)), ("dram_wr_MB", round(fam_wr[f]/1e6,2)),
            ("dram_pct", wavg(fam_dram[f])), ("sm_pct", wavg(fam_sm[f])),
            ("warps_active_pct", wavg(fam_warp[f])),
            ("l1_hit_pct", wavg(fam_l1[f])), ("l2_hit_pct", wavg(fam_l2[f])),
            ("occ_limit", dict(fam_lim[f])),
        ]))
    return out

summary = {}
for D in sorted(glob.glob(f"{ROOT}/agent_*")):
    reg = os.path.basename(D)
    csvp = f"{D}/ncu_raw.csv"
    if not os.path.exists(csvp): continue
    summary[reg] = agg(csvp)

json.dump(summary, open(f"{ROOT}/ncu_summary.json","w"), indent=2)

# print decode regimes
for reg in ["agent_decode_b32","agent_decode_b64","agent_decode_b128","agent_prefill_b1"]:
    if reg not in summary: continue
    print(f"\n===== {reg} =====")
    print(f"{'kernel':26s}{'n':>3}{'dur_us':>9}{'rd_MB':>8}{'dram%':>7}{'sm%':>6}{'warp%':>7}{'L2%':>6}  occ_limit")
    tot=sum(k['dur_us'] for k in summary[reg])
    for k in summary[reg]:
        lim=",".join(f"{a}={b}" for a,b in k['occ_limit'].items())
        print(f"{k['kernel']:26s}{k['n']:>3}{k['dur_us']:>9.1f}{k['dram_rd_MB']:>8.1f}"
              f"{str(k['dram_pct']):>7}{str(k['sm_pct']):>6}{str(k['warps_active_pct']):>7}{str(k['l2_hit_pct']):>6}  {lim}")
    print(f"  total dur: {tot:.1f} us")
print(f"\nwrote {ROOT}/ncu_summary.json")
