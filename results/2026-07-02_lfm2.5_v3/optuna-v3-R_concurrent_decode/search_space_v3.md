V3 SEARCH SPACE (2026-07-02, fixed TPE failure via warm-start):

  ACTIVE (7 knobs, all sweeping):
    --mem-fraction-static ∈ {0.75, 0.85, 0.90}
    --max-running-requests ∈ {8, 16, 32, 64}
    --chunked-prefill-size ∈ {-1, 2048, 8192}
    --schedule-policy ∈ {lpm, fcfs}
    --attention-backend ∈ {fa3}     [others rejected by model/env]
    --disable-cuda-graph ∈ {True, False}
    --moe-runner-backend ∈ {triton, flashinfer_cutlass, auto}  [+auto vs v2]

  Total combos: 3 × 4 × 3 × 2 × 1 × 2 × 3 = 432

  WARM-START (enqueued before TPE):
    Trial 0: moe=auto,   good batching prior (cookbook-equivalent)
    Trial 1: moe=triton, good batching prior [key: v2 missed this!]
    Trial 2: moe=flashinfer_cutlass, good batching prior [v2 winner]
    Trial 3: moe=auto,   fcfs schedule policy (control)

  Good batching prior: cap=32, chunk=-1, sched=lpm, mem=0.9, cg-on

  TPE runs from trial 4 onwards with the full space.

INACTIVE / OUT-OF-SCOPE (unchanged from v2):
  - Parallelism (tp/dp/ep/pp), Speculative, PD disagg, Quantization,
    KV dtype, HiCache, LoRA, Multimodal.