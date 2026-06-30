ACTIVE SUBSPACES (sweeping):

  [Memory / KV cache]
    --mem-fraction-static : 0.75 / 0.85 / 0.90
        Larger = more KV cache, more parallel requests; too large risks OOM.

  [Batching / scheduling]
    --max-running-requests : 8 / 16 / 32 / 64
        Scheduler concurrency cap. Higher → more in-flight requests for
        concurrent-decode regimes; lower → less KV pressure.
    --chunked-prefill-size : -1 / 2048 / 8192
        -1 = no chunking. Affects long-prefill regimes only.
    --schedule-policy : lpm / fcfs
        lpm = longest-prefix-match (cache-friendly), fcfs = first-come.

  [Attention backend]
    --attention-backend : fa3 / flashinfer / triton
        fa3 = FlashAttention v3 (Hopper-optimized default).
        flashinfer = flashinfer attention kernel.
        triton = pure-Triton (slowest but most portable).

  [CUDA graph]
    --disable-cuda-graph : true / false
        false (default) = capture. true = eager mode (slower for decode).

  [MoE]
    --moe-runner-backend : triton / flashinfer_cutlass
        Which MoE GEMM kernel to use. Note: LFM2.5 only has 32 experts top-4
        (vs Qwen3 128/8), so the GEMM shapes are quite different.

INACTIVE SUBSPACES (held fixed):
  - tp_size / dp_size / ep_size / pp_size : all = 1 (single GPU)
  - speculative_algorithm : None (no spec decode in this study)
  - disaggregation_mode : null (single-server mode)
  - quantization : None (bf16; no fp8/awq versions of LFM2.5-8B-A1B available)
  - kv_cache_dtype : auto (we don't risk fp8 KV; needs separate validation)
  - lora_* / hicache_* / multimodal_* : not applicable

OUT-OF-SCOPE FOR v2 (would be v3 expansion):
  - cuda_graph_max_bs / cuda_graph_bs (explicit lists; integer-valued)
  - schedule_conservativeness (float; effect typically subtle)
  - max_prefill_tokens (interacts with chunked_prefill_size)
  - radix_eviction_policy (lru/lfu/etc; only matters under heavy contention)