# 4-way MoE Backend Benchmark Results

**Setup**: Qwen3-30B-A3B-Instruct-2507 / H200 / bf16 / TP=1 / max-model-len=32768
Same bench harness (`/tmp/run_bench_4way.py`), same prompts (seed=2026), same `max_new=256, temperature=0, ignore_eos=True`.

## Absolute throughput

| Regime | sglang_triton | sglang_cutlass | vllm_triton | vllm_cutlass |
|---|---|---|---|---|
| R_short | 1.92 req/s (123 tok/s) | 0.71 req/s (45 tok/s) | 3.05 req/s (195 tok/s) | 3.05 req/s (195 tok/s) |
| R_medium | 3.02 req/s (772 tok/s) | 1.26 req/s (324 tok/s) | 4.32 req/s (1105 tok/s) | 4.37 req/s (1120 tok/s) |
| R_long | 2.95 req/s (754 tok/s) | 1.20 req/s (306 tok/s) | 3.53 req/s (903 tok/s) | 3.66 req/s (937 tok/s) |

## Relative speed

| Regime | sglang Triton→CUTLASS | vLLM Triton→CUTLASS | sglang→vLLM (Triton) | sglang→vLLM (CUTLASS) |
|---|---|---|---|---|
| R_short | 0.37× | 1.00× | 1.59× | 4.29× |
| R_medium | 0.42× | 1.01× | 1.43× | 3.46× |
| R_long | 0.41× | 1.04× | 1.20× | 3.06× |

## Raw data

```json
{
  "sglang_triton": {
    "R_short": {
      "num_prompts": 8,
      "prompt_words": 200,
      "max_new": 64,
      "concurrency": 1,
      "wall_s": 4.165067195892334,
      "req_per_s": 1.9207373191697237,
      "tokens_per_s": 122.92718842686232,
      "total_out_tokens": 512
    },
    "R_medium": {
      "num_prompts": 16,
      "prompt_words": 800,
      "max_new": 256,
      "concurrency": 8,
      "wall_s": 5.303118467330933,
      "req_per_s": 3.017092697167828,
      "tokens_per_s": 772.3757304749639,
      "total_out_tokens": 4096
    },
    "R_long": {
      "num_prompts": 8,
      "prompt_words": 2000,
      "max_new": 256,
      "concurrency": 16,
      "wall_s": 2.7164483070373535,
      "req_per_s": 2.9450219903963712,
      "tokens_per_s": 753.925629541471,
      "total_out_tokens": 2048
    }
  },
  "sglang_cutlass": {
    "R_short": {
      "num_prompts": 8,
      "prompt_words": 200,
      "max_new": 64,
      "concurrency": 1,
      "wall_s": 11.270219087600708,
      "req_per_s": 0.7098353579303046,
      "tokens_per_s": 45.429462907539495,
      "total_out_tokens": 512
    },
    "R_medium": {
      "num_prompts": 16,
      "prompt_words": 800,
      "max_new": 256,
      "concurrency": 8,
      "wall_s": 12.658296585083008,
      "req_per_s": 1.263993136237223,
      "tokens_per_s": 323.5822428767291,
      "total_out_tokens": 4096
    },
    "R_long": {
      "num_prompts": 8,
      "prompt_words": 2000,
      "max_new": 256,
      "concurrency": 16,
      "wall_s": 6.693678140640259,
      "req_per_s": 1.1951575549216338,
      "tokens_per_s": 305.96033405993825,
      "total_out_tokens": 2048
    }
  },
  "vllm_triton": {
    "R_short": {
      "num_prompts": 8,
      "prompt_words": 200,
      "max_new": 64,
      "concurrency": 1,
      "wall_s": 2.6234402656555176,
      "req_per_s": 3.049430972273746,
      "tokens_per_s": 195.16358222551975,
      "total_out_tokens": 512
    },
    "R_medium": {
      "num_prompts": 16,
      "prompt_words": 800,
      "max_new": 256,
      "concurrency": 8,
      "wall_s": 3.7063517570495605,
      "req_per_s": 4.316913517333496,
      "tokens_per_s": 1105.1298604373749,
      "total_out_tokens": 4096
    },
    "R_long": {
      "num_prompts": 8,
      "prompt_words": 2000,
      "max_new": 256,
      "concurrency": 16,
      "wall_s": 2.2668545246124268,
      "req_per_s": 3.529119276574571,
      "tokens_per_s": 903.4545348030902,
      "total_out_tokens": 2048
    }
  },
  "vllm_cutlass": {
    "R_short": {
      "num_prompts": 8,
      "prompt_words": 200,
      "max_new": 64,
      "concurrency": 1,
      "wall_s": 2.625805616378784,
      "req_per_s": 3.0466840157927226,
      "tokens_per_s": 194.98777701073425,
      "total_out_tokens": 512
    },
    "R_medium": {
      "num_prompts": 16,
      "prompt_words": 800,
      "max_new": 256,
      "concurrency": 8,
      "wall_s": 3.6577374935150146,
      "req_per_s": 4.374288758656737,
      "tokens_per_s": 1119.8179222161248,
      "total_out_tokens": 4096
    },
    "R_long": {
      "num_prompts": 8,
      "prompt_words": 2000,
      "max_new": 256,
      "concurrency": 16,
      "wall_s": 2.186481475830078,
      "req_per_s": 3.6588464564799805,
      "tokens_per_s": 936.664692858875,
      "total_out_tokens": 2048
    }
  }
}
```
