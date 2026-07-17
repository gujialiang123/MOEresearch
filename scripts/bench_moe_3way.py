"""
MoE GEMM 3-way performance benchmark on H200 bf16. v2 — fixed API.
"""
import os, sys, json, time, torch, triton

os.environ["TRITON_CACHE_DIR"] = "/tmp/bench_triton_cache_v2_" + str(int(time.time()))
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe
from sglang.srt.layers.moe.fused_moe_triton import override_config
from sglang.srt.layers.moe.topk import TopK, TopKConfig, TopKOutputFormat
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)

set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

# Init distributed environment so TopK + FusedMoE work standalone
init_distributed_environment(
    world_size=1, rank=0,
    distributed_init_method="tcp://127.0.0.1:23457",
    local_rank=0, backend="nccl",
)
initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)

E, TOPK, HIDDEN_SIZE, N = 128, 8, 2048, 768
DTYPE = torch.bfloat16
DEVICE = "cuda"

OLD_CONFIG_PATH = "/home/t-jialianggu/work/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_2_0/E=128,N=768,device_name=NVIDIA_H200.json"
NEW_CONFIG_PATH = "/home/t-jialianggu/work/sglang/benchmark/kernels/fused_moe_triton/E=128,N=768,device_name=NVIDIA_H200.json"

with open(OLD_CONFIG_PATH) as f:
    OLD_CONFIGS = {int(k): v for k, v in json.load(f).items()}
with open(NEW_CONFIG_PATH) as f:
    NEW_CONFIGS = {int(k): v for k, v in json.load(f).items()}

print(f"Old config: {len(OLD_CONFIGS)} batch sizes, New config: {len(NEW_CONFIGS)} batch sizes")


def make_inputs(num_tokens, seed=42):
    torch.manual_seed(seed)
    x = torch.randn(num_tokens, HIDDEN_SIZE, dtype=DTYPE, device=DEVICE)
    w1 = torch.randn(E, 2 * N, HIDDEN_SIZE, dtype=DTYPE, device=DEVICE) * 0.02
    w2 = torch.randn(E, HIDDEN_SIZE, N, dtype=DTYPE, device=DEVICE) * 0.02
    gating = torch.randn(num_tokens, E, dtype=DTYPE, device=DEVICE)
    return x, w1, w2, gating


def compute_topk(gating, hidden_states):
    topk_op = TopK(
        top_k=TOPK,
        renormalize=False,
        use_grouped_topk=False,
        output_format=TopKOutputFormat.STANDARD,
    )
    return topk_op.forward_cuda(hidden_states=hidden_states, router_logits=gating)


def bench_triton(num_tokens, config_dict, num_iters=100):
    x, w1, w2, gating = make_inputs(num_tokens)
    topk_output = compute_topk(gating, x)
    runner_cfg = MoeRunnerConfig(inplace=False)
    
    config = config_dict.get(num_tokens) or config_dict[min(config_dict, key=lambda k: abs(k - num_tokens))]
    
    # Warmup
    for _ in range(10):
        with override_config(config):
            _ = fused_moe(hidden_states=x, w1=w1, w2=w2, topk_output=topk_output,
                          moe_runner_config=runner_cfg)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        with override_config(config):
            out = fused_moe(hidden_states=x, w1=w1, w2=w2, topk_output=topk_output,
                            moe_runner_config=runner_cfg)
    end.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(end) * 1000 / num_iters
    return us, out


def naive_cuda_moe(x, w1, w2, gating, topk):
    """Naive baseline: Python loop + cuBLAS GEMM per expert (transformers-style)."""
    num_tokens = x.shape[0]
    num_experts = w1.shape[0]
    routing_weights = torch.softmax(gating, dim=-1, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(routing_weights, topk, dim=-1)
    topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True)).to(x.dtype)
    final = torch.zeros(num_tokens, x.shape[1], dtype=x.dtype, device=x.device)
    for e in range(num_experts):
        mask = (topk_ids == e)
        if not mask.any():
            continue
        token_idx, k_idx = mask.nonzero(as_tuple=True)
        if len(token_idx) == 0:
            continue
        x_e = x[token_idx]
        gate_up = torch.matmul(x_e, w1[e].t())  # cuBLAS GEMM
        gate, up = gate_up.chunk(2, dim=-1)
        intermediate = torch.nn.functional.silu(gate) * up
        expert_out = torch.matmul(intermediate, w2[e].t())  # cuBLAS GEMM
        w = topk_weights[token_idx, k_idx].unsqueeze(-1)
        final.index_add_(0, token_idx, expert_out * w)
    return final


def bench_naive(num_tokens, num_iters=10):
    x, w1, w2, gating = make_inputs(num_tokens)
    for _ in range(3):
        _ = naive_cuda_moe(x, w1, w2, gating, TOPK)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        out = naive_cuda_moe(x, w1, w2, gating, TOPK)
    end.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(end) * 1000 / num_iters
    return us, out


def compute_flops(num_tokens):
    flops_per_active_token = 2 * (2 * N * HIDDEN_SIZE + HIDDEN_SIZE * N)
    return num_tokens * TOPK * flops_per_active_token


def main():
    BATCH_SIZES = [1, 8, 32, 128, 512, 2048]
    
    print(f"\n{'Batch':>6} {'OLD µs':>10} {'NEW µs':>10} {'Naive µs':>11} | "
          f"{'NEW/OLD':>8} {'Naive/NEW':>10} | "
          f"{'OLD TFLOPS':>11} {'NEW TFLOPS':>11} {'%peak':>7}")
    print("-" * 110)
    
    H200_PEAK = 989.0
    results = []
    for bs in BATCH_SIZES:
        try:
            t_old, _ = bench_triton(bs, OLD_CONFIGS)
        except Exception as e:
            print(f"  bs={bs}: OLD failed: {str(e)[:80]}")
            t_old = None
        if bs in NEW_CONFIGS:
            try:
                t_new, _ = bench_triton(bs, NEW_CONFIGS)
            except Exception as e:
                print(f"  bs={bs}: NEW failed: {str(e)[:80]}")
                t_new = None
        else:
            t_new = None  # we only autotuned bs=32
        try:
            t_naive, _ = bench_naive(bs)
        except Exception as e:
            print(f"  bs={bs}: NAIVE failed: {str(e)[:80]}")
            t_naive = None
        
        gflops = compute_flops(bs) / 1e9
        tflops_old = (gflops / (t_old / 1e6)) / 1e3 if t_old else 0
        tflops_new = (gflops / (t_new / 1e6)) / 1e3 if t_new else 0
        tflops_naive = (gflops / (t_naive / 1e6)) / 1e3 if t_naive else 0
        speedup_new_old = (t_old / t_new) if (t_new and t_old) else None
        speedup_new_naive = (t_naive / t_new) if (t_new and t_naive) else None
        pct = tflops_new / H200_PEAK * 100 if t_new else 0
        
        s_old = f"{t_old:>8.1f}" if t_old else "    N/A"
        s_new = f"{t_new:>8.1f}" if t_new else "    N/A"
        s_naive = f"{t_naive:>9.1f}" if t_naive else "      N/A"
        snovo = f"{speedup_new_old:>6.2f}x" if speedup_new_old else "    N/A"
        snvna = f"{speedup_new_naive:>8.2f}x" if speedup_new_naive else "      N/A"
        
        print(f"  {bs:>4} {s_old} {s_new} {s_naive} | {snovo} {snvna} | "
              f"{tflops_old:>10.1f} {tflops_new:>10.1f} {pct:>6.1f}%")
        
        results.append({
            "batch_size": bs, "triton_old_us": t_old, "triton_new_us": t_new, "naive_us": t_naive,
            "new_vs_old_speedup": speedup_new_old, "new_vs_naive_speedup": speedup_new_naive,
            "tflops_old": tflops_old, "tflops_new": tflops_new, "tflops_naive": tflops_naive,
            "pct_of_h200_peak_new": pct,
        })
    
    print(f"\nH200 bf16 theoretical peak: {H200_PEAK} TFLOPS")
    
    out_path = "/home/t-jialianggu/work/MOEresearch/results/cuda_vs_triton_bench.json"
    with open(out_path, "w") as f:
        json.dump({"model": "Qwen3-30B-A3B", "gpu": "NVIDIA H200",
                   "h200_bf16_peak_tflops": H200_PEAK,
                   "moe_dims": {"E": E, "topk": TOPK, "hidden": HIDDEN_SIZE, "N": N, "dtype": "bf16"},
                   "results": results}, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
