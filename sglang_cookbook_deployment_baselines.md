# SGLang Cookbook Deployment Baselines

Generated: 2026-06-29

Scope: commands are normalized from the official `docs_new/src/snippets/configs/*/*.jsx` Deployment config `cells` and rendered with the same Python-mode convention as `_deployment.jsx`: `sglang serve` plus the cell flags. This is a baseline document, not a claim that every possible SGLang model/GPU/workload combination is covered.

Defaults used while rendering: `HOST_IP=0.0.0.0`, `PORT=30000`. Multi-node commands intentionally leave `{{NODE_RANK}}` and `{{NODE0_IP}}` as placeholders.

Verified meaning: `true` matches `verified: true` in the upstream config; `false` matches `verified: false`; `null` means the upstream cell omitted the field and should be treated as inferred/unverified.

## Index

| Model | Command cells | Source config |
|---|---:|---|
| GLM-5.2 | 29 | `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` |
| MiniMax-M3 | 9 | `docs_new/src/snippets/configs/MiniMaxAI/minimax-m3.jsx` |
| LFM2.5 | 24 | `docs_new/src/snippets/configs/LiquidAI/lfm2.5.jsx` |
| Unlimited-OCR | 1 | `docs_new/src/snippets/configs/baidu/unlimited-ocr.jsx` |
| Laguna-M.1 | 14 | `docs_new/src/snippets/configs/poolside/laguna-m1.jsx` |
| DeepSeek-V4 | 0 | `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` |

Total fixed command cells: **77**.

## Commands

### GLM-5.2

#### 1. b200 / default / bf16 / balanced / multi-2

- verified: `false`
- model_path: `zai-org/GLM-5.2`
- docker_image: `lmsysorg/sglang:latest`
- notes: Inferred multi-node BF16 cell in the official config.

```bash
# Multi-node (2 nodes). Run this on every node.
# Fill {{NODE_RANK}} as 0 on the head node, then 1..N-1 on others.
# Fill {{NODE0_IP}} with the head-node IP reachable from all nodes.
sglang serve \
  --model-path zai-org/GLM-5.2 \
  --tp 16 \
  --nnodes 2 \
  --node-rank {{NODE_RANK}} \
  --dist-init-addr {{NODE0_IP}}:20000 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 32768 \
  --max-running-requests 80 \
  --host 0.0.0.0 \
  --port 30000
```

#### 2. b200 / default / bf16 / high-throughput / multi-2

- verified: `false`
- model_path: `zai-org/GLM-5.2`
- docker_image: `lmsysorg/sglang:latest`
- notes: Inferred multi-node BF16 cell in the official config.

```bash
# Multi-node (2 nodes). Run this on every node.
# Fill {{NODE_RANK}} as 0 on the head node, then 1..N-1 on others.
# Fill {{NODE0_IP}} with the head-node IP reachable from all nodes.
sglang serve \
  --model-path zai-org/GLM-5.2 \
  --tp 16 \
  --nnodes 2 \
  --node-rank {{NODE_RANK}} \
  --dist-init-addr {{NODE0_IP}}:20000 \
  --mem-fraction-static 0.85 \
  --max-running-requests 256 \
  --host 0.0.0.0 \
  --port 30000
```

#### 3. b200 / default / bf16 / low-latency / multi-2

- verified: `false`
- model_path: `zai-org/GLM-5.2`
- docker_image: `lmsysorg/sglang:latest`
- notes: Inferred multi-node BF16 cell in the official config.

```bash
# Multi-node (2 nodes). Run this on every node.
# Fill {{NODE_RANK}} as 0 on the head node, then 1..N-1 on others.
# Fill {{NODE0_IP}} with the head-node IP reachable from all nodes.
sglang serve \
  --model-path zai-org/GLM-5.2 \
  --tp 16 \
  --nnodes 2 \
  --node-rank {{NODE_RANK}} \
  --dist-init-addr {{NODE0_IP}}:20000 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --mem-fraction-static 0.85 \
  --host 0.0.0.0 \
  --port 30000
```

#### 4. b200 / default / fp8 / balanced / single

- verified: `true`
- model_path: `zai-org/GLM-5.2-FP8`
- docker_image: `lmsysorg/sglang:latest`

```bash
sglang serve \
  --model-path zai-org/GLM-5.2-FP8 \
  --tp 8 \
  --dp 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 32768 \
  --max-running-requests 256 \
  --host 0.0.0.0 \
  --port 30000
```

#### 5. b200 / default / fp8 / high-throughput / single

- verified: `true`
- model_path: `zai-org/GLM-5.2-FP8`
- docker_image: `lmsysorg/sglang:latest`

```bash
sglang serve \
  --model-path zai-org/GLM-5.2-FP8 \
  --tp 8 \
  --dp 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.85 \
  --max-running-requests 256 \
  --host 0.0.0.0 \
  --port 30000
```

#### 6. b200 / default / fp8 / low-latency / single

- verified: `true`
- model_path: `zai-org/GLM-5.2-FP8`
- docker_image: `lmsysorg/sglang:latest`

```bash
sglang serve \
  --model-path zai-org/GLM-5.2-FP8 \
  --tp 8 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --mem-fraction-static 0.8 \
  --host 0.0.0.0 \
  --port 30000
```

#### 7. b300 / default / bf16 / balanced / single

- verified: `true`
- model_path: `zai-org/GLM-5.2`
- docker_image: `lmsysorg/sglang:latest`

```bash
sglang serve \
  --model-path zai-org/GLM-5.2 \
  --tp 8 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --mem-fraction-static 0.9 \
  --chunked-prefill-size 32768 \
  --max-running-requests 80 \
  --host 0.0.0.0 \
  --port 30000
```

#### 8. b300 / default / bf16 / high-throughput / single

- verified: `true`
- model_path: `zai-org/GLM-5.2`
- docker_image: `lmsysorg/sglang:latest`

```bash
sglang serve \
  --model-path zai-org/GLM-5.2 \
  --tp 8 \
  --mem-fraction-static 0.9 \
  --max-running-requests 256 \
  --host 0.0.0.0 \
  --port 30000
```

#### 9. b300 / default / bf16 / low-latency / single

- verified: `true`
- model_path: `zai-org/GLM-5.2`
- docker_image: `lmsysorg/sglang:latest`

```bash
sglang serve \
  --model-path zai-org/GLM-5.2 \
  --tp 8 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --mem-fraction-static 0.9 \
  --host 0.0.0.0 \
  --port 30000
```

#### 10. b300 / default / fp8 / balanced / single

- verified: `true`
- model_path: `zai-org/GLM-5.2-FP8`
- docker_image: `lmsysorg/sglang:latest`

```bash
sglang serve \
  --model-path zai-org/GLM-5.2-FP8 \
  --tp 8 \
  --dp 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 32768 \
  --max-running-requests 256 \
  --host 0.0.0.0 \
  --port 30000
```

#### 11. b300 / default / fp8 / high-throughput / single

- verified: `true`
- model_path: `zai-org/GLM-5.2-FP8`
- docker_image: `lmsysorg/sglang:latest`

```bash
sglang serve \
  --model-path zai-org/GLM-5.2-FP8 \
  --tp 8 \
  --dp 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.85 \
  --max-running-requests 256 \
  --host 0.0.0.0 \
  --port 30000
```

#### 12. b300 / default / fp8 / low-latency / single

- verified: `true`
- model_path: `zai-org/GLM-5.2-FP8`
- docker_image: `lmsysorg/sglang:latest`

```bash
sglang serve \
  --model-path zai-org/GLM-5.2-FP8 \
  --tp 8 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --mem-fraction-static 0.8 \
  --host 0.0.0.0 \
  --port 30000
```

#### 13. b300 / default / nvfp4 / balanced / single

- verified: `true`
- model_path: `nvidia/GLM-5.2-NVFP4`
- docker_image: `lmsysorg/sglang:dev-glm52-nvfp4`

```bash
sglang serve \
  --model-path nvidia/GLM-5.2-NVFP4 \
  --tp 4 \
  --quantization modelopt_fp4 \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.8 \
  --host 0.0.0.0 \
  --port 30000
```

#### 14. b300 / default / nvfp4 / low-latency / single

- verified: `true`
- model_path: `nvidia/GLM-5.2-NVFP4`
- docker_image: `lmsysorg/sglang:dev-glm52-nvfp4`

```bash
sglang serve \
  --model-path nvidia/GLM-5.2-NVFP4 \
  --tp 4 \
  --quantization modelopt_fp4 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.8 \
  --host 0.0.0.0 \
  --port 30000
```

#### 15. gb300 / default / bf16 / balanced / multi-2

- verified: `false`
- model_path: `zai-org/GLM-5.2`
- docker_image: `lmsysorg/sglang:latest`
- notes: Inferred multi-node BF16 cell in the official config.

```bash
# Multi-node (2 nodes). Run this on every node.
# Fill {{NODE_RANK}} as 0 on the head node, then 1..N-1 on others.
# Fill {{NODE0_IP}} with the head-node IP reachable from all nodes.
sglang serve \
  --model-path zai-org/GLM-5.2 \
  --tp 8 \
  --nnodes 2 \
  --node-rank {{NODE_RANK}} \
  --dist-init-addr {{NODE0_IP}}:20000 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 32768 \
  --max-running-requests 80 \
  --host 0.0.0.0 \
  --port 30000
```

#### 16. gb300 / default / bf16 / high-throughput / multi-2

- verified: `false`
- model_path: `zai-org/GLM-5.2`
- docker_image: `lmsysorg/sglang:latest`
- notes: Inferred multi-node BF16 cell in the official config.

```bash
# Multi-node (2 nodes). Run this on every node.
# Fill {{NODE_RANK}} as 0 on the head node, then 1..N-1 on others.
# Fill {{NODE0_IP}} with the head-node IP reachable from all nodes.
sglang serve \
  --model-path zai-org/GLM-5.2 \
  --tp 8 \
  --nnodes 2 \
  --node-rank {{NODE_RANK}} \
  --dist-init-addr {{NODE0_IP}}:20000 \
  --mem-fraction-static 0.85 \
  --max-running-requests 256 \
  --host 0.0.0.0 \
  --port 30000
```

#### 17. gb300 / default / bf16 / low-latency / multi-2

- verified: `false`
- model_path: `zai-org/GLM-5.2`
- docker_image: `lmsysorg/sglang:latest`
- notes: Inferred multi-node BF16 cell in the official config.

```bash
# Multi-node (2 nodes). Run this on every node.
# Fill {{NODE_RANK}} as 0 on the head node, then 1..N-1 on others.
# Fill {{NODE0_IP}} with the head-node IP reachable from all nodes.
sglang serve \
  --model-path zai-org/GLM-5.2 \
  --tp 8 \
  --nnodes 2 \
  --node-rank {{NODE_RANK}} \
  --dist-init-addr {{NODE0_IP}}:20000 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --mem-fraction-static 0.85 \
  --host 0.0.0.0 \
  --port 30000
```

#### 18. gb300 / default / fp8 / balanced / single

- verified: `true`
- model_path: `zai-org/GLM-5.2-FP8`
- docker_image: `lmsysorg/sglang:latest`

```bash
sglang serve \
  --model-path zai-org/GLM-5.2-FP8 \
  --tp 4 \
  --dp 4 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 32768 \
  --max-running-requests 256 \
  --host 0.0.0.0 \
  --port 30000
```

#### 19. gb300 / default / fp8 / high-throughput / single

- verified: `true`
- model_path: `zai-org/GLM-5.2-FP8`
- docker_image: `lmsysorg/sglang:latest`
- env: `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512`

```bash
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512 \
sglang serve \
  --model-path zai-org/GLM-5.2-FP8 \
  --tp 4 \
  --dp 4 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.85 \
  --host 0.0.0.0 \
  --port 30000
```

#### 20. gb300 / default / fp8 / low-latency / single

- verified: `true`
- model_path: `zai-org/GLM-5.2-FP8`
- docker_image: `lmsysorg/sglang:latest`

```bash
sglang serve \
  --model-path zai-org/GLM-5.2-FP8 \
  --tp 4 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --mem-fraction-static 0.85 \
  --host 0.0.0.0 \
  --port 30000
```

#### 21. gb300 / default / nvfp4 / balanced / single

- verified: `true`
- model_path: `nvidia/GLM-5.2-NVFP4`
- docker_image: `lmsysorg/sglang:dev-glm52-nvfp4`

```bash
sglang serve \
  --model-path nvidia/GLM-5.2-NVFP4 \
  --tp 4 \
  --quantization modelopt_fp4 \
  --dp 4 \
  --enable-dp-attention \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 2 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 3 \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.92 \
  --max-running-requests 256 \
  --host 0.0.0.0 \
  --port 30000
```

#### 22. gb300 / default / nvfp4 / high-throughput / single

- verified: `true`
- model_path: `nvidia/GLM-5.2-NVFP4`
- docker_image: `lmsysorg/sglang:dev-glm52-nvfp4`

```bash
sglang serve \
  --model-path nvidia/GLM-5.2-NVFP4 \
  --tp 4 \
  --quantization modelopt_fp4 \
  --dp 4 \
  --enable-dp-attention \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.92 \
  --max-running-requests 512 \
  --host 0.0.0.0 \
  --port 30000
```

#### 23. gb300 / default / nvfp4 / low-latency / single

- verified: `true`
- model_path: `nvidia/GLM-5.2-NVFP4`
- docker_image: `lmsysorg/sglang:dev-glm52-nvfp4`

```bash
sglang serve \
  --model-path nvidia/GLM-5.2-NVFP4 \
  --tp 4 \
  --quantization modelopt_fp4 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.85 \
  --host 0.0.0.0 \
  --port 30000
```

#### 24. h200 / default / bf16 / balanced / multi-2

- verified: `false`
- model_path: `zai-org/GLM-5.2`
- docker_image: `lmsysorg/sglang:latest`
- notes: Inferred multi-node BF16 cell in the official config.

```bash
# Multi-node (2 nodes). Run this on every node.
# Fill {{NODE_RANK}} as 0 on the head node, then 1..N-1 on others.
# Fill {{NODE0_IP}} with the head-node IP reachable from all nodes.
sglang serve \
  --model-path zai-org/GLM-5.2 \
  --tp 16 \
  --nnodes 2 \
  --node-rank {{NODE_RANK}} \
  --dist-init-addr {{NODE0_IP}}:20000 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 32768 \
  --max-running-requests 80 \
  --host 0.0.0.0 \
  --port 30000
```

#### 25. h200 / default / bf16 / high-throughput / multi-2

- verified: `false`
- model_path: `zai-org/GLM-5.2`
- docker_image: `lmsysorg/sglang:latest`
- notes: Inferred multi-node BF16 cell in the official config.

```bash
# Multi-node (2 nodes). Run this on every node.
# Fill {{NODE_RANK}} as 0 on the head node, then 1..N-1 on others.
# Fill {{NODE0_IP}} with the head-node IP reachable from all nodes.
sglang serve \
  --model-path zai-org/GLM-5.2 \
  --tp 16 \
  --nnodes 2 \
  --node-rank {{NODE_RANK}} \
  --dist-init-addr {{NODE0_IP}}:20000 \
  --mem-fraction-static 0.85 \
  --max-running-requests 256 \
  --host 0.0.0.0 \
  --port 30000
```

#### 26. h200 / default / bf16 / low-latency / multi-2

- verified: `false`
- model_path: `zai-org/GLM-5.2`
- docker_image: `lmsysorg/sglang:latest`
- notes: Inferred multi-node BF16 cell in the official config.

```bash
# Multi-node (2 nodes). Run this on every node.
# Fill {{NODE_RANK}} as 0 on the head node, then 1..N-1 on others.
# Fill {{NODE0_IP}} with the head-node IP reachable from all nodes.
sglang serve \
  --model-path zai-org/GLM-5.2 \
  --tp 16 \
  --nnodes 2 \
  --node-rank {{NODE_RANK}} \
  --dist-init-addr {{NODE0_IP}}:20000 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --mem-fraction-static 0.85 \
  --host 0.0.0.0 \
  --port 30000
```

#### 27. h200 / default / fp8 / balanced / single

- verified: `true`
- model_path: `zai-org/GLM-5.2-FP8`
- docker_image: `lmsysorg/sglang:latest`

```bash
sglang serve \
  --model-path zai-org/GLM-5.2-FP8 \
  --tp 8 \
  --dp 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 32768 \
  --max-running-requests 256 \
  --host 0.0.0.0 \
  --port 30000
```

#### 28. h200 / default / fp8 / high-throughput / single

- verified: `true`
- model_path: `zai-org/GLM-5.2-FP8`
- docker_image: `lmsysorg/sglang:latest`

```bash
sglang serve \
  --model-path zai-org/GLM-5.2-FP8 \
  --tp 8 \
  --dp 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.85 \
  --max-running-requests 256 \
  --host 0.0.0.0 \
  --port 30000
```

#### 29. h200 / default / fp8 / low-latency / single

- verified: `true`
- model_path: `zai-org/GLM-5.2-FP8`
- docker_image: `lmsysorg/sglang:latest`

```bash
sglang serve \
  --model-path zai-org/GLM-5.2-FP8 \
  --tp 8 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --mem-fraction-static 0.8 \
  --host 0.0.0.0 \
  --port 30000
```

### MiniMax-M3

#### 1. b200 / default / mxfp8 / balanced / single

- verified: `true`
- model_path: `MiniMaxAI/MiniMax-M3-MXFP8`
- docker_image: `lmsysorg/sglang:dev-minimax-m3`

```bash
sglang serve \
  --trust-remote-code \
  --model-path MiniMaxAI/MiniMax-M3-MXFP8 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --tp 8 \
  --attention-backend fa4 \
  --moe-runner-backend deep_gemm \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.65 \
  --host 0.0.0.0 \
  --port 30000
```

#### 2. b300 / default / mxfp8 / balanced / single

- verified: `true`
- model_path: `MiniMaxAI/MiniMax-M3-MXFP8`
- docker_image: `lmsysorg/sglang:dev-cu13-minimax-m3`

```bash
sglang serve \
  --trust-remote-code \
  --model-path MiniMaxAI/MiniMax-M3-MXFP8 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --tp 4 \
  --attention-backend fa4 \
  --moe-runner-backend deep_gemm \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.75 \
  --host 0.0.0.0 \
  --port 30000
```

#### 3. gb200 / default / mxfp8 / balanced / single

- verified: `null`
- model_path: `MiniMaxAI/MiniMax-M3-MXFP8`
- docker_image: `lmsysorg/sglang:dev-cu13-minimax-m3`

```bash
sglang serve \
  --trust-remote-code \
  --model-path MiniMaxAI/MiniMax-M3-MXFP8 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --tp 4 \
  --attention-backend fa4 \
  --moe-runner-backend deep_gemm \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.75 \
  --host 0.0.0.0 \
  --port 30000
```

#### 4. gb300 / default / mxfp8 / balanced / single

- verified: `true`
- model_path: `MiniMaxAI/MiniMax-M3-MXFP8`
- docker_image: `lmsysorg/sglang:dev-cu13-minimax-m3`

```bash
sglang serve \
  --trust-remote-code \
  --model-path MiniMaxAI/MiniMax-M3-MXFP8 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --tp 4 \
  --attention-backend fa4 \
  --moe-runner-backend deep_gemm \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.75 \
  --host 0.0.0.0 \
  --port 30000
```

#### 5. h200 / default / bf16 / balanced / single

- verified: `true`
- model_path: `MiniMaxAI/MiniMax-M3`
- docker_image: `lmsysorg/sglang:dev-cu12-minimax-m3`

```bash
sglang serve \
  --trust-remote-code \
  --model-path MiniMaxAI/MiniMax-M3 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --tp 8 \
  --mem-fraction-static 0.75 \
  --host 0.0.0.0 \
  --port 30000
```

#### 6. mi300x / default / mxfp8 / balanced / single

- verified: `true`
- model_path: `MiniMaxAI/MiniMax-M3-MXFP8`
- docker_image: `aigmkt/minimax-m3-sglang-rocm700-mi30x`
- env: `SGLANG_USE_AITER=1`

```bash
SGLANG_USE_AITER=1 \
sglang serve \
  --trust-remote-code \
  --model-path MiniMaxAI/MiniMax-M3-MXFP8 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --tp 8 \
  --quantization mxfp8 \
  --dtype bfloat16 \
  --attention-backend aiter \
  --moe-runner-backend triton \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.80 \
  --watchdog-timeout 3600 \
  --skip-server-warmup \
  --host 0.0.0.0 \
  --port 30000
```

#### 7. mi325x / default / mxfp8 / balanced / single

- verified: `null`
- model_path: `MiniMaxAI/MiniMax-M3-MXFP8`
- docker_image: `aigmkt/minimax-m3-sglang-rocm700-mi30x`
- env: `SGLANG_USE_AITER=1`

```bash
SGLANG_USE_AITER=1 \
sglang serve \
  --trust-remote-code \
  --model-path MiniMaxAI/MiniMax-M3-MXFP8 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --tp 8 \
  --quantization mxfp8 \
  --dtype bfloat16 \
  --attention-backend aiter \
  --moe-runner-backend triton \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.80 \
  --watchdog-timeout 3600 \
  --skip-server-warmup \
  --host 0.0.0.0 \
  --port 30000
```

#### 8. mi350x / default / mxfp8 / balanced / single

- verified: `null`
- model_path: `MiniMaxAI/MiniMax-M3-MXFP8`
- docker_image: `aigmkt/minimax-m3-sglang-rocm720-mi35x`
- env: `SGLANG_USE_AITER=1`

```bash
SGLANG_USE_AITER=1 \
sglang serve \
  --trust-remote-code \
  --model-path MiniMaxAI/MiniMax-M3-MXFP8 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --tp 8 \
  --quantization mxfp8 \
  --dtype bfloat16 \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.80 \
  --host 0.0.0.0 \
  --port 30000
```

#### 9. mi355x / default / mxfp8 / balanced / single

- verified: `true`
- model_path: `MiniMaxAI/MiniMax-M3-MXFP8`
- docker_image: `aigmkt/minimax-m3-sglang-rocm720-mi35x`
- env: `SGLANG_USE_AITER=1`

```bash
SGLANG_USE_AITER=1 \
sglang serve \
  --trust-remote-code \
  --model-path MiniMaxAI/MiniMax-M3-MXFP8 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --tp 8 \
  --quantization mxfp8 \
  --dtype bfloat16 \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.80 \
  --host 0.0.0.0 \
  --port 30000
```

### LFM2.5

#### 1. b200 / 230m / bf16 / default / single

- verified: `true`
- model_path: `LiquidAI/LFM2.5-230M`
- docker_image: `lmsysorg/sglang:dev-cu13`

```bash
sglang serve \
  --trust-remote-code \
  --model-path LiquidAI/LFM2.5-230M \
  --tp 1 \
  --attention-backend trtllm_mha \
  --tool-call-parser lfm2 \
  --host 0.0.0.0 \
  --port 30000
```

#### 2. b200 / 350m / bf16 / default / single

- verified: `true`
- model_path: `LiquidAI/LFM2.5-350M`
- docker_image: `lmsysorg/sglang:dev-cu13`

```bash
sglang serve \
  --trust-remote-code \
  --model-path LiquidAI/LFM2.5-350M \
  --tp 1 \
  --attention-backend trtllm_mha \
  --tool-call-parser lfm2 \
  --host 0.0.0.0 \
  --port 30000
```

#### 3. b200 / 8b-a1b / bf16 / default / single

- verified: `true`
- model_path: `LiquidAI/LFM2.5-8B-A1B`
- docker_image: `lmsysorg/sglang:dev-cu13`

```bash
sglang serve \
  --trust-remote-code \
  --model-path LiquidAI/LFM2.5-8B-A1B \
  --tp 1 \
  --attention-backend flashinfer \
  --reasoning-parser qwen3 \
  --tool-call-parser lfm2 \
  --host 0.0.0.0 \
  --port 30000
```

#### 4. b200 / instruct / bf16 / default / single

- verified: `true`
- model_path: `LiquidAI/LFM2.5-1.2B-Instruct`
- docker_image: `lmsysorg/sglang:dev-cu13`

```bash
sglang serve \
  --trust-remote-code \
  --model-path LiquidAI/LFM2.5-1.2B-Instruct \
  --tp 1 \
  --attention-backend trtllm_mha \
  --tool-call-parser lfm2 \
  --host 0.0.0.0 \
  --port 30000
```

#### 5. b200 / jp / bf16 / default / single

- verified: `true`
- model_path: `LiquidAI/LFM2.5-1.2B-JP-202606`
- docker_image: `lmsysorg/sglang:dev-cu13`

```bash
sglang serve \
  --trust-remote-code \
  --model-path LiquidAI/LFM2.5-1.2B-JP-202606 \
  --tp 1 \
  --attention-backend trtllm_mha \
  --tool-call-parser lfm2 \
  --host 0.0.0.0 \
  --port 30000
```

#### 6. b200 / thinking / bf16 / default / single

- verified: `true`
- model_path: `LiquidAI/LFM2.5-1.2B-Thinking`
- docker_image: `lmsysorg/sglang:dev-cu13`

```bash
sglang serve \
  --trust-remote-code \
  --model-path LiquidAI/LFM2.5-1.2B-Thinking \
  --tp 1 \
  --attention-backend trtllm_mha \
  --reasoning-parser qwen3-thinking \
  --tool-call-parser lfm2 \
  --host 0.0.0.0 \
  --port 30000
```

#### 7. b200 / vl / bf16 / default / single

- verified: `true`
- model_path: `LiquidAI/LFM2.5-VL-1.6B`
- docker_image: `lmsysorg/sglang:dev-cu13`
- env: `SGLANG_USE_CUDA_IPC_TRANSPORT=1, SGLANG_USE_IPC_POOL_HANDLE_CACHE=1`

```bash
SGLANG_USE_CUDA_IPC_TRANSPORT=1 \
SGLANG_USE_IPC_POOL_HANDLE_CACHE=1 \
sglang serve \
  --trust-remote-code \
  --model-path LiquidAI/LFM2.5-VL-1.6B \
  --tp 1 \
  --attention-backend flashinfer \
  --mm-attention-backend fa4 \
  --tool-call-parser lfm2 \
  --host 0.0.0.0 \
  --port 30000
```

#### 8. b200 / vl-450m / bf16 / default / single

- verified: `true`
- model_path: `LiquidAI/LFM2.5-VL-450M`
- docker_image: `lmsysorg/sglang:dev-cu13`
- env: `SGLANG_USE_CUDA_IPC_TRANSPORT=1, SGLANG_USE_IPC_POOL_HANDLE_CACHE=1`

```bash
SGLANG_USE_CUDA_IPC_TRANSPORT=1 \
SGLANG_USE_IPC_POOL_HANDLE_CACHE=1 \
sglang serve \
  --trust-remote-code \
  --model-path LiquidAI/LFM2.5-VL-450M \
  --tp 1 \
  --attention-backend flashinfer \
  --mm-attention-backend fa4 \
  --tool-call-parser lfm2 \
  --mem-fraction-static 0.8 \
  --host 0.0.0.0 \
  --port 30000
```

#### 9. h100 / 230m / bf16 / default / single

- verified: `true`
- model_path: `LiquidAI/LFM2.5-230M`
- docker_image: `lmsysorg/sglang:dev-cu13`

```bash
sglang serve \
  --trust-remote-code \
  --model-path LiquidAI/LFM2.5-230M \
  --tp 1 \
  --tool-call-parser lfm2 \
  --host 0.0.0.0 \
  --port 30000
```

#### 10. h100 / 350m / bf16 / default / single

- verified: `true`
- model_path: `LiquidAI/LFM2.5-350M`
- docker_image: `lmsysorg/sglang:dev-cu13`

```bash
sglang serve \
  --trust-remote-code \
  --model-path LiquidAI/LFM2.5-350M \
  --tp 1 \
  --tool-call-parser lfm2 \
  --host 0.0.0.0 \
  --port 30000
```

#### 11. h100 / 8b-a1b / bf16 / default / single

- verified: `true`
- model_path: `LiquidAI/LFM2.5-8B-A1B`
- docker_image: `lmsysorg/sglang:dev-cu13`

```bash
sglang serve \
  --trust-remote-code \
  --model-path LiquidAI/LFM2.5-8B-A1B \
  --tp 1 \
  --reasoning-parser qwen3 \
  --tool-call-parser lfm2 \
  --host 0.0.0.0 \
  --port 30000
```

#### 12. h100 / instruct / bf16 / default / single

- verified: `true`
- model_path: `LiquidAI/LFM2.5-1.2B-Instruct`
- docker_image: `lmsysorg/sglang:dev-cu13`

```bash
sglang serve \
  --trust-remote-code \
  --model-path LiquidAI/LFM2.5-1.2B-Instruct \
  --tp 1 \
  --tool-call-parser lfm2 \
  --host 0.0.0.0 \
  --port 30000
```

#### 13. h100 / jp / bf16 / default / single

- verified: `true`
- model_path: `LiquidAI/LFM2.5-1.2B-JP-202606`
- docker_image: `lmsysorg/sglang:dev-cu13`

```bash
sglang serve \
  --trust-remote-code \
  --model-path LiquidAI/LFM2.5-1.2B-JP-202606 \
  --tp 1 \
  --tool-call-parser lfm2 \
  --host 0.0.0.0 \
  --port 30000
```

#### 14. h100 / thinking / bf16 / default / single

- verified: `true`
- model_path: `LiquidAI/LFM2.5-1.2B-Thinking`
- docker_image: `lmsysorg/sglang:dev-cu13`

```bash
sglang serve \
  --trust-remote-code \
  --model-path LiquidAI/LFM2.5-1.2B-Thinking \
  --tp 1 \
  --reasoning-parser qwen3-thinking \
  --tool-call-parser lfm2 \
  --host 0.0.0.0 \
  --port 30000
```

#### 15. h100 / vl / bf16 / default / single

- verified: `true`
- model_path: `LiquidAI/LFM2.5-VL-1.6B`
- docker_image: `lmsysorg/sglang:dev-cu13`
- env: `SGLANG_USE_CUDA_IPC_TRANSPORT=1, SGLANG_USE_IPC_POOL_HANDLE_CACHE=1`

```bash
SGLANG_USE_CUDA_IPC_TRANSPORT=1 \
SGLANG_USE_IPC_POOL_HANDLE_CACHE=1 \
sglang serve \
  --trust-remote-code \
  --model-path LiquidAI/LFM2.5-VL-1.6B \
  --tp 1 \
  --tool-call-parser lfm2 \
  --host 0.0.0.0 \
  --port 30000
```

#### 16. h100 / vl-450m / bf16 / default / single

- verified: `true`
- model_path: `LiquidAI/LFM2.5-VL-450M`
- docker_image: `lmsysorg/sglang:dev-cu13`
- env: `SGLANG_USE_CUDA_IPC_TRANSPORT=1, SGLANG_USE_IPC_POOL_HANDLE_CACHE=1`

```bash
SGLANG_USE_CUDA_IPC_TRANSPORT=1 \
SGLANG_USE_IPC_POOL_HANDLE_CACHE=1 \
sglang serve \
  --trust-remote-code \
  --model-path LiquidAI/LFM2.5-VL-450M \
  --tp 1 \
  --tool-call-parser lfm2 \
  --mem-fraction-static 0.8 \
  --host 0.0.0.0 \
  --port 30000
```

#### 17. h200 / 230m / bf16 / default / single

- verified: `true`
- model_path: `LiquidAI/LFM2.5-230M`
- docker_image: `lmsysorg/sglang:dev-cu13`

```bash
sglang serve \
  --trust-remote-code \
  --model-path LiquidAI/LFM2.5-230M \
  --tp 1 \
  --tool-call-parser lfm2 \
  --host 0.0.0.0 \
  --port 30000
```

#### 18. h200 / 350m / bf16 / default / single

- verified: `true`
- model_path: `LiquidAI/LFM2.5-350M`
- docker_image: `lmsysorg/sglang:dev-cu13`

```bash
sglang serve \
  --trust-remote-code \
  --model-path LiquidAI/LFM2.5-350M \
  --tp 1 \
  --tool-call-parser lfm2 \
  --host 0.0.0.0 \
  --port 30000
```

#### 19. h200 / 8b-a1b / bf16 / default / single

- verified: `true`
- model_path: `LiquidAI/LFM2.5-8B-A1B`
- docker_image: `lmsysorg/sglang:dev-cu13`

```bash
sglang serve \
  --trust-remote-code \
  --model-path LiquidAI/LFM2.5-8B-A1B \
  --tp 1 \
  --reasoning-parser qwen3 \
  --tool-call-parser lfm2 \
  --host 0.0.0.0 \
  --port 30000
```

#### 20. h200 / instruct / bf16 / default / single

- verified: `true`
- model_path: `LiquidAI/LFM2.5-1.2B-Instruct`
- docker_image: `lmsysorg/sglang:dev-cu13`

```bash
sglang serve \
  --trust-remote-code \
  --model-path LiquidAI/LFM2.5-1.2B-Instruct \
  --tp 1 \
  --tool-call-parser lfm2 \
  --host 0.0.0.0 \
  --port 30000
```

#### 21. h200 / jp / bf16 / default / single

- verified: `true`
- model_path: `LiquidAI/LFM2.5-1.2B-JP-202606`
- docker_image: `lmsysorg/sglang:dev-cu13`

```bash
sglang serve \
  --trust-remote-code \
  --model-path LiquidAI/LFM2.5-1.2B-JP-202606 \
  --tp 1 \
  --tool-call-parser lfm2 \
  --host 0.0.0.0 \
  --port 30000
```

#### 22. h200 / thinking / bf16 / default / single

- verified: `true`
- model_path: `LiquidAI/LFM2.5-1.2B-Thinking`
- docker_image: `lmsysorg/sglang:dev-cu13`

```bash
sglang serve \
  --trust-remote-code \
  --model-path LiquidAI/LFM2.5-1.2B-Thinking \
  --tp 1 \
  --reasoning-parser qwen3-thinking \
  --tool-call-parser lfm2 \
  --host 0.0.0.0 \
  --port 30000
```

#### 23. h200 / vl / bf16 / default / single

- verified: `true`
- model_path: `LiquidAI/LFM2.5-VL-1.6B`
- docker_image: `lmsysorg/sglang:dev-cu13`
- env: `SGLANG_USE_CUDA_IPC_TRANSPORT=1, SGLANG_USE_IPC_POOL_HANDLE_CACHE=1`

```bash
SGLANG_USE_CUDA_IPC_TRANSPORT=1 \
SGLANG_USE_IPC_POOL_HANDLE_CACHE=1 \
sglang serve \
  --trust-remote-code \
  --model-path LiquidAI/LFM2.5-VL-1.6B \
  --tp 1 \
  --tool-call-parser lfm2 \
  --host 0.0.0.0 \
  --port 30000
```

#### 24. h200 / vl-450m / bf16 / default / single

- verified: `true`
- model_path: `LiquidAI/LFM2.5-VL-450M`
- docker_image: `lmsysorg/sglang:dev-cu13`
- env: `SGLANG_USE_CUDA_IPC_TRANSPORT=1, SGLANG_USE_IPC_POOL_HANDLE_CACHE=1`

```bash
SGLANG_USE_CUDA_IPC_TRANSPORT=1 \
SGLANG_USE_IPC_POOL_HANDLE_CACHE=1 \
sglang serve \
  --trust-remote-code \
  --model-path LiquidAI/LFM2.5-VL-450M \
  --tp 1 \
  --tool-call-parser lfm2 \
  --mem-fraction-static 0.8 \
  --host 0.0.0.0 \
  --port 30000
```

### Unlimited-OCR

#### 1. h100 / default / default / balanced / single

- verified: `true`
- model_path: `baidu/Unlimited-OCR`
- docker_image: `lmsysorg/sglang:dev`

```bash
sglang serve \
  --model-path baidu/Unlimited-OCR \
  --attention-backend fa3 \
  --page-size 1 \
  --context-length 32768 \
  --enable-custom-logit-processor \
  --disable-radix-cache \
  --host 0.0.0.0 \
  --port 30000
```

### Laguna-M.1

#### 1. b200 / default / bf16 / balanced / single

- verified: `true`
- model_path: `poolside/Laguna-M.1`
- docker_image: `lmsysorg/sglang:dev-cu13-618-nightly`

```bash
sglang serve \
  --model-path poolside/Laguna-M.1 \
  --trust-remote-code \
  --reasoning-parser poolside_v1 \
  --tool-call-parser poolside_v1 \
  --tp 8 \
  --host 0.0.0.0 \
  --port 30000
```

#### 2. b200 / default / fp8 / balanced / single

- verified: `true`
- model_path: `poolside/Laguna-M.1-FP8`
- docker_image: `lmsysorg/sglang:dev-cu13-618-nightly`

```bash
sglang serve \
  --model-path poolside/Laguna-M.1-FP8 \
  --trust-remote-code \
  --reasoning-parser poolside_v1 \
  --tool-call-parser poolside_v1 \
  --tp 8 \
  --fp8-gemm-backend triton \
  --host 0.0.0.0 \
  --port 30000
```

#### 3. b200 / default / nvfp4 / balanced / single

- verified: `true`
- model_path: `poolside/Laguna-M.1-NVFP4`
- docker_image: `lmsysorg/sglang:dev-cu13-618-nightly`

```bash
sglang serve \
  --model-path poolside/Laguna-M.1-NVFP4 \
  --trust-remote-code \
  --reasoning-parser poolside_v1 \
  --tool-call-parser poolside_v1 \
  --tp 8 \
  --host 0.0.0.0 \
  --port 30000
```

#### 4. b300 / default / bf16 / balanced / single

- verified: `null`
- model_path: `poolside/Laguna-M.1`
- docker_image: `lmsysorg/sglang:dev-cu13-618-nightly`

```bash
sglang serve \
  --model-path poolside/Laguna-M.1 \
  --trust-remote-code \
  --reasoning-parser poolside_v1 \
  --tool-call-parser poolside_v1 \
  --tp 8 \
  --host 0.0.0.0 \
  --port 30000
```

#### 5. b300 / default / fp8 / balanced / single

- verified: `null`
- model_path: `poolside/Laguna-M.1-FP8`
- docker_image: `lmsysorg/sglang:dev-cu13-618-nightly`

```bash
sglang serve \
  --model-path poolside/Laguna-M.1-FP8 \
  --trust-remote-code \
  --reasoning-parser poolside_v1 \
  --tool-call-parser poolside_v1 \
  --tp 8 \
  --fp8-gemm-backend triton \
  --host 0.0.0.0 \
  --port 30000
```

#### 6. b300 / default / nvfp4 / balanced / single

- verified: `null`
- model_path: `poolside/Laguna-M.1-NVFP4`
- docker_image: `lmsysorg/sglang:dev-cu13-618-nightly`

```bash
sglang serve \
  --model-path poolside/Laguna-M.1-NVFP4 \
  --trust-remote-code \
  --reasoning-parser poolside_v1 \
  --tool-call-parser poolside_v1 \
  --tp 8 \
  --host 0.0.0.0 \
  --port 30000
```

#### 7. gb200 / default / bf16 / balanced / single

- verified: `null`
- model_path: `poolside/Laguna-M.1`
- docker_image: `lmsysorg/sglang:dev-cu13-618-nightly`

```bash
sglang serve \
  --model-path poolside/Laguna-M.1 \
  --trust-remote-code \
  --reasoning-parser poolside_v1 \
  --tool-call-parser poolside_v1 \
  --tp 4 \
  --host 0.0.0.0 \
  --port 30000
```

#### 8. gb200 / default / fp8 / balanced / single

- verified: `null`
- model_path: `poolside/Laguna-M.1-FP8`
- docker_image: `lmsysorg/sglang:dev-cu13-618-nightly`

```bash
sglang serve \
  --model-path poolside/Laguna-M.1-FP8 \
  --trust-remote-code \
  --reasoning-parser poolside_v1 \
  --tool-call-parser poolside_v1 \
  --tp 4 \
  --fp8-gemm-backend triton \
  --host 0.0.0.0 \
  --port 30000
```

#### 9. gb200 / default / nvfp4 / balanced / single

- verified: `null`
- model_path: `poolside/Laguna-M.1-NVFP4`
- docker_image: `lmsysorg/sglang:dev-cu13-618-nightly`

```bash
sglang serve \
  --model-path poolside/Laguna-M.1-NVFP4 \
  --trust-remote-code \
  --reasoning-parser poolside_v1 \
  --tool-call-parser poolside_v1 \
  --tp 4 \
  --host 0.0.0.0 \
  --port 30000
```

#### 10. gb300 / default / bf16 / balanced / single

- verified: `null`
- model_path: `poolside/Laguna-M.1`
- docker_image: `lmsysorg/sglang:dev-cu13-618-nightly`

```bash
sglang serve \
  --model-path poolside/Laguna-M.1 \
  --trust-remote-code \
  --reasoning-parser poolside_v1 \
  --tool-call-parser poolside_v1 \
  --tp 4 \
  --host 0.0.0.0 \
  --port 30000
```

#### 11. gb300 / default / fp8 / balanced / single

- verified: `null`
- model_path: `poolside/Laguna-M.1-FP8`
- docker_image: `lmsysorg/sglang:dev-cu13-618-nightly`

```bash
sglang serve \
  --model-path poolside/Laguna-M.1-FP8 \
  --trust-remote-code \
  --reasoning-parser poolside_v1 \
  --tool-call-parser poolside_v1 \
  --tp 4 \
  --fp8-gemm-backend triton \
  --host 0.0.0.0 \
  --port 30000
```

#### 12. gb300 / default / nvfp4 / balanced / single

- verified: `null`
- model_path: `poolside/Laguna-M.1-NVFP4`
- docker_image: `lmsysorg/sglang:dev-cu13-618-nightly`

```bash
sglang serve \
  --model-path poolside/Laguna-M.1-NVFP4 \
  --trust-remote-code \
  --reasoning-parser poolside_v1 \
  --tool-call-parser poolside_v1 \
  --tp 4 \
  --host 0.0.0.0 \
  --port 30000
```

#### 13. h200 / default / bf16 / balanced / single

- verified: `true`
- model_path: `poolside/Laguna-M.1`
- docker_image: `lmsysorg/sglang:dev-cu13-618-nightly`

```bash
sglang serve \
  --model-path poolside/Laguna-M.1 \
  --trust-remote-code \
  --reasoning-parser poolside_v1 \
  --tool-call-parser poolside_v1 \
  --tp 8 \
  --host 0.0.0.0 \
  --port 30000
```

#### 14. h200 / default / fp8 / balanced / single

- verified: `true`
- model_path: `poolside/Laguna-M.1-FP8`
- docker_image: `lmsysorg/sglang:dev-cu13-618-nightly`

```bash
sglang serve \
  --model-path poolside/Laguna-M.1-FP8 \
  --trust-remote-code \
  --reasoning-parser poolside_v1 \
  --tool-call-parser poolside_v1 \
  --tp 8 \
  --host 0.0.0.0 \
  --port 30000
```

## Configs with no fixed Deployment cells

### DeepSeek-V4

- source: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`
- reason: I did not include DeepSeek-V4 commands because I could not find fixed `config.cells`/flag rows in the current raw config snapshot; the benchmark file has match tuples and metrics, but not the launch flags needed to reconstruct exact commands safely.
- supported_hardware: `h100, h200, b200, b300, gb200, gb300, rtx6000, mi300x, mi355x`
- variants: `flash, pro`
- quantizations: `fp8, fp4, nvfp4`
- strategies: `low-latency, balanced, high-throughput`
- nodes_options: `single, multi-2`

## Notes for agent-system use

- Treat this file as a baseline lookup table. For production, still tune `--tp`, `--dp`, `--mem-fraction-static`, max concurrency, context length, prefill chunking, and speculative decoding against your traffic shape.
- Prefer `verified: true` rows when available. Treat `verified: false` and `verified: null` rows as starting points that need validation on your fleet.
- For multi-node rows, fill `{{NODE_RANK}}` and `{{NODE0_IP}}` per node before launching.
- The JSON export contains the same Python commands with structured fields and also includes expanded `docker_command` strings.