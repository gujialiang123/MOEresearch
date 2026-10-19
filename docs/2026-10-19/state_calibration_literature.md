# State calibration: primary-source implementation notes

Checked 2026-10-19 against the full, versioned arXiv HTML papers. This is a source audit, not an experimental result or a claim of novelty. The user-supplied experiment plan governs the new study. No applicable `AGENTS.md` was found in this checkout or its parent directories.

## LDA: requirements verified from the full paper

Kim, Soro, and Hwang, **Distribution-Consistent Inference for Dynamic Sparse Mixture-of-Experts**, [arXiv:2609.09241v1](https://arxiv.org/html/2609.09241v1), September 8, 2026; §4.2–4.3 and Appendix G, Algorithms 1–2.

- Algorithm 1 computes native and reduced SMoE outputs on the **same layer input**. Estimate a mean and standard deviation **across tokens, separately for each output dimension**. It enumerates `k=1,...,k0`.
- The next layer receives the **default `k0` forward** (Algorithm 1, line 24); reduced branches do not propagate during baseline calibration.
- Algorithm 2 corrects the SMoE output before layer continuation, including residual addition. Native `k=k0` is identity.

\[
\hat y_k^{(l)}=\sigma_{k_0}^{(l)}\odot
\frac{y_k^{(l)}-\mu_k^{(l)}}{\sigma_k^{(l)}+\epsilon}
+\mu_{k_0}^{(l)},\quad k<k_0.
\]

The denominator is **standard deviation plus epsilon**. A scalar-only correction is not this method. §5.1 uses C4 `4×2048` calibration tokens. Numerical epsilon, variance `ddof`, and low-variance clamps were not specified in the inspected algorithm/formula. No author implementation repository was verified; do not claim code-level equivalence. The paper also studies routing-trajectory mismatch (§3.3), so observing trajectory divergence alone is not a new contribution. [Full paper](https://arxiv.org/html/2609.09241v1)

## Reference-set baseline

Chen and Yao, **Training-Free Halving of Activated Experts in Fine-Grained Mixture-of-Experts Models**, [arXiv:2609.04575v1](https://arxiv.org/html/2609.04575v1), September 4, 2026. Its Eq. 2 separates experts computed (`k1`) from normalization reference (`k2`):

\[
w_i=p_i/\sum_{j\in\mathrm{top}\text{-}k_2}p_j,
\quad i\in\mathrm{top}\text{-}k_1.
\]

Only `k1` expert FFNs execute. `k2=k1` is survivor renormalization; `k2=k_native` preserves the native reference; `k2=E` removes normalization over selected experts. Hence this repository's native-reference `no_renorm` name differs from the paper's fully unnormalized endpoint. The paper reports greedy GSM8K generation and truncation at **512 tokens without an answer** (Table 6), which differs from the new study's explicit EOS/cap stop reasons. It uses Qwen3.6-35B-A3B and Qwen3.5-397B-A17B, not the main checkpoint here. [Method and evaluation](https://arxiv.org/html/2609.04575v1)

## Fixed histories and free continuation

Nguyen et al., **To Reason but Not to Stop: Understanding and Recovering Pruned Large Reasoning Models**, [arXiv:2610.19934v1](https://arxiv.org/html/2610.19934v1), October 14, 2026. §3 compares teacher-forced stopping and continuation preferences with free-running behavior. §4 builds TGAS from the within-prefix difference between mean hidden states of terminating and looping continuations, then steers the residual stream. It studies weight-pruned reasoning models; its stopping marker includes `</think>`. This supplies relevant precedent for fixed-history versus free-generation diagnosis and activation intervention, but does not establish the new study's hypothesis for reduced-expert, non-thinking Qwen. [Full paper, §3–4](https://arxiv.org/html/2610.19934v1)

## Official model facts

The official [Qwen model card](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) identifies this checkpoint as **non-thinking**, with 128 experts, 8 activated experts, and 48 layers. The [official config](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/blob/main/config.json) sets `norm_topk_prob=true`, `hidden_size=2048`, and BF16. Its [generation config](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/blob/main/generation_config.json) lists EOS IDs **151645 and 151643**. Runtime must resolve these files from the pinned local revision, because these links track `main`; the config's single EOS ID alone is insufficient. Do not use `</think>` as this checkpoint's termination target.

## Controlled adaptation and contribution boundary for this study

The plan deliberately changes the calibration corpus to GSM8K training rollouts and limits interventions/statistics to decode positions, with native prefill. Label `L_N` **LDA under our decode-only protocol**, not a reproduction of the paper's absolute benchmark scores. Use full-renormalized top-4 output as its reduced branch. Pool all valid fit tokens with token weighting and `ddof=0`; record FP32/FP64 accumulation, epsilon, any variance protection, and model-dtype → FP32 affine → model-dtype casts. These are explicit study conventions where the paper is underspecified.

For `M_NT` and `M_ST`, keep formula, capacity, text, position mask, and token count equal; change only the driver that supplies states. Their shadow native target must be computed at each driver's own input. Each updated model must then be evaluated on its newly generated own trajectory. The proposed evidence is the same-text driver contrast and any verified benefit beyond strong LDA/reference controls. General rescaling, moment correction, longer outputs, stopping differences, and using student data cannot alone support a novelty claim. A broader current related-work audit remains necessary before an ACL novelty assertion.
