---
name: "TALON"
description: "LLM-based forecaster: segments are routed by trend/variation/autocorrelation statistics to Linear, CNN or LSTM experts whose outputs feed a frozen GPT-2 trunk, with segment-by-segment rolling forecasts. Use for long-term forecasting of temporally heterogeneous series; not for tight compute budgets or short lookbacks."
---

# TALON

## Idea

- `PatchComplexity` computes per-segment cues `c = [trend strength, local variation, |lag-1 autocorrelation|]` (Appendix D, Algorithm 1) with STL expressed as fixed linear operators.
- `HeterogeneousTemporalEncoder` routes each instance-normalized segment (`revin`) with noisy top-k gating (Eqs. 1-6) over `LinearExpert`, `CNNExpert` and `LSTMExpert` (Eqs. 7-10), plus the importance/load balance loss of Eq. 11.
- Expert outputs are GPT-2 tokens for a frozen `gpt2_backbone` trunk truncated to `llm_layers = 6`; `SegmentMLP` maps every position to the next segment (Eq. 13).
- `training_objective` supervises all next-segment predictions plus `alpha * L_MoE` (Eq. 14 without `beta * L_align`); `forward` rolls the forecast segment by segment, recomputing routing cues on every shifted window.

## When to use

- Long-term forecasting where segments differ in character (smooth trend vs. volatile vs. autocorrelated), which the statistic-driven expert routing targets.
- Long lookbacks made of whole segments (the preset uses 672 = 7 segments of 96).
- Requires the pinned GPT-2 artifact and a GPU-scale budget (768-wide tokens, 1024-wide experts and MLP); not for tight compute budgets.
- Channel-independent: not for data where cross-channel interactions drive the target; marks are ignored.

## Configure

- `enc_in`: number of channels.
- `token_len`: segment length (at least 3); `seq_len` must be a multiple of `token_len`, and `seq_len / token_len` must fit the GPT-2 context. The STL period inside each segment is `max(token_len // 2, 2)`.
- `pred_len`: rolled out in `token_len` steps; with `pred_len < token_len` only the available future values are supervised.

Other hyperparameters: preset defaults in `configs/models/TALON.toml`; tune generically.

## Differences

- Independent rewrite from Section IV (Eqs. 1-14), Section V-A and Appendices B and D after reading the pinned official code (no license file, `NOASSERTION`); nothing copied.
- The Representation Alignment Module (prompts, frozen-LLM prompt embeddings, InfoNCE `beta * L_align`, Eq. 12) is not implemented; it is training-only, so inference follows the paper and training uses `beta = 0`.
- `W_H` is applied in both training and evaluation, the routing load is computed in the Eq. 4 score space, and routing cues are recomputed for every rolled window (official code deviates; see `issues`).
- GPT-2 runs in float32 (official: float16 with AMP); experts run only on segments routed to them.
- Preset follows the released ETTh1 script (`top_k = 3`), not the paper's best `k = 2`.
- Full detail: reference.md, `## Differences in detail`.
