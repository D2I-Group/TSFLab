---
name: "S4"
description: "Structured state-space model (diagonal S4D-style): ZOH-discretized diagonal SSM kernels applied by FFT convolution in residual blocks, with a linear horizon projection. Use for long input windows where long-range dependencies matter; not when the full NPLR S4 parameterization is required."
---

# S4

## Idea

- `DiagonalSSMKernel` parameterizes a diagonal complex state matrix of the SSM `x'(t) = Ax(t) + Bu(t), y(t) = Cx(t) + Du(t)` and ZOH-discretizes it (`zoh_discretize_diagonal`) into a length-`seq_len` convolution kernel.
- `DiagonalS4Layer` applies that kernel by FFT convolution with a skip term, then a GELU and a GLU-gated 1x1 convolution.
- `S4ResidualBlock` adds pre-norm and residuals; blocks are stacked after `input_projection` (a linear map over all channels).
- `horizon_projection` maps `seq_len` to `pred_len` over time and `output_projection` back to channels; inputs are standardized per window.

## When to use

- Long lookbacks: SSM convolution kernels span the whole window at FFT cost, the paper's motivation for long-range dependencies.
- Per-window standardization absorbs level and scale differences between windows.
- Channels are mixed in the input projection; not a faithful full S4 (diagonal approximation only).

## Configure

- `enc_in` follows the channel count: must equal the number of input channels.
- `c_out` follows the channel count: number of output channels; must equal `enc_in` when `use_norm` is on (defaults to `enc_in`).

Other hyperparameters: preset defaults in `configs/models/S4.toml`; tune generically (`d_state` must be even).

## Differences

Clean-room implementation; the linked Apache-2.0 codebase is reference-only and no source was copied.

- A diagonal S4D-style approximation rather than the paper's full NPLR (low-rank corrected) S4 with Cauchy kernels.
- Exact diagonal zero-order-hold discretization, impulse response, FFT convolution, and gradients are checked locally.
- Inputs are `[B, seq_len, enc_in]`, outputs `[B, pred_len, c_out]`; marks are ignored.

Cite: Albert Gu, Karan Goel, Christopher Re, "Efficiently Modeling Long Sequences with Structured State Spaces", ICLR 2022 (arXiv:2111.00396).
