---
name: "TFKAN"
description: "Dual-branch Kolmogorov-Arnold network: a shared spline KAN on the rFFT of each lifted channel, a KAN along time, and a KAN predictor to the horizon. Use for channel-independent long-term forecasting with mixed time/frequency structure; not for cross-channel structure or covariates."
---

# TFKAN

## Idea

- Layers are B-spline KANs (Eqs. 1-4): a SiLU base path plus per-edge spline functions on a uniform knot grid (`grid_size`, `spline_order`), built on the cataloged `bspline_basis`.
- Dimension adjustment (Eq. 5): the frequency branch lifts each channel's series with a learnable `[1, d]` vector; the time branch keeps the raw `[N, L]` series.
- Frequency branch (Eqs. 6-9): orthonormal rFFT along time, one shared FreqKAN on real and imaginary parts, soft-shrinkage, inverse rFFT; time branch (Eq. 10): TimeKAN along time per channel.
- Both branch outputs plus the bias term `F_t + T_t` (Eq. 11) are flattened (`L x d`) and mapped to the horizon by the KAN predictor (Eq. 12).

## When to use

- Long-term forecasting where structure shows in both the time and the frequency domain, and an interpretable nonlinear (spline) mapping is wanted.
- Channels that can be modelled independently with shared weights.
- No normalization is built in: data with level shifts relies on the pipeline's scaling.
- Not for cross-channel interactions, covariates, or tight memory budgets (the predictor KAN maps `seq_len * embed_size` inputs).

## Configure

- `enc_in`: number of channels; input must be `[B, seq_len, enc_in]`.

Other hyperparameters: preset defaults in `configs/models/TFKAN.toml`; tune generically.

## Differences

- Independent rewrite (the official repository has no license file, `NOASSERTION`; nothing copied).
- KAN hidden width `256` (code) rather than the paper's 258; the three KANs share `hidden_size`, and the predictor is the code's two-layer `L*d -> 256 -> tau` KAN rather than the paper's single KAN.
- The code's soft-shrinkage before the inverse FFT (not in the paper) is kept as `sparsity_threshold = 0.001` (0 disables it).
- The bias term of Eq. (11) is on by default (`use_bias = true`, as in every released script).
- KAN initialization follows the official `KANLinear.reset_parameters`; the code's unused regularization settings are not reproduced.
