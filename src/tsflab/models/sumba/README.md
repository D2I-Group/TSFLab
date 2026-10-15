---
name: "Sumba"
description: "Multivariate forecaster whose per-sample dynamic channel graph is a convex combination of a few learnable low-rank matrix bases, with multi-kernel gated temporal convs. Use for correlated multivariate series whose cross-channel structure drifts over time; not for weakly correlated channels or very many channels."
---

# Sumba

## Idea

- `StructuredMatrixBasis` keeps a few learnable low-rank row-stochastic matrices and mixes them with softmax coefficients predicted from the pooled hidden state, so each sample's graph lies in a constrained, lower-variance space.
- `DynamicBasisGraphConv` propagates over that graph for several diffusion steps with a residual mix, then projects the concatenated steps.
- `MultiScaleTemporalConv` runs gated depthwise causal convs with kernels (2, 3, 5) inside each pre-LayerNorm `SumbaBlock`.
- Each variable is z-scored over the lookback (statistics detached) and restored after a flatten linear head.

## When to use

- Multivariate data with spatial correlations that have structure and evolve over time (the paper's motivation); the basis mixture is interpretable as dynamics.
- Short input windows where two-stage graph learning is noisy: the convex basis parameterization constrains the generated graph.
- No predefined graph is needed; channels play the role of nodes.
- Not for weakly correlated channels (channel mixing adds little) or very large channel counts (bases are `N x N`).

## Configure

- `enc_in`: number of channels; the bases are sized `enc_in x enc_in`.

Other hyperparameters: preset defaults in `configs/models/Sumba.toml`; tune generically.

## Differences

- Clean-room implementation; the unlicensed reference repository is link-only and nothing was copied.
- Paper mapping: structured parameterization → `StructuredMatrixBasis`; convex dynamic generation → its context-conditioned `forward`; temporal modeling → `MultiScaleTemporalConv`; spatial propagation → `DynamicBasisGraphConv`.
- The structure regularization exists as `diversity_penalty()` but is not added to the training loss; dataset-specific regularization weights are not reproduced.
