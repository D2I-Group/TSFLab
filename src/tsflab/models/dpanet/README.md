---
name: "DPANet"
description: "Average-pool and log-band rfft pyramids, one token per level and channel, fused coarse-to-fine by cross-attention. Use for channel-independent forecasting of series mixing patterns at several scales and frequency bands; not for tasks needing cross-channel modelling."
---

# DPANet

## Idea

- Temporal pyramid: level `s` is the normalized history average-pooled `s` times with kernel and stride 2 (length `L // 2^s`).
- Frequency pyramid: the rfft spectrum is cut into `S` contiguous log-spaced bands; each level is the inverse rfft of one band (band 0, the lowest, pairs with the finest temporal level).
- Each channel's level is mapped by a scale-specific `Linear(len_s, d_model)` to a single token (channel-independent), as in the official code.
- `CrossPyramidFusionBlock`: the temporal token attends to the spectral token, then the spectral token to the updated temporal token, then a residual GELU FFN over both.
- Fusion runs coarse-to-fine; finer levels add the coarser fused states, and `Linear(d_model, H)` maps the finest temporal state to the horizon under affine RevIN.

## When to use

- Targets series whose structure differs across time scales and frequency bands, combining a multi-resolution and a spectral view of the same history.
- Channel-independent: no cross-channel interaction, so it ignores useful inter-channel correlation.
- With one token per level, attention reduces to a value projection; expect MLP-like capacity rather than sequence attention.

## Configure

- `enc_in`: number of channels (affine RevIN parameters are per channel).
- `num_scales`: pyramid depth; requires `seq_len // 2^(num_scales - 1) >= 1`.

Other hyperparameters: preset defaults in `configs/models/DPANet.toml`; tune generically.

## Differences

Independent rewrite from the paper (Sections 3.1-3.4, 4.2) after reading `hit636/DPANet` at `f8dbdac2` (Apache-2.0); nothing copied. The official code is followed where it departs from the paper (one-token embedding, sequential cross-attention, band pairing).

- The cataloged `revin` inverts the affine with `gamma + 1e-10` where the official layer uses `gamma + 1e-8`.
- The configuration is validated up front instead of failing in the forward pass.
- Hard-coded official values (`num_scales`, `d_model`, `n_heads`) are defaults here and remain configurable.
- Paper ablation variants are not implemented (not in the code). Details in `reference.md`.
