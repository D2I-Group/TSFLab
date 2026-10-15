---
name: "LiNo"
description: "Recursive residual decomposition alternating a full-receptive AR linear extractor and a time-frequency nonlinear block, inside RevIN. Use for multivariate series mixing linear trends and nonlinear patterns; not for strictly channel-independent setups (the No block pools variates)."
---

# LiNo

## Idea

- `LiNoBlock` is one Recursive Residual Decomposition level (Eq. 1): `R^L = H - Li(H)`, then `R^N = R^L - No(R^L)`; `R^N` feeds the next block.
- `LinearExtractor` (Li block, Eq. 2) is a per-variate autoregressive filter: a depthwise convolution whose kernel spans the whole `d_model` axis over the input left-padded with zeros, started as a uniform average.
- `NonlinearExtractor` (No block) fuses `Tanh(Linear(R) + FrequencyLinear(R))` (complex linear map over rFFT bins); `softmax_channel_mean` pools variates for an inter-series MLP; two LayerNorm residual stages give `N`.
- `Model.forward` (Eq. 4) sums every level's linear and nonlinear horizon projections, inside non-affine `revin`, after the inverted `embed` token embedding.

## When to use

- Designed for robust forecasting of series that interleave linear (trend, AR) and nonlinear patterns, by peeling them off level by level.
- Non-affine RevIN handles level and scale drift across windows.
- The No block mixes variates through a softmax-weighted channel mean; with independent channels this adds little.
- The inverted embedding works on whole-window tokens per variate, so `d_model` is the working length, not `seq_len`.

## Configure

- `enc_in`: must equal the channel count (variates become inverted tokens).

Other hyperparameters: preset defaults in `configs/models/LiNo.toml`; tune generically.

## Differences

- Independent rewrite from Section 3 after reading the pinned official code (no license file, `NOASSERTION`); nothing copied.
- The No block concatenates the channel mean with the block input `R^L` as in the official code, not with `N^TF` as in the paper text.
- The inverse rFFT gets the explicit length `d_model` so odd widths are well defined.
- Underspecified Li-block AR coefficients follow the official code; see reference.md.
- `Model.block_predictions` returns normalized per-level forecasts instead of the paper's visualisation return.
