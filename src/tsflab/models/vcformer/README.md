---
name: "VCformer"
description: "Inverted-token Transformer whose attention scores variable pairs by learned-weighted lagged cross-correlation, plus a Koopman (eDMD) temporal detector per token. Use for multivariate data with lead-lag relations between variables; not for hundreds of channels (pairwise all-lag maps) or weakly related channels."
---

# VCformer

## Idea

- `DataEmbedding_inverted` maps each normalized variate history (and, with `use_marks`, each hourly `timeF` calendar feature) to one `d_model` token.
- `VariableCorrelationAttention` (VCA, Eqs. 3-6): FFT lagged cross-correlation for every token pair and every circular lag, aggregated with learnable lag weights; its row softmax mixes the value tokens.
- `KoopmanTemporalDetector` (KTD, Eqs. 7-10): cuts each token into `d_model / snap_size` snapshots, encodes them, fits `K = pinv(Z_back) Z_fore` per token, rolls out `z_P K^t`, and decodes back.
- `VCformerLayer`: `LayerNorm(X + VCA(X))` then `LayerNorm(X + KTD(X))`; a final LayerNorm and a linear token-to-horizon projection inside non-affine `revin`.

## When to use

- Designed for multivariate series where variables are correlated with time lags, which plain attention scores miss.
- Calendar features become extra tokens, but only for hourly data.
- The correlation map is `N x N x d_model` per sample, so cost grows quickly with channel count.
- Point output only.

## Configure

- `enc_in`: must equal the dataset channel count.
- `use_marks` / `freq`: calendar tokens support hourly `timeF` marks only (`freq = "h"`); set `use_marks = false` for other sampling rates.

Other hyperparameters: preset defaults in `configs/models/VCformer.toml`; tune generically.

## Differences

- Lag weights of Eq. (5) are learnable (initialized to `1 / d_model`, equal to the official plain mean over lags at initialization).
- The rollout follows Eq. (9) from the last snapshot; the official code advances every snapshot by `K^(d_model / snap_size)`.
- `snap_size` must divide `d_model` instead of the official left padding, which breaks the residual.
- Lags run over the `d_model` axis of projected tokens, as in the code (the paper writes length-`T` series).
- The RevIN scale is detached (catalog `revin`); the official code backpropagates through the standard deviation.
- Official dead code (overwritten per-variate loop, unused convolutions, `factor`, `n_heads`) is not reproduced.
