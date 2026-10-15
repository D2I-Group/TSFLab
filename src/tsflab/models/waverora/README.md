---
name: "WaveRoRA"
description: "Wavelet-domain inverted Transformer: DWT coefficient embeddings form one token per variate, mixed by rotary router-token attention at linear cost in variates; per-band heads and inverse DWT. Use for multivariate data with many correlated channels; not for weakly related channels or calendar-driven tasks."
---

# WaveRoRA

## Idea

- `orthogonal_dwt` (`mode="zero"`, Symlet-3 default; Coiflet-3, Haar) splits each normalized variate into `J` detail and one approximation series (Eqs. 1-4); the inverse DWT rebuilds the horizon.
- Every coefficient series has its own `Linear(len_j, D)` embedding and `Linear(D, 2D) -> GELU -> Linear(2D, H_j)` predictor (Eqs. 7, 13).
- `RotaryRouteAttention` (Alg. 1, Eq. 11): `r` learned routing tokens gather from all variates and every variate reads from them, with rotary angles by variate position; cost is linear in the number of variates. A linear skip and a SiLU gate complete the layer.
- `WaveNormBlock` (Eq. 12): a variate's `J + 1` embeddings are concatenated and projected to `d_model` for attention, then split back; each band gets a post-norm residual block with a convolutional feed-forward along the variate axis.
- Non-affine `revin` wraps the model.

## When to use

- Designed for multivariate forecasting where variates inform each other; routing tokens keep attention linear in channel count, so it scales to hundreds of channels.
- The wavelet split separates trend-like and fine-scale content before mixing.
- Calendar marks are not used; point output only.

## Configure

- `enc_in`: must equal the dataset channel count.
- `router_num`: routing tokens; `0` applies the official rule `int(sqrt(M) + log2(M)) // 2` rounded up to even (at least 2) for `M` channels; the official Electricity and Traffic scripts pass 20.
- `kernel_size`: odd convolution width over the variate axis; values above 1 mix neighbouring variates in file order, so keep 1 unless channel order is meaningful.
- `wavelet_layers`: decomposition levels relative to `seq_len` and `pred_len` (the scripts use 2-5 per dataset and horizon; preset 3 for `seq_len = 96`).

Other hyperparameters: preset defaults in `configs/models/WaveRoRA.toml`; tune generically.

## Differences

- Follows the code where it departs from the paper: values are the input tokens, the router is an unprojected learned parameter, no output projection, `d_model` projections around the attention, and a two-layer predictor.
- `rotary`, `gate`, and `residual` default to on (official flags default off but every script enables them).
- Odd horizons are cropped instead of failing; at least two routing tokens; the predictor is always GELU.
- The DWT is reimplemented with tabulated filters instead of `pytorch_wavelets`; the RevIN standard deviation is detached.
- Not implemented: time/frequency-domain variants, SA/LA attention ablations, and calendar tokens.
