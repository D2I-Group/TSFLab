---
name: "HADL"
description: "Linear forecaster: Haar approximation drops the detail band, a scaled DCT-II gives frequency features, and one shared low-rank layer maps them to the horizon. Use for long, noisy lookbacks under tight compute; not for channel interactions or probabilistic output."
---

# HADL

## Idea

- Haar step (Eq. 2): a one-level Haar DWT (`haar_dwt1d`) keeps only the approximation `(x_2i + x_2i+1)/sqrt(2)` of the mean-centered window; the detail band is dropped as noise, halving the input length (odd lengths are zero-padded).
- Frequency features (Eq. 4): orthonormal DCT-II of the approximation divided by its length, `A_F = (2/L) DCT(A_T)`, as a fixed matrix.
- `LowRankLinear` (Eq. 5): `Y = A_F P Q + B` with `P: [L/2, r]`, `Q: [r, H]`, shared by all channels, mapping frequency coefficients straight to time-domain forecasts (no inverse DCT, Sec. 3.4).
- The removed channel mean is added back; while training `aux_loss = rate * mean|Y|` (the official L1 regularizer, default 0.1) is exposed.
- `enable_haar`, `enable_dct`, `enable_lowrank` and `individual` switch the parts the paper ablates (Sec. 6, Appendix C).

## When to use

- Designed for long-term forecasting from long, noisy lookbacks (scripts use `seq_len = 512`): the Haar low-pass and L1 penalty suppress high-frequency noise.
- Very small parameter count (one shared low-rank layer), suited to tight compute or memory budgets.
- Not for data where the dropped high-frequency detail or cross-channel interactions carry the signal (channels share one linear map, no mixing).
- Point output only.

## Configure

- `enc_in`: the dataset's channel count (matters with `individual = true`).
- `rank`: low-rank width; the paper asks `r << min(seq_len / 2, pred_len)`; the scripts use 50 with `seq_len = 512` (which violates this for `pred_len = 96`).

Other hyperparameters: preset defaults in `configs/models/HADL.toml`; tune generically.

## Differences

- Independent rewrite of Sec. 3 (Eqs. 1-5) and Algorithm 1 after reading the pinned official code; nothing copied. Mean centering, DCT scaling, initialization and the L1 penalty follow the code.
- The official L1 regularizer penalizes the forecast, not the weights (paper says weights); the penalty here does not depend on precision (the official AMP branch drops it).
- The DCT is a fixed differentiable matrix instead of a scipy CPU round trip; the buggy inverse-DCT option (`--enable_iDCT`, off in all scripts) is not provided.
- With `MS` features the official code penalizes only the target channel; here the full forecast tensor.
- Training uses the catalog trainer and configured loss; reported benchmark numbers are not reproduction claims.

Full detail: `reference.md`.
