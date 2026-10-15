---
name: "LMSAutoTSF"
description: "Multi-scale pooled inputs split into trend and seasonal parts by learnable FFT sigmoid filters, with lag-difference-gated MLP encoders. Use for series with trend plus seasonal structure at several resolutions; not for very short lookbacks (each scale halves seq_len)."
---

# LMSAutoTSF

## Idea

- `Model.multi_scale` (Eq. 2): the window and its successive window-2 average-pooled copies give `num_scales` (K = 4) resolutions.
- `learnable_decomposition` / `frequency_masks` (Eqs. 3-6): FFT, masks `sigmoid(-(f - f_c) s)` and `sigmoid((f - f_c) s)` with trainable cutoff `f_c` (init 0.2) and steepness `s` (init 10) per scale and channel; inverse FFTs give trend and seasonal parts that sum to the input.
- `ScaleEncoder` (Eqs. 8-10): a temporal MLP output multiplied by `lagged_difference` of its input, added back to the input, plus a channel MLP, then a linear map to the horizon; separate encoders per part and scale, summed (Eqs. 11-12).
- `Model.forward` (Eq. 13): per-scale forecasts are concatenated along time and projected to the horizon, inside `revin`.

## When to use

- Designed for series whose trend and seasonal components should be separated adaptively: the cutoff between them is learned per scale and channel.
- Multi-scale pooling targets patterns at several temporal resolutions; the lagged difference acts as an integrated autocorrelation gate.
- Channel mixing is on by default (`channel_independence = false`); switch it on for weakly correlated channels.
- RevIN handles level shifts across windows.

## Configure

- `enc_in`: must equal the channel count.
- `num_scales`: `seq_len // 2**(num_scales - 1)` must be at least 1 (window-2 average pooling per scale).

Other hyperparameters: preset defaults in `configs/models/LMSAutoTSF.toml`; tune generically.

## Differences

- Independent rewrite from Section 4 (Eqs. 2-13) after reading the pinned official code (no license file, `NOASSERTION`); nothing copied.
- Eq. (10) as printed omits the input residual; this entry follows the official encoder, which adds it.
- Mask frequency grid follows the official signed `fftfreq` grid, giving an effective mask `(m(f) + m(-f)) / 2`.
- The EAAI extension (`LMSAutoTSFV2`) and the anomaly-detection head are not implemented.
