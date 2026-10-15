---
name: "Fredformer"
description: "Full-FFT frequency-patch Transformer with cross-channel attention and separate real/imaginary residual forecast heads. Use for correlated multivariate forecasting; channel attention grows quadratically with channel count."
---

# Fredformer

## Idea

- RevIN normalizes each input variable; a full complex FFT supplies real and imaginary frequency patches.
- Each frequency patch forms channel tokens for a pre-normalized Transformer with independent Q/K/V projections.
- Separate real/imaginary projections and three-layer residual MLP heads predict the horizon spectrum.
- Inverse FFT followed by a real/imaginary projection produces the forecast; RevIN restores the input scale.

## When to use

- Multivariate series with correlated channels and relevant mid/high-frequency structure.
- Point forecasts; no calendar marks or probabilistic head.
- Channel attention scales quadratically; validate memory at the actual channel count.

## Configure

- `enc_in`: dataset channel count.
- `band_width`: frequency patch length; must fit `seq_len`.
- `stride`: frequency patch stride; patches are unpadded as in the reference.
- Other dimensions and dropout: `configs/models/Fredformer.toml`.

## Differences

Independent implementation checked against the standard forecasting path at official revision `fa64775e`; no official source is vendored or used at runtime. See `reference.md`.

- Uses fused Q/K/V storage with the same independent affine projections.
- Omits unused official ablation parameters, unused layers, the individual-channel head and the Nystrom variant.
- Preserves batch/channel dimensions instead of the official unrestricted squeeze.
- Legacy TSFLab RMS-band equalization, learned band gate and history-linear head were removed because they differ materially from the reference.
