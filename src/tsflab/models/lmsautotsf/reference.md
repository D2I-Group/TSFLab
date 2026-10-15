# LMSAutoTSF — reference

## Differences in detail

- Sources read (`mribrahim/LMS-TSF`, revision `41e4fa1c1d0db6780aa76ccd528c97052f1ddc00`): `models/LMSAutoTSF.py`, `layers/StandardNorm.py`, `run.py`, `scripts/long_term_forecast/*/LMSAutoTSF*.sh`. Nothing was copied or imported. The README's later EAAI extension (`models/LMSAutoTSFV2.py`, confidence-aware forecasting) is a different method and is not implemented.
- Resolved from the official code: three window-2 average poolings (K = 4); the masks use the signed `fftfreq` grid on a full complex FFT and keep the real part of the inverse, so the effective symmetric mask is `(m(f) + m(-f)) / 2`; cutoff and steepness start at 0.2 and 10 for every scale and channel.
- Each encoder's temporal and channel blocks are Linear-ReLU-Linear-Dropout MLPs; the lagged difference keeps the first step unchanged; the lag-gated features are added to the encoder input before the channel MLP and the horizon projection.
- Normalization is RevIN with a learnable affine (the catalog `revin` is equivalent to the official `Normalize`, including the `eps^2` affine inversion).
- Defaults: `d_model = 512`, dropout 0.1, and channel mixing on (`--channel_independence 0` in every official script).
- Eq. (10) as printed projects `x_temp + x_channel` without the input residual; this entry follows the official encoder, which adds the input as well.
- The official anomaly-detection head (RNN feedback) is out of scope. Training uses the catalog trainer and loss (the paper trains with L2 loss, Adam, learning rate `1e-4`, batch size 32). Reported benchmark numbers are not reproduction claims of this implementation.

## Citation

```bibtex
@article{delibasoglu2024lms,
  title   = {LMS-AutoTSF: Learnable Multi-Scale Decomposition and Integrated Autocorrelation for Time Series Forecasting},
  author  = {Delibasoglu, Ibrahim and Chakraborty, Sanjay and Heintz, Fredrik},
  journal = {arXiv preprint arXiv:2412.06866},
  year    = {2024}
}
```
