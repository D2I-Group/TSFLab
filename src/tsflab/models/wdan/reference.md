# WDAN — reference

## Paper

Wavelet-based Disentangled Adaptive Normalization for Non-stationary Times Series Forecasting (arXiv 2506.05857, 2025). Sources used: Methodology (Eqs. 1-17) and the experiment-setting appendix.

```bibtex
@article{lin2025wdan,
  title   = {Wavelet-based Disentangled Adaptive Normalization for Non-stationary Times Series Forecasting},
  author  = {Lin, Junpeng and Lan, Tian and Zhang, Bo and Lin, Ke and Miao, Dandan and He, Huiru and Ye, Jiantao and Zhang, Chen and Li, Yan-fu},
  journal = {arXiv preprint arXiv:2506.05857},
  year    = {2025}
}
```

## Implementation mapping

Independent rewrite after reading the pinned official code (`MonBG/WDAN` at `f01994ad`; no license file, recorded as `NOASSERTION`): `nns/models/WDAN.py`, `nns/layers/learnable_dwt.py`, `nns/spv/WDAN_spv.py`, `run.py`, `scripts/WDAN_PatchTST.sh`. Nothing was copied or imported.

Resolved from the official code:

- DWT: correlates a half-sample symmetric extension with the decomposition filters at stride 2 and inverts with the transposed operation; filters fixed (the learnable option is off in `run.py`).
- Statistics module: one linear layer with dropout per input; a linear fusion of three embeddings per branch with shared trend and residual embeddings; bias-free GELU feed-forward blocks without residual; linear heads to the horizon; ReLU on the predicted scale.
- Statistics targets are the horizon's own trend and residual sliding scale, with the forecast loss (MSE).
- Stage three adds the statistics parameters to the backbone optimizer at the same learning rate after `twice_epoch` backbone epochs.
- Defaults from `run.py`: `coif3`, one DWT level, window 5, width 512/1024, two feed-forward blocks, dropout 0.1, 5 statistics epochs, statistics learning rate `1e-4`, `twice_epoch = 1`; backbone normalization disabled (`use_norm 0`).

## Differences in detail

- Backbone: catalog `patchtst` (official-PatchTST layout, preset defaults, RevIN off) instead of the TSLib-style iTransformer, PatchTST, Crossformer, and FEDformer backbones.
- Residual: taken as `x - x_l`; Eq. (3) sums inverse transforms of the detail bands, which is equal because the filter bank reconstructs perfectly.
- Eq. (5): population standard deviation; the code's `sliding_std` uses `x_window.std(dim=-1)` (unbiased).
- The stage-2/3 switch counts training batches (one epoch = the training loader length recorded before the first epoch).
- DWT convention: the local `WaveletTrend` is not the cataloged `orthogonal_dwt`. It correlates the symmetrically extended signal (`L - 1` samples left, `L - 1` or `L` right) with the unreversed decomposition filters, `a[n] = sum_j rec_lo[j] x[2n - j]` instead of `sum_j dec_lo[j] x[2n + 1 - j]`, so its coefficients are mirrored and shifted by one sample relative to PyWavelets; its db4 taps follow the `wavelet` table rather than PyWavelets.
