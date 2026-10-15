---
name: "SDGFNet"
description: "Static prior graph plus per-wavelet-band dynamic graphs over channels, attention-gated fusion, dilated inception block, and linear head under RevIN. Use for multivariate data whose channel correlations are strong and change across scales; not for weakly correlated channels or probabilistic output."
---

# SDGFNet

## Idea

- `Model.static_branch` (Eqs. 1-3): a per-sample prior adjacency (Gaussian kernel of squared distances, or `softmax(relu(PCC))` with `static_graph = "pcc"`) drives `StaticMixProp` (self loops, degree normalization, `depth` mix-hop steps `h <- a x + (1 - a) A h`, 1x1 conv); a full-history conv maps back to `L` steps with a residual LayerNorm.
- `wavelet_band_projections` (Sec. 3.3): a level-`J` db4 DWT with symmetric extension; each band (approximation, then details finest first) is inverted alone back to `L` samples, so the `J + 1` bands sum to the input.
- `DynamicMixProp` (Eqs. 4-5): per band and time step, the adjacency is the mean of row and column softmax of `tanh(W1 x)^T tanh(W2 x)`, then the same mix-hop propagation; one dynamic branch is shared by all bands.
- `AttentionGatedFusion` (Eqs. 7-9) summarizes each graph output (linear map of the flattened `L x N` values plus LayerNorm), scores it against a learned query and softmax-weights them into one `[B, L, N]` tensor.
- `InceptionBlock` (Eqs. 10-12): four dilated convolutions (`k` in {3, 5}, `d` in {1, 2}) with bottleneck, residual and LayerNorm, treating the `L` history positions as channels; a linear head maps `L` to the horizon and RevIN denormalizes.

## When to use

- Multivariate data with strong inter-channel correlation, where dependencies differ by time scale (static prior plus wavelet-band dynamic graphs).
- No external graph needed: graphs are built from each input window.
- The inception convolutions slide over the variable axis, so results depend on channel order; fusion projects `L x N` values, so width grows with channel count. Point forecasts only.

## Configure

- `enc_in` follows the dataset channel count; it must equal it exactly.
- Other hyperparameters: preset defaults in `configs/models/SDGFNet.toml` (from official `scripts/ETTh1.sh`); tune generically.

## Differences

Independent rewrite of arXiv 2509.18135 (v2) after reading `shaoxun6033/SDGFNet@bf76b56f` (Apache-2.0); no source copied. Recorded in `card.toml` issues:

- The static prior graph is per sample, not batch-averaged (the official batch average leaks later test windows into each prediction).
- The db4 analysis/synthesis is written as fixed `L x L` band-projection matrices with PyWavelets filter taps, replacing the vendored `pytorch_wavelets` DWT.
- Default static graph is the official RBF kernel (`sigma = 1`, close to identity after self loops); `static_graph = "pcc"` implements the paper's Eqs. (1)-(2), broken in the official code.
- Otherwise follows the official structure: shared dynamic branch, per-step dynamic adjacency, flattened-summary fusion (width 128, `fusion_dim`), variable-axis inception (width 512, `inception_width`); dead official top-k sparsification and unused CLI flags (dropout, `d_model`) are omitted.
- Defaults: `decomp_level = 3`, `conv_channel = 32`, `gcn_depth = 2`, `propalpha = 0.3`, MSE loss. No checkpoint or published-metric reproduction is claimed.

Citation: Wang, Zhang, Li, Cao, Tan, "SDGF: Fusing Static and Multi-Scale Dynamic Correlations for Multivariate Time Series Forecasting", arXiv:2509.18135, 2025.
