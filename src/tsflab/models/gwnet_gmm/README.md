---
name: "GWNetGMM"
description: "Graph WaveNet whose output layer is a per-location Gaussian mixture with anchored means, trained by mixture NLL. Use for traffic sensor networks with multi-modal, uncertain future states; not for data without nodes, or when calibrated intervals must be scored inside TSFLab."
---

# GWNetGMM

## Idea

- The method (Eq. 3) keeps a spatio-temporal backbone and swaps only its last layer for `GaussianMixtureHead`: a per-location `Linear -> LayerNorm -> ReLU -> Dropout` projection (Eq. 6) feeds three linear branches giving `K` mixing weights (softmax, Eq. 8), mean offsets and log-variances per future step (Eq. 7).
- Anchored means: `mu = r + offset * s * (1 + spacing_scale)` with references `r = [-2, -1, 0, 1, 2]` covering the z-scored 3-sigma range and `s = 1` (Eq. 9).
- The head starts as a uniform mixture of unit Gaussians on the anchors (weakly informative prior).
- Training uses only the mixture negative log-likelihood (Eq. 5, NaN labels masked); `forward` returns the mixture mean (Eq. 10) as the point forecast, and `Model.mixture` exposes the mixture.
- Backbone: Graph WaveNet with value and time-of-day inputs, valid dilated gated convolutions that collapse `seq_len + 1` steps to one, diffusion convolution over the single adaptive adjacency `softmax(relu(E1 E2))`, residual plus BatchNorm, summed skips.

## When to use

- Designed for traffic forecasting where future states are stochastic and multi-modal (e.g. free flow vs congestion), so a single Gaussian or point forecast is too narrow.
- Needs a node-structured dataset; the graph is learned (predefined adjacency is unused).
- Uses time of day as an input, so timestamps should carry signal.
- Inside TSFLab the runner scores only the mixture mean with point metrics; probabilistic metrics (CRPS, intervals) are not wired in.

## Configure

- `enc_in`: the dataset's node count (runner-injected `num_nodes` takes precedence).
- `layers` (with `blocks`, `kernel_size`): `seq_len + 1` must not exceed the receptive field `1 + blocks * (2**layers - 1) * (kernel_size - 1)` (13 with the preset, for 12 input steps).

Other hyperparameters: preset defaults in `configs/models/GWNetGMM.toml`; tune generically.

## Differences

- Independent rewrite of Secs. 3.1-3.3 (Eqs. 3-10) of arXiv 2604.16084v2; the official code has no license (`NOASSERTION`) and was read for reference only. Named after the official `GWNET_GMM` class (the paper names no model); only the Graph WaveNet pairing is cataloged.
- The learnable spacing factor follows the code (the paper calls `s` fixed); the loss keeps the `log(2 pi)` constant the official loss drops (identical gradients).
- Predefined road-graph supports and the SVD node-embedding init are not exposed; `layers` is explicit and too-long windows are rejected.
- Output type is `point` (mixture mean); CRPS, HDPS intervals, mAW and mCCE are not wired into TSFLab evaluation.

Full detail: `reference.md`.
