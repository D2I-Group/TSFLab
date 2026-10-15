---
name: "GRAMODE"
description: "Spatio-temporal graph model: road-graph and DTW-graph streams of TCN-wrapped global, local and edge graph-ODE blocks with gated fusion and a node-attention head. Use for sensor-network traffic forecasting with a road graph; not for data without a graph or for long horizons."
---

# GRAMODE

## Idea

- Two graphs: the dataset's road adjacency, and a DTW graph built from the training split before the first epoch (per-node mean day profiles, exact pairwise DTW, the official z-score / Gaussian-kernel / threshold transform). Both are normalized as `alpha/2 (I + D^-1/2 A D^-1/2)` (Eq. 3).
- `MultiODEBlock` integrates three graph ODEs with unit Euler steps (Eqs. 9-17): a global `NodeODE` over the whole window, an `EdgeODE` over pairwise features `mean(H_i) + mean(H_j)`, and `K = 4` local `NodeODE`s that start from node-attention tokens and unroll `T / K` steps each.
- Local messages are clipped to `[GM - e, GM + e]` with a learnable `e` (Eq. 18); messages are fused as `1/6 sum_{m != n} p_m * sigmoid(p_n)` (Eq. 19) and a gated residual with ReLU updates the input (Eq. 20).
- `GRAMODELayer` is TCN (causal dilations 1, 2, 4), block, TCN, `BatchNorm2d` over nodes; three channels of two layers per graph feed a 12-head attention across nodes that maps each node's flattened outputs to the horizon (Sec. 4.3).
- Trains with the Huber loss (Eq. 24, `delta = 1`).

## When to use

- Designed for spatio-temporal traffic forecasting on sensor networks (PEMS, 5-minute data, 12 -> 12 steps), where both road proximity and similar daily patterns between distant sensors matter.
- Needs a road adjacency; without it only the DTW graph carries spatial information.
- The DTW graph uses day profiles, so the data should have a daily cycle at a known sampling rate.
- Not for long horizons or long windows: cost grows with node pairs (edge ODE) and the preset is 12 steps.

## Configure

- `enc_in`: the dataset's node count.
- `adj_mx`: the road graph, injected by the runner.
- `local_chunks`: `seq_len` must be a multiple of it (default 4).
- `steps_per_day`: samples per day for the DTW day profiles (288 for 5-minute data).

Other hyperparameters: preset defaults in `configs/models/GRAMODE.toml`; tune generically.

## Differences

- Independent rewrite of arXiv 2305.18687 (TMLR) after reading the pinned official code (no license file, `NOASSERTION`); `torchdiffeq` is not needed because the official solver is fixed-grid Euler with unit steps.
- Window length, horizon and widths are parameters (officially hard-coded 12 steps, 64 channels).
- Two code bugs fixed: the TCN residual and the order of the local segments (Eq. 11).
- Leakage removed: the DTW graph uses exact DTW on the training split only; scaling is the runner's training-split scaling.
- Only the value channel by default; `input_dim > 1` appends dataset covariates from the marks. The dataset's adjacency is used as the road graph. Catalog optimizer and schedule.

Full detail: `reference.md`.
