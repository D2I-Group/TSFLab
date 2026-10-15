---
name: "DCRNN"
description: "GRU encoder-decoder whose gates use bidirectional random-walk diffusion convolution on a directed sensor graph. Use for traffic-like node data with a known road or sensor graph; not for data without a graph or for long-horizon channel forecasting."
---

# DCRNN

## Idea

- `DiffusionConvolution` applies K-step diffusion over forward and reverse random-walk supports of the adjacency (Eq. 2).
- `DCGRUCell` replaces the affine maps in the GRU reset, update and candidate equations with that diffusion convolution.
- An encoder reads history features; a decoder starts from zeros and feeds back its own projection at each step, with no target leakage.

## When to use

- Spatio-temporal forecasting on a directed graph (road sensors, flow networks) where values diffuse along edges.
- Needs a meaningful adjacency; without one it falls back to identity supports and loses the spatial model.
- Short horizons decoded step by step; point output only.

## Configure

- `enc_in`: number of graph nodes N.
- `adj_mx`: the dataset's directed `[N, N]` adjacency (identity when absent).

Other hyperparameters: preset defaults in `configs/models/DCRNN.toml`; tune generically.

## Differences

- Clean-room implementation from the paper equations; the MIT official repository is reference only and no BasicTS code is retained.
- Raw calendar or node-structured marks are accepted as encoder input features and fitted to `input_dim`.
- The preset uses one 16-unit layer and three input channels.
- Scheduled sampling, teacher forcing, official preprocessing, masked-MAE training, and checkpoint/metric reference comparison are out of scope.
