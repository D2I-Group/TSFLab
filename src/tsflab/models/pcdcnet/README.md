---
name: "PCDCNet"
description: "Physics-inspired air-quality surrogate: per-station local MLP dynamics, graph-Laplacian transport, and GRU accumulation, rolled out hour by hour with calendar covariates. Use for station-level pollutant forecasting on a spatial graph; not for data without a meaningful graph or for probabilistic output."
---

# PCDCNet

## Idea

- `LocalInteractionDynamics` (RMSNorm and a SiLU MLP) models local sources and sinks at each station.
- `SpatialTransportDynamics` propagates the hidden state with a symmetric-normalized graph Laplacian from `adj_mx`; only its zero-mean (conservative) flux enters the forecast update.
- A `nn.GRUCell` accumulates state over the history, then the model rolls out `pred_len` steps, adding an increment to the previous value each step.
- Known-future time covariates (from `x_mark_dec`) feed every step; `domain_informed_constraint` exposes a transport penalty that the trainer does not call.

## When to use

- Station-level air-quality or similar transport-dominated fields (formation, transport, dissipation) where neighbouring nodes exchange mass along a known graph.
- Tasks where calendar covariates carry signal (diurnal and weekly emission cycles); they enter every step.
- Not without a meaningful adjacency: with no `adj_mx` the Laplacian is built from the identity and the transport carries no spatial information. The step-wise rollout makes long horizons slow; point output only.

## Configure

- `enc_in`: number of stations (nodes); must equal the dataset's node count and the adjacency size.
- `adj_mx`: the dataset's `(nodes, nodes)` adjacency, supplied by graph datasets; it is symmetric-normalized into a Laplacian. Omitted, the identity is used.

Other hyperparameters: preset defaults in `configs/models/PCDCNet.toml`; tune generically.

## Differences

- Independent derivation from the paper equations; the author implementation (`src/models/pcdcnet.py` in CauAir, no license at the pinned revision) was not copied. The LID/STD/TAD update order is kept.
- Transport: one projection of the normalized-Laplacian message instead of the official two-layer residual GELU graph convolution with DropEdge.
- The official feature-mixing MLP is 4x wide (local 2x); the local update also adds a zero-mean transport flux to the increment.
- The official spatial/temporal regularizer (weight 10) is exposed but not called by the trainer.
- Calendar covariates are a reduced substitute for the paper's meteorology and emissions inputs. Details in reference.md.
