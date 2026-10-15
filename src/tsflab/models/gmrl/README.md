---
name: "GMRL"
description: "Tensor time-series model: per-channel Gaussian-mixture Cluster Norm before gated source-spanning dilated convolutions, plus a memory-bank augmenter. Use for multi-source location data (e.g. bike and taxi demand per zone) with short fixed windows; not for long lookbacks or graph-based tasks."
---

# GMRL

## Idea

- Inputs are a tensor time series flattened source-major (`enc_in = num_sources * L`), lifted to `[B, 1, S, L, T]`; a Tensor Time Series Embedding adds `e_t + e_l + e_s` to a 1x1 value projection (Sec. 3.1).
- `GMRE` (Sec. 3.2) maps each hidden channel's `S x L x T` slice to `K` mixture priors, means and scales, takes the Bayes posterior, and Cluster-Norms every scalar with its MAP cluster's `(mu, sigma)`; `L_cluster` is Eq. 7.
- `GMRETELayer` applies `tanh(filter) * sigmoid(gate)` over `[ClusterNorm(H), H]` with valid `(S, 1, k)` convolutions at dilation `2^i` (Eq. 8), spanning all sources, with residual and skip 1x1 convs.
- `HRA` (Secs. 3.4-3.5) queries an `m = 8` record memory with the skip representation (Eq. 9) and predicts all horizons with a ReLU-conv predictor (Eq. 10).
- Training adds `aux_loss = cluster_weight * mean over layers of L_cluster` to the regression loss (Eq. 12).

## When to use

- Designed for tensor time series: several co-evolving sources per location (NYC bike/taxi inflow and outflow in the paper), where sources share structure and the convolution spans them jointly.
- The mixture Cluster Norm targets heterogeneous patterns across locations and sources by normalizing each value within its inferred cluster.
- Runs on ordinary node datasets with `num_sources = 1`; it uses no adjacency or marks, so it does not exploit a known graph.
- Not for long lookbacks: the window is fixed by the dilation stack (16 steps by default).

## Configure

- `enc_in`: `num_sources * num_locations`, flattened source-major.
- `num_sources`: co-located sources per location; must divide `enc_in`.
- `layers` (with `kernel_size`): `seq_len` must equal `1 + (kernel_size - 1)(2^layers - 1)` (16 for kernel 2, 4 layers).

Other hyperparameters: preset defaults in `configs/models/GMRL.toml`; tune generically.

## Differences

- Independent rewrite from the paper (Secs. 3-4) after reading the official code at `ef37b8a4` (no license, `NOASSERTION`); nothing copied.
- Fixes official bugs: Cluster Norm uses the cluster means (official aliasing bug), TTSE/HRA parameters are registered and trained, the training objective runs (`lam` undefined officially).
- Follows Eq. 7 for the cluster objective (averaged over layers) and Eq. 9 biases; follows the code for dilations 1, 2, 4, 8, shared GMRE projections, `exp` as standard deviation, and `proj3`.
- Input scaling and the official optimizer belong to the run configuration; no reference comparison was run because the official code cannot train as shipped.

Full detail: `reference.md`.
