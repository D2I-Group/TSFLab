---
name: "TimeFilter"
description: "Graph forecaster over (channel, patch) nodes: a dense spatial-temporal affinity is filtered per node to its top-p strongest links and passed through a mixture of graph experts. Use for multivariate data with time-varying, selective cross-channel dependencies; not for very many channels."
---

# TimeFilter

## Idea

- `PatchGraphBuilder` turns every (channel, patch) into a node and builds a dense query/key affinity over all nodes, mixing channel and time dependencies.
- `PatchSpecificGraphFilter` keeps the top-p fraction of each node's affinities as a sparse adjacency and routes each node to `num_experts` `RegionExpert` message-passing experts with a softmax router.
- The affinity is recomputed from node features between filter layers; a flatten linear head maps each channel's patches to the horizon under `revin`.
- `training_objective` adds the router load-balancing `last_moe_loss` with weight 0.05 to the configured criterion; validation and test use the standard loss only.

## When to use

- Multivariate data where channel-independent models miss useful covariate relations but full channel mixing adds noise; dependencies are filtered per patch, so they can change over time.
- Moderate graph sizes: the affinity is dense over `enc_in * ceil(seq_len / patch_len)` nodes, so cost grows quadratically with channels and patches.
- Not for many-channel datasets without raising `patch_len` or reducing channels, or for probabilistic output.

## Configure

- `enc_in`: number of channels; nodes = `enc_in * ceil(seq_len / patch_len)`.
- `patch_len`: ideally divides `seq_len` (otherwise the lookback is zero-padded at the end); larger patches shrink the node graph.

Other hyperparameters: preset defaults in `configs/models/TimeFilter.toml`; tune generically.

## Differences

- Clean-room implementation from the paper's channel-patch graph, patch-specific top-p filtration, and differentiable region-expert router; the unlicensed reference repository was not copied.
- The balance weight 0.05 is the `alpha` of the official training loop.
