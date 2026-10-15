---
name: "MAGE"
description: "Spatiotemporal forecaster built from a sparse top-k mixture of low-rank adaptive graph experts with linear node cost and calendar prompts. Use for node-level traffic or sensor networks with many nodes where the graph is learned; not for univariate or few-channel series."
---

# MAGE

## Idea

- `AdaptiveGraphExpert` propagates with a factorised low-rank kernel `softmax(E_target) softmax(E_source)` applied to node features, so cost is linear in nodes and no `N x N` adjacency is built.
- `MixtureGraphBlock` routes every node to its top-k experts (`topk`) by raw-logit selection, a masked softmax, and a 5% dense mix for balance, then applies an RMSNorm feed-forward.
- Each node embeds its whole window (value plus calendar channels, `to_calendar_spatiotemporal`) with a linear layer, plus a pooled calendar embedding.
- Three blocks are applied with 1, 2 and 3 recurrent passes (`recur_num` experts); the last result is subtracted from a skip, and a linear head gives the horizon.

## When to use

- Designed for spatiotemporal node forecasting (traffic, sensor networks) where the graph topology is learned end to end rather than given.
- Linear cost in the number of nodes suits large networks under tight compute budgets.
- Uses time-of-day and day-of-week marks as calendar prompts.
- Not for univariate or few, unrelated channels: the method exists to learn node interactions; the supplied adjacency is ignored.

## Configure

- `enc_in`: number of graph nodes (channels); the expert graphs are learned over them.

Other hyperparameters: preset defaults in `configs/models/MAGE.toml`; tune generically.

## Differences

- Clean-room implementation from the NeurIPS paper; the unlicensed author repository was a reference only and none of its source is included.
- Calendar prompting is reduced to node-window calendar channels plus one pooled calendar embedding.
- The training expert-count (balance) objective is omitted; MSE only.
- The supplied adjacency matrix is unused.
- The routing gate stays model-local rather than reusing `topk_expert_router` (see reference.md).
