---
name: "VisiFold"
description: "Spatio-temporal Transformer that folds each node's whole history into one token and, in training, masks, shuffles, and groups node tokens into subgraphs for attention. Use for long-horizon forecasting on large traffic-style node sets with time-of-day/day-of-week signal; not for few-channel or non-spatial data."
---

# VisiFold

## Idea

- Temporal folding: `fold_input` maps each node's full lookback to one token, concatenated with time-of-day, day-of-week, and learnable node embeddings, so spatial mixing is one attention pass over `N` node tokens instead of one per step.
- Node visibility (training only, `node_visibility`): `random_mask_tokens` drops `mask_ratio` of nodes, `shuffle_tokens` permutes the rest, and `group_into_subgraphs` splits them into `subgraph_size` groups to bound attention cost; the pipeline is inverted afterwards.
- `SelfAttentionLayer` blocks attend within groups (no adjacency; `adj_mx` is unused) and an MLP head predicts the whole horizon per node.
- Masked nodes get zero outputs in training and `training_objective` restricts the loss to the visible nodes; evaluation keeps every node.

## When to use

- Designed for long-term traffic forecasting over many sensor nodes, where per-step spatial graphs inflate compute.
- Node masking and subgraph sampling bound memory for large node counts; the paper reports keeping its advantage even at 80% masking.
- Node relations are learned by attention, not from a given adjacency.
- Uses time-of-day and day-of-week embeddings, so the data needs calendar marks.

## Configure

- `enc_in`: number of nodes `N`; `num_nodes` and `adj_mx` are injected by the runner.
- `steps_per_day`: steps per day at the dataset's sampling frequency (24 hourly, 288 for 5-minute traffic); indexes the time-of-day embedding.
- `input_dim`: value plus time-of-day plus day-of-week inputs (3).
- `subgraph_size`: group size for node attention, chosen relative to the node count (paper presets 30-50; the preset 4 is smoke scale).

Other hyperparameters: preset defaults in `configs/models/VisiFold.toml`; tune generically.

## Differences

- Attention uses `nn.MultiheadAttention` with the official post-norm order; the official `edge_emb` bias path is never used by the model and is dropped.
- In training, outputs are scattered back to all nodes (masked positions zero) to meet the full-shape contract; the official code returns only kept nodes and drops the same nodes from the target. The visible-node loss gives the same objective.
- `adj_mx` is accepted for interface uniformity and discarded, as the official model uses no fixed adjacency.
- Preset widths are smoke scale; the paper's PEMS04/PEMS08/SEATTLE presets use widths up to 64-256 and `subgraph_size` 30-50.
- A single `dropout` (default 0.0) is applied inside `SelfAttentionLayer`; the official default is also 0.
