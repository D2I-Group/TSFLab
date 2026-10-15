# VisiFold — reference

## Paper

VisiFold: Long-Term Traffic Forecasting via Temporal Folding Graph and Node Visibility (ICDE 2026, arXiv 2603.11816).

Long-horizon traffic forecasting escalates compute and spatio-temporal complexity; spatio-temporal graphs that treat time and space separately suffer from snapshot-stacking inflation and cross-step fragmentation. VisiFold consolidates a sequence of temporal snapshots into one temporal folding graph and adds node visibility (node-level masking and subgraph sampling) to break the bottleneck of large node counts. It reduces resource use, outperforms baselines in long-term forecasting, and keeps its advantage at an 80% mask ratio.

```bibtex
@article{zhang2026visifold,
  title   = {VisiFold: Long-Term Traffic Forecasting via Temporal Folding Graph and Node Visibility},
  author  = {Zhang, Zhiwei and Du, Xinyi and Wang, Weihao and Guo, Xuanchi and Han, Wenjuan},
  journal = {arXiv preprint arXiv:2603.11816},
  year    = {2026}
}
```

## Implementation mapping

Local rewrite after reading the paper and the pinned official code (`PlanckChang/VisiFold` at `cc035f7e`: `model/VisiFold.py`, `model/config.yaml`, `README.md`).

- Temporal folding graph: `history[..., 0].transpose(1, 2)` reshapes `(B, T, N)` values to `(B, N, T)`; `fold_input` (`Linear(in_steps, D) -> ReLU -> Linear(D, D)`) matches the official `s_input` block exactly. Time-of-day, day-of-week, and adaptive node embeddings are concatenated as in the official `forward`.
- Node visibility (training only, the official `if self.training:` branch): `random_mask_tokens` keeps a sorted random `(1 - mask_ratio)` subset (official `random_mask_token`), `shuffle_tokens` applies a per-sample permutation (`random_shuffle`), `group_into_subgraphs` reshapes survivors into zero-padded `subgraph_size` groups (`split_group`); `ungroup_subgraphs` and `unshuffle_tokens` invert it (`recover_group`, `unshuffle`).

## Differences in detail

- Attention: the official `MultiHeadAttention` supports an additive `edge_emb` bias, but VisiFold's `forward` never passes it to `attn_layers_s`, so the path is dead. The local `SelfAttentionLayer` uses batch-first `nn.MultiheadAttention`, keeping post-norm order (attend, add, norm, feed-forward, add, norm).
- Masked-node outputs: `full.scatter_` on `keep_indices` returns `(B, N, pred_len)` regardless of `mask_ratio`. The official code returns only kept nodes and the training script drops the same nodes from `y` (`random_token_dropout_y`). The model records `last_keep_indices`, and `ModelSpec.training_objective` computes the criterion on those nodes only, so zero-filled positions never act as targets.
- `num_nodes`/`adj_mx` are runner-injected, matching the STID/STAEformer convention.
- Preset: `input_embedding_dim = 8`, `feed_forward_dim = 16`, `num_heads = 2`, `num_layers = 1`, `subgraph_size = 4` for CPU contract checks, versus the official presets in `model/config.yaml`.
