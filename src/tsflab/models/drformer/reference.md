# DRFormer — reference

## Implementation mapping

- `DynamicSparseLinear` (Eq. 1): group `i` starts with the last
  `ceil(active_ratio * i * P / G)` positions active.
- `DynamicSparseLinear.prune_and_regrow` (Algorithm 1): per group, deactivate
  `ceil(death_rate * active)` smallest-magnitude active weights, then activate as
  many random inactive positions of the group's region with a fresh
  `kaiming_uniform` value. `Model.advance_sparse_schedule` runs it every
  `floor(update_frequency * steps_per_epoch)` steps during the first
  `mask_epochs` epochs with the cosine-annealed rate of Eq. (3)
  (`CosineDecay` floor `0.001`, `update_frequency = 0.3` epoch, from
  `exp/exp_main.py` and `utils/sparse_learning.py`).
- `Model.multi_scale` (Eqs. 4-5); `GroupAwareRoPEAttention` (Eqs. 7-11): intra-group
  angle on the base patch grid, inter-group angle by group index.
- `Model.fuse` (Eqs. 13-15): transposed convolution of kernel and stride `K`,
  padding `ceil((K - N mod K) / 2)` (`pooling_geometry`); the shared flatten
  head maps `d_model * N` features per channel to the horizon.

## Constraints

- `d_model / n_heads` must be an even integer; `d_model` and `patch_len` must be
  divisible by `mask_groups` (the official indicator mismatches the weight shape
  otherwise); `patch_len <= seq_len`.
- The indicator schedule needs the epoch length recorded by `training_setup`.

## Citation

Ding, R., Chen, Y., Lan, Y.-T., Zhang, W. "DRFormer: Multi-Scale Transformer Utilizing Diverse Receptive Fields for Long Time-Series Forecasting." CIKM '24. doi:10.1145/3627673.3679724.
