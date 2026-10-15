# quantile_head — reference

## Origin and granularity

Introduced with the first probabilistic-forecasting batch (commit `b9946b39`, PR
#20, originally the file `_quantile_head.py` under the then-flat models directory) for TiRex-, QuantileDLinear-,
QuantilePatchTST-, and MQRNN-style models, then moved into `_components`
(`33ea2050`) and extended with level validation (`db5b2970`). The construction is
a repository design choice; no paper is recorded for the anchor-plus-gaps scheme,
and the models using it differ from their published quantile heads accordingly. The pinball
loss, quantile levels from `config.evaluation.quantile_levels`, and the
trunk producing `base` stay model-local.

## Invariants and equivalence evidence

- Contract checks (at extraction): the `None` default and the rejected level lists;
  for levels with Q = 5, 3, 1, 2, shape and dtype, ascending order,
  `output[..., median_idx] == anchor_proj(base)`, the median index being the level
  closest to 0.5, and `_levels` absent from the state dict; input and parameter
  gradients, the four state-dict keys, and seeded values pinned as a regression value.
- A repository-level check confirmed the `[2, 8, 3, 3]` output, monotonicity,
  gradient finiteness, the default median level, and rejected level lists.
- An `mqrnn` model-level check (pre-consolidation suite) confirmed that `mqrnn`
  equals `quantile_head(local_decoder(...))`, ascending quantiles, and its
  `[2, 3, 2, 9]` output.
- The tie rule for even level lists was not covered by a check.

## Variants and options

The level set is configurable (any strictly increasing set in (0, 1)); odd
`Q` with 0.5 included makes the anchor the exact median. Not covered: crossing
heads that fit each level independently, heteroscedastic Gaussian heads (see
`gaussian_parameter_head`), or learned level embeddings.

## Related components

`gaussian_parameter_head` (parametric Gaussian `loc`/`scale` output scored by NLL; this
head is distribution-free, emits a non-crossing quantile grid, and is scored by pinball
loss), `dlinear` and
`patchtst` (backbones feeding the anchor in the quantile consumers),
`flatten_forecast_head` (point-forecast head). `composed` names this head in its
`head` literal but rejects it (point output only).
