# positional_encoding — reference

## Origin and granularity

The mode names match the position-table choices of the PatchTST code base; the
original module was rewritten as part of `fba5fa99` ("finish clean-room shared
forecasting layer"), and the history does not record numerical equivalence with
the original. It is a pure factory for tables only: where the table is added,
dropout after addition, and 3-D (per-channel) positions stay in the models.
Consumers: the `patchtst` component backbone (configurable `pe`/`learn_pe`),
`gateformer` (`"sincos"`, frozen), `semixer`, `lsinet` and `mou` (`"zeros"`,
learnable), `canet` (`"exp2d"`, learnable, width = patch size), and the composed
`temporal` slot adapter in `_slots` (`"sincos"`, frozen), which is why `composed`
declares it.

## Invariants and equivalence evidence

- Contract check (at extraction) of every kind except the aliases at
  `length=6, width=4`: result is an `nn.Parameter`, float32 on CPU, shape `[6, 1]`
  for `zero`, `normal`, `gauss`, `uniform`, `lin1d`, `exp1d` and `[6, 4]`
  otherwise, finite, `requires_grad` true exactly when `learnable` and
  `kind is not None`, and `learnable=False` always frozen.
- A second contract check covered the `ValueError` cases (unknown kind, `length=0`,
  `width=0`), that the same seed reproduces a random table, that a `sincos` table
  has mean 0 and std 0.1, and pinned `sincos` (6x5), `lin2d` (5x3) and `exp1d`
  (5x3) as regression values.
- A repository-level check included `positional_encoding` in the dependency
  closure of the `patchtst` component.
- No regression values for the random kinds, the aliases, `exp2d`/`lin1d` values,
  or the single-element NaN case (the latter is by inspection of the
  standardization), and none against the original PatchTST tables.

## Variants and options

Choose by `kind` as above; `learnable=False` freezes the table (still registered
as a parameter, so it appears in `state_dict` and `parameters()`, with
`requires_grad=False`). `kind=None` is always frozen.

## Related components

`patchtst` (configurable consumer), `tst_transformer` (encodes tokens after the
table is added), `embed` (its `PositionalEmbedding` is also a fixed sin/cos table,
but as a non-trainable `[1, max_len, d_model]` buffer inside `DataEmbedding`, with
no standardization and an even-`d_model` requirement), `periodic_alibi_bias`
(an attention bias instead of an additive token table).
- `periodic_query_bank`: phase-indexed learned table, versus a fixed position table.
