# patchtst — reference

## Origin and granularity

Implements PatchTST (Nie et al., ICLR 2023); `origin_models` lists `patchtst`
as the source design although that model no longer imports this component. It was a shared component from the
initial refactors and was rewritten as a clean-room composition in `fba5fa99`
("finish clean-room shared forecasting layer"); the flatten head was split out
in `d38451c3`. The cut keeps the composition (normalize, patch, embed, encode,
head) separate from the pieces that are reusable alone: `revin`,
`positional_encoding`, `tst_transformer`, `flatten_forecast_head`. In the
repository the backbone is consumed only by `quantile_patchtst`; the
`patchtst` model package keeps its own model-local implementation (it imports
only `revin`), so this component is not what the point-forecast `patchtst`
model runs. Quantile heads, decomposition and masked pretraining stay outside.

## Invariants and equivalence evidence

- Contract checks (at extraction), over end-padding/shared head and no
  padding/individual head: output shape and dtype, the state-dict prefixes above,
  that extra positional arguments are ignored, finite input gradients and parameter
  gradients; seeded reference outputs for both configurations were pinned as
  regression values. A second check covered channel-permutation equivariance with
  `revin=False` and the `ValueError` cases (wrong length, `head_type`,
  `padding_patch`, `patch_len > context_window`).
- A repository-level check pinned the dependency closure of `patchtst` to
  `flatten_forecast_head`, `patchtst`, `positional_encoding`, `revin`,
  `tst_transformer`.
- A model-level check (pre-consolidation suite) ran it through `QuantilePatchTST`
  (shape, quantile contracts).
- Model-level checks of patch overlap and channel independence ran on the
  model-local `PatchTST` (channel permutation equivariance), not on this backbone
  directly; the backbone shares the same channel-folding structure.
- No numeric reference compares this backbone with the original PatchTST code or
  with the model-local `patchtst` model; the only regression values are seeded
  self-references.

## Variants and options

`padding_patch="end"` (original PatchTST option) adds one extra patch;
`individual=True` gives one linear head per channel; `revin=False` /
`subtract_last=True` / `affine` for normalization; `pre_norm`, `pe`, `learn_pe`,
`norm` for the encoder. Decomposition (`DLinear`-style) and `head_type` other
than `"flatten"` are not supported.

## Related components

- `revin`, `positional_encoding`, `tst_transformer`, `flatten_forecast_head`: the
  pieces this composition wires together; use them directly for variants.
- `quantile_head`: probabilistic output layer used with it in `quantile_patchtst`;
  the backbone itself is point-forecast only.
- `dlinear`, `channel_wise_linear`: linear channel-independent point backbones
  with no patching or attention; `global_patch_compression_attention`: patch
  attention that mixes channels, unlike this channel-independent encoder.
- `mixer_block`: an MLP-mixer alternative to patch attention (`patchtst`'s
  encoder is a TST attention stack).
