# self_attention_family — reference

## Origin and granularity

Present since the initial commit and consolidated in the component move
(`33ea2050`, "colocate shared components under models"). The cut keeps the
head-split core separate from the projection wrapper so encoder and decoder
blocks (`transformer_encdec`) can swap cores (full vs ProbSparse) without
changing anything else; `informer` and `transformer` build exactly this way, while
`dualformer` and `gpht` reuse `AttentionLayer` (`dualformer` also with its own
frequency-domain core).
Per-model choices stay model-local: which core to use per role, mask objects,
`factor`, and distilling. Which paper each non-Informer core was copied from is
not recorded in history; the identification above is by class behaviour only.

## Interface details

- Core arguments: `FullAttention` uses `scale`, `attention_dropout` and
  `output_attention`; `ProbAttention` the same except that its softmax is not
  dropped out. `FlashAttention` accepts but ignores `mask_flag`, `factor`,
  `scale`, `attention_dropout` and `output_attention` (it always scales by
  `1/sqrt(d_head)`, applies no dropout and returns `None` for the map).
  `FlowAttention` stores a dropout module that is never applied.
- `ReformerLayer`: `attention`, `d_keys`, `d_values` are unused;
  `forward(queries, keys, values, attn_mask, tau, delta)` requires `tau` and
  `delta` positionally and returns `(out [B, L, d_model], None)`. It raises
  `ImportError` at construction if `reformer_pytorch` is missing
  (`LSHSelfAttention` is then `None`).
- `AttentionLayer`: `d_model` should be divisible by `n_heads` when the default
  widths are used; projections map `d_model -> d_keys * n_heads` (queries, keys)
  and `d_values * n_heads` (values).
- Masks: `FullAttention` with `mask_flag=False` ignores the mask; `FlashAttention`
  never applies a causal mask. `tau`/`delta` exist for de-stationary attention
  consumers.
- Attention maps: `output_attention=True` returns `[B, H, L_q, L_k]`
  (`ProbAttention`: `[B, H, L_v, L_v]`, with uniform `1/L_v` rows for unselected
  queries); `FlowAttention` and `FlashAttention` always return `None`.
- Randomness: `ProbAttention` samples with unseeded `torch.randint` on CPU;
  `FullAttention` dropout is the only other randomness.
- Layout: `ProbAttention.forward` transposes its context back to
  `[B, len_q, heads, d_head]` like the other cores and the original Informer code
  (zhouhaoyi/Informer2020), so `AttentionLayer`'s `view(B, L, -1)` concatenates
  heads per time step. The pinned Time-Series-Library revision keeps the
  un-transposed `[B, heads, len_q, d_head]` context (a head/time reinterpretation
  before `out_projection`); TSFLab deliberately follows the original Informer. The
  sampled-score squeeze is the explicit `squeeze(-2)`, so batch or head count 1 is
  safe. `_get_initial_context` still asserts `len_q == len_k` when `mask_flag=True`.

## Invariants and equivalence evidence

- Contract checks (at extraction): shape, state-dict key, invariant, gradient-flow and seeded numerical-regression checks for every public symbol; seeded reference values for the flash, flow, full, layer and prob variants were pinned as regression values.
- Model-level checks (pre-consolidation suite): `FullAttention` matches the
  scaled dot-product equation; `transformer` uses `FullAttention` in the encoder
  and decoder roles (decoder self-attention causal, cross-attention not);
  `informer` uses `ProbAttention` with distilling.
- A `dualformer` test checked `AttentionLayer` wrapping
  `FullAttention` alongside a model-local core.
- The same contract checks covered `FullAttention` against the explicit masked softmax
  and the causal default, `AttentionLayer` state-dict keys and gradients, `FlowAttention`
  finiteness and gradients, `FlashAttention` equal to `FullAttention` unmasked and
  invariant to values at padded keys, and `ProbAttention` shapes (including the
  time-first layout and batch/head count 1), row sums, the `len_q == len_k` assertion, and seeded
  reproducibility. `ReformerLayer` is only checked for its optional-dependency behavior.
  All of these passed in the full suite run of 2026-10-03 before the test suite was
  consolidated.
- no fixture: no pre-refactor fixture exists, and `FlowAttention`, `FlashAttention` and
  `ReformerLayer` have no recorded upstream reference.

## Variants and options

Choose the core by role: `FullAttention` (exact, `O(L^2)`), `ProbAttention`
(sub-quadratic, stochastic, needs `factor`), `FlowAttention` (linear in length,
no map), `FlashAttention` (exact, key-padding mask only, slow Python loops),
`ReformerLayer` (optional dependency; ignores keys/values and `attn_mask`,
pads length to a multiple of `2 * bucket_size`).

## Related components

`transformer_encdec` (blocks that consume `AttentionLayer`), `tst_transformer`
(patch-token encoder with fused PyTorch attention), `differential_attention`,
`topk_expert_attention`, `global_patch_compression_attention`, `graph_masked_attention`
(adjacency-restricted attention) and `periodic_alibi_bias` (additive score bias; these
cores have no bias hook, so a model that needs one writes its own attention as `penguin`
does) differ in the attention rule they implement; `masking` supplies the mask objects.
