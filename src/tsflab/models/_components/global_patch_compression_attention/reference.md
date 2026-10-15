# global_patch_compression_attention — reference

## Origin and granularity

Added with Sensorformer, whose paper describes the Sensor Attention Block; the only consumer is `sensorformer`,
which stacks `layers` copies. The cut is the block alone: the module docstring
frames it as a paper-neutral "compress tokens, then attend through the
compressed set" primitive. Patching, embedding and the flatten forecast head
stay in the model. The choice of the last patch as the compression query is
inherited from the model description and is hard-coded here.

## Invariants and equivalence evidence

- Contract check (at extraction): state-dict key prefixes, output
  shape and dtype, per-token zero mean after the final LayerNorm, finite input
  gradient and parameter gradients, `ValueError` for 3-D input and for
  `d_model` not divisible by `n_heads`, and that perturbing a non-last patch changes the
  output. Seeded output values were pinned as a regression value.
- A second check confirmed output shape equals input shape and that perturbing
  non-last patches changes the output (information flows through the shared
  summaries). These passed in the full suite run of 2026-10-03 before the test
  suite was consolidated.
- No comparison against the official Sensorformer code exists, so equivalence with
  the paper's block is by structure only; `sensorformer` was also exercised by
  model runtime checks (pre-consolidation suite).

## Variants and options

Only `d_model`, `n_heads`, `d_ff`, `dropout`. A different compression query
(learned or mean-pooled summary) or multiple summaries per group are not options.

## Related components

`patchtst` and `tst_transformer` (channel-independent encoders without
cross-variable mixing), `topk_expert_attention` (routed alternative),
`flatten_forecast_head` (the head `sensorformer` pairs with this block),
`self_attention_family` (`FullAttention`: the quadratic full attention over all patches that this block replaces; it also offers other efficient variants that do not use a compression bottleneck).
