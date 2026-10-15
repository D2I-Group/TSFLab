# periodic_alibi_bias — reference

## Origin and granularity

Extracted with PENGUIN, which calls it with `dtype` and `device` of the scores and
converts its periods from time steps to patch units before the call. Only the bias
table is shared; grouped-query projections, the causal mask, and softmax stay in the
model because they differ across attention designs. It is a pure function with no
parameters.

## Invariants and equivalence evidence

- Component checks (at extraction): hand-computed values for a single non-periodic group, the standard ALiBi slopes `2**-k` for 8 heads, the triangle wave for period 4 next to a plain group, symmetry, non-positivity, zero diagonal and the `P / 2` bound (float64 dtype honored), and `ValueError` for zero sizes, indivisible head counts and a zero period.
- No stored reference tensors; the expected tensors were written out inline. `penguin` (`PeriodicNestedGroupAttention`) consumes it, so its model checks exercise it indirectly.
- For a single non-periodic group and a power-of-two head count this is the symmetric (bidirectional) ALiBi bias.

## Variants and options

Not covered: the extra slope interpolation ALiBi uses for head counts that are not powers of two, per-head (not per-group) periods, learned slopes, or the cross-attention form with different query and key lengths.

## Related components

`self_attention_family` (attention cores that take no additive bias; a model adds this table to the scores itself), `positional_encoding` (absolute position tables added to tokens, not to scores), `dominant_periods` (discovers periods from data that could be passed here as `periods`), `graph_masked_attention` (restricts attention by an adjacency mask rather than a distance bias).
