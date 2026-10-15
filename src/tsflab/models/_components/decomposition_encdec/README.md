---
name: "decomposition_encdec"
description: "Autoformer-style encoder/decoder layers: pluggable mixers and FFN, each followed by a moving-average split; the decoder accumulates projected trends. Use for trend-carrying decomposition Transformers (Autoformer, FEDformer); not for plain post-norm Transformers (transformer_encdec) or biased trend projections."
---

# decomposition_encdec

## What it does

Progressive series decomposition inside a Transformer-style encoder and decoder.
With `decomp(z) = (z - MA(z), MA(z))` (`series_decomposition`) and
`FFN(z) = Drop(W2 Drop(act(W1 z)))`:

- Encoder layer: `s = decomp(x + M(x))[0]`, `s = decomp(s + FFN(s))[0]`; trends are dropped.
- Decoder layer: `s, t1 = decomp(x + M_self(x))`, `s, t2 = decomp(s + M_cross(s, memory))`,
  `s, t3 = decomp(s + FFN(s))`, `trend' = trend + sum_i P_i t_i` with three bias-free
  `P_i: d_model -> c_out`; returns `(s, trend')`.

`M`, `M_self` and `M_cross` are any token mixers with the shapes above
(auto-correlation, Fourier blocks, attention).

## When to use

Use for decomposition Transformers in which every sub-layer is followed by a
moving-average split and the decoder carries a trend stream to `c_out`. Do not
use for post-norm Transformer layers (`transformer_encdec`), for models that keep
the encoder trends, or when the trend projection must be convolutional or biased.

## Interface

`decomposition_feed_forward(d_model: int, d_ff: int, dropout: float, activation: str) -> nn.Sequential`

- `Linear(d_model, d_ff)`, `GELU` if `activation == "gelu"` else `ReLU`, `Dropout`,
  `Linear(d_ff, d_model)`, `Dropout` (state keys `0.*` and `3.*`).

`DecompositionEncoderLayer(mixer, d_model, d_ff, moving_avg, dropout, activation, *, mixer_name="mixer")`

- `mixer`: module with `mixer(values [B, L, d_model]) -> [B, L, d_model]`,
  registered as the submodule `mixer_name` (first in registration order).
- Submodules in order: `<mixer_name>`, `decomposition_one`, `feed_forward`,
  `decomposition_two`; `moving_avg` is the odd moving-average kernel.
- `forward(values [B, L, d_model]) -> [B, L, d_model]`.

`DecompositionDecoderLayer(self_mixer, cross_mixer, d_model, d_ff, moving_avg, dropout, activation, c_out, *, mixer_names=("self_mixer", "cross_mixer"))`

- `self_mixer(seasonal) -> same shape`; `cross_mixer(seasonal, memory) -> seasonal shape`
  (the mixer handles `len_enc != len_dec`). `mixer_names` must be two distinct
  names, otherwise `ValueError`.
- Submodules in order: the two mixers, `feed_forward`, `decompositions` (3 x
  `SeriesDecomposition`), `trend_projections` (3 x `Linear(d_model, c_out, bias=False)`).
- `forward(seasonal [B, Ld, d_model], memory [B, Le, d_model], trend [B, Ld, c_out])`
  `-> (seasonal [B, Ld, d_model], trend [B, Ld, c_out])`.
- No buffers or state of its own; the mixers keep theirs.
