---
name: "positional_encoding"
description: "Factory positional_encoding(kind, learnable, length, width) returning an additive nn.Parameter position table (sincos, 1-D/2-D coordinate, or random init). Use for fixed-length token sequences such as patches; not for variable lengths, rotary or relative encodings, or per-channel positions."
---

# positional_encoding

## What it does

`positional_encoding(kind, learnable, length, width)` builds the additive
positional table for `length` token positions and `width` features and wraps it
in an `nn.Parameter` (`requires_grad=learnable`). The caller adds it to the token
tensor (`tokens + table`), broadcasting over batch.

- `sincos`: sinusoidal table with sine on even columns and cosine on odd columns,
  `sin/cos(pos * 10000^(-2i/width))`, then standardized (below).
- `lin1d`/`exp1d`/`lin2d`/`exp2d`: coordinate tables, standardized. 1-D kinds
  have width 1 and equal `2 * t^p - 1`; 2-D kinds equal `2 * t^p * f^p - 1`.
  Here `t` is a linspace over positions in [0, 1], `f` a linspace over features
  in [0, 1], and `p = 1` (lin) or `0.5` (exp). A 2-D kind with `width == 1`
  reduces to the 1-D formula.
- `zeros` (alias `small_uniform_2d`): uniform(-0.02, 0.02) of shape `[length, width]`; not all zeros.
- `zero` (alias `small_uniform`; not all zeros)/`normal`/`gauss`/`uniform`: width-1 random columns (uniform(-0.02, 0.02),
  normal(0, 0.1), normal(0, 0.1), uniform(0, 0.1)) that broadcast over features.
- `None`: uniform(-0.02, 0.02) `[length, width]` and forced non-learnable.

Standardization (`sincos` and the four coordinate kinds): subtract the table
mean and divide by `10 *` the unbiased standard deviation of the whole table, so
a non-degenerate table has mean 0 and std 0.1; if that std is exactly 0 the table
is only mean-centred.

## When to use

Use for an additive per-position table over a fixed number of tokens (patch
index) in attention or mixer models. Do not use when the sequence length varies
at run time (the table is fixed-length), for rotary or relative encodings (see
`periodic_alibi_bias` for a score bias), or when per-channel positions are
needed.

## Interface

`positional_encoding(kind: str | None, learnable: bool, length: int, width: int)
-> nn.Parameter`. The only public symbol (the catalog lists it as the public symbol). Raises
`ValueError` if `length < 1` or `width < 1`, or if `kind` is not one of `None`,
`zero`, `small_uniform`, `zeros`, `small_uniform_2d`, `normal`, `gauss`,
`uniform`, `sincos`, `lin1d`, `exp1d`, `lin2d`, `exp2d`. Returns a
default-dtype (float32) CPU parameter; the caller registers it on a module
(`self.position = ...`), so the state-dict key is whatever attribute name the
model chooses. Random kinds draw from the global torch RNG at call time. Note the
naming trap: `zero` is a random width-1 column, while `zeros` is the full
`[length, width]` random table (neither is all zeros). `learnable` is coerced
with `bool(...)`. A standardized table with a single element (`sincos` with
`length == width == 1`, or `lin1d`/`exp1d` with `length == 1`) has an undefined
unbiased std and returns `NaN`; `length == 1` with `width > 1` is fine.
Odd `width` is supported for `sincos`.
