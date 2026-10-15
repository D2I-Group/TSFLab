---
name: "periodic_alibi_bias"
description: "Additive attention-score bias [heads, N, N]: ALiBi slopes times |i - j| or its triangle-wave periodic distance, one period per head group. Use for attention over equally spaced tokens of series with known periods (PENGUIN); not for irregular timestamps, unequal query/key lengths, or learned slopes."
---

# periodic_alibi_bias

## What it does

`periodic_alibi_bias(seq_len, num_heads, periods)` builds the relative-position
bias added to pre-softmax attention scores. For a head with slope `m_k` the
bias is `-m_k * d(i, j)`. Plain ALiBi uses `d = |i - j|`; the periodic form
replaces it with the triangle wave of `|i - j| mod P`, so tokens a whole period
apart are as close as neighbours. Heads are grouped, each group owning one period
(or `None` for the plain distance).

## When to use

Use for self-attention over equally spaced tokens of a series with one or
several known periods: each head group can favour tokens one period apart, and a
plain group keeps the usual recency bias. Do not use for irregular timestamps,
for cross-attention with unequal query and key lengths, when the periods are
unknown (find them first, for example with `dominant_periods`), or when slopes
should be learned.

## Interface

`periodic_alibi_bias(seq_len, num_heads, periods=None, *, device=None, dtype=torch.float32) -> Tensor`

- `seq_len`, `num_heads` (int >= 1, else `ValueError`). `periods`: sequence of `int >= 1` or `None`, one entry per head group, in head order (group `g` owns heads `g * n .. (g + 1) * n - 1`); `None` or empty means one non-periodic group. `num_heads` must be divisible by the number of groups, and a period below 1 raises `ValueError`; integer-ness is not checked.
- Slopes inside a group of `n` heads are `2 ** (-8 k / n)`, `k = 1..n`, identical across groups.
- `device` and `dtype` (a floating dtype; default float32) set where and how the table is built. Returns `[num_heads, seq_len, seq_len]`, non-positive, symmetric in `(i, j)`, zero on the diagonal. Period distances never exceed `P / 2`.
- It is the bidirectional form `-m * |i - j|`: no causal masking and no `-inf` entries, so callers that need causality add their own mask.
- No state, no parameters; deterministic.
