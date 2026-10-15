---
name: "freq_band_moe"
description: "Instance-normalize a [B, C, T] series, split its rfft into contiguous bands at sorted sigmoid boundaries, mix bands with input-conditioned softmax gates, restore scale. Use for a frequency-band gating front-end on fixed-length windows; not for variable sequence length or when a shared revin should own normalization."
---

# freq_band_moe

## What it does

`FrequencyBandMixtureOfExperts(expert_num, seq_len)` denoises a series by a
gated recombination of frequency bands. For `x [B, C, T]`:

1. Instance-normalize over time: `mu = mean_t x`, `v = var_t(x - mu) + 1e-5`
   (unbiased variance), `z = (x - mu) / sqrt(v)`.
2. `F = rfft(z)` (`T//2 + 1` bins).
3. Boundaries: `sort(sigmoid(band_boundaries))`, padded with 0 and 1, scaled by
   the bin count and truncated to integers (floor); band `e` is the half-open bin range
   `[idx[e], idx[e+1])` (the last index is forced to the bin count). With
   `expert_num == 1` the raw (empty) buffer is used and the boundaries are `[0, 1]`.
4. Gate: `g = softmax(MLP(mean_c |F|))` with an MLP
   `Linear(F, F) -> ReLU -> Linear(F, expert_num)`, so `g` is `[B, expert_num]`.
5. `out = irfft(sum_e g[b, e] * (F masked to band e), n=T) * sqrt(v) + mu`.

## When to use

Use as a front-end that gates frequency bands of a `[B, C, T]` window with a
fixed window length. Pass `learnable_boundaries=True` when the boundaries should
be learned. Do not use when the sequence length varies, or when the normalization
should be handled by a shared `revin` (this module normalizes internally and
restores scale itself). Note it expects channels on axis 1 (`[B, C, T]`), unlike
the `[B, T, C]` layout of most components, so permute first (as `freqmoe` does).

## Interface

`FrequencyBandMixtureOfExperts(expert_num: int, seq_len: int, learnable_boundaries: bool = False, boundary_temperature: float = 1.0)`

- `expert_num` (int >= 1): number of bands/experts. `seq_len` (int >= 1): exact
  time length accepted; `freq_len = seq_len // 2 + 1`. Violations raise
  `ValueError`.
- `learnable_boundaries` (default `False`): fixed boundaries, as in the official
  code. `True`: boundaries train (see Variants). `boundary_temperature` (> 0): soft
  edge width in bins for the straight-through gradient, used only when learnable.
- State-dict keys: `band_boundaries` (shape `[max(expert_num - 1, 0)]`, init
  `torch.rand`; a registered buffer by default, an `nn.Parameter` when
  `learnable_boundaries=True`; the key is the same in both, so checkpoints saved
  from the former Parameter, or from either mode, load into either mode),
  `gating_network.0.weight/bias` (`[F, F]`, `[F]`), `gating_network.2.weight/bias`
  (`[expert_num, F]`, `[expert_num]`).
- `forward(x)`: `x` floating `[batch, channels, seq_len]`; wrong rank or last
  length raises `ValueError`. Returns `(combined, boundaries, gating_scores)`:
  `combined` same shape as `x`; `boundaries` shape `[expert_num + 1]` in
  `[0, 1]` (0 first, 1 last); `gating_scores` `[batch, expert_num]`, rows sum to 1.
- Stateless between calls (no cached tensors in the module). No dropout. Input
  must be a real floating tensor (`rfft`); the gate and masks follow its dtype and
  device. `torch.var` is the unbiased estimate, so `seq_len == 1` yields NaN.
