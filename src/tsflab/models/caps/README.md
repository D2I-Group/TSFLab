---
name: "CAPS"
description: "RoPE linear attention scored by three additive Clock-weighted paths: Riemann softmax, prefix-product decay and baseline. Use for compact, linear-cost long-term forecasting on standard multivariate benchmarks; not for calendar-driven or probabilistic tasks."
---

# CAPS

## Idea

- Clock (Eq. 7): `Delta_t = softplus(W_c h_t) + 1e-6`, one positive weight per head and step, shared by three paths.
- Path queries and keys (Eq. 16) from RoPE-rotated `q`, `k` with learned frequencies: Path 1 a Clock-weighted Riemann softmax (Eqs. 9-11), Path 2 a prefix-product decay (Eqs. 12-14), Path 3 a Clock baseline (Eq. 15).
- Causal linear attention (Eq. 17) with identity feature map, so the three path weights add without a global softmax denominator.
- Each channel is centered on its last value and extended by a learned `Linear(L -> H)` fill (Eq. 18); tokens `[W_c x_t ; x_{c,t} V_c]` plus position and channel tables run through shared pre-RMSNorm blocks.
- The value slice of each horizon token is decoded with the transposed value embedding and the last value is restored.

## When to use

- Long-term forecasting with a small parameter budget (the scripted ETTm1 setting has about 59K parameters) and attention cost linear in `seq_len + pred_len`.
- Recency-weighted dependence: the decay path favors recent steps while the softmax path keeps global alignment.
- Calendar marks are ignored; point output only.

## Configure

- `enc_in`: number of channels.
- `d_model`, `value_dim`: the official scripts derive both from the channel count (`4 * nearest_power_of_two(sqrt(enc_in + 4))`, 16 for ETT); `d_model + value_dim` must split into `n_heads` heads of even width.

Other hyperparameters: preset defaults in `configs/models/CAPS.toml`; tune generically.

## Differences

Independent rewrite after reading the pinned official code (no license file).

- RoPE is a true pair rotation with learned frequencies; the official interleaved/concatenated layout is not a rotation.
- No calendar channels (the official scripts append four).
- Only the linear variant; the official softmax ablation is not causal.
- Explicit `d_model`/`value_dim` defaults (16 + 16) instead of the derived widths; the paper's 64 + 64 and parameter counts do not match its scripts.
- The cumulative log-decay keeps the official -50 floor as a numerical guard.

Full detail in `reference.md`.
