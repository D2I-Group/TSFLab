---
name: "AdaMamba"
description: "Selective SSM over multi-scale patches whose state is a time-frequency grid with adaptive Fourier bases and gates. Use for long-horizon multivariate forecasting with several periodicities; not for probabilistic output or graph-structured data."
---

# AdaMamba

## Idea

- Interaction encoding: a `Conv1d` over the variable axis (kernel 3) blended with the input by a learned scalar `beta` (initialised to 1).
- Multi-scale patch embedding: per patch length (default 96, 18, 9) right replicate padding by the stride, unfold, a per-scale `Linear(P_i -> d)`, concatenated into `M` tokens per channel.
- Adaptive frequencies: `omega = clamp(omega_base + Adapter(mean of tokens), 0)` over a learnable `2 pi k / S` grid.
- Frequency-gated SSM: a `[d, S]` real/imaginary state updated by a linear scan with an outer-product forgetting gate and a gated input modulated by `cos`/`sin(omega m / M)`; amplitudes are projected per frequency and summed under a sigmoid output gate.
- Residual layers with dropout, a flatten-linear head to the horizon, all inside affine `revin`.

## When to use

- Long-term multivariate forecasting where the series mixes several periodic components that a fixed Fourier basis would miss (adaptive frequency grid, multi-scale patches).
- Channels are encoded independently after a local interaction convolution over neighbouring variables only; it does not model a graph or long-range channel dependence.
- Needs a lookback at least as long as the largest patch length (96 by default).
- Point forecasts only (MSE).

## Configure

- `enc_in`: number of channels in the dataset.
- `patch_lens`: every length must lie in `[1, seq_len]`; shrink the 96/18/9 default for shorter lookbacks.
- No period-specific parameter: frequencies are learned and adapted per input.

Other hyperparameters: preset defaults in `configs/models/AdaMamba.toml`; tune generically.

## Differences

Independent rewrite following the pinned official code where it departs from the paper:

- Gates (Eqs. 15, 18, 19) use the previous input token `u_{m-1}` instead of the recurrent output `z_{m-1}`, so all gates are computed before the scan.
- Input enters the state through a gated drive per `(d, S)` cell, not a scalar `B_m u_m`; the output has no `D^u`/`D^y` residual terms and uses `tanh` of a per-frequency amplitude projection.
- The trigonometric argument is the normalised step `m / M`; `omega_base` is trained and `omega` clamped at 0.
- The interaction weight `beta = 1 - alpha` weights the input and starts at 1.
- The official parallel scan is replaced by an exact sequential scan; dead official modules are omitted; the effective (buggy) official initialisation is reproduced.
- Only one official script is released (ETTm1, horizon 96); defaults come from it and the parser.

Full detail in `reference.md`.
