---
name: "TimeMachine"
description: "Four Mamba state-space blocks over two embedding scales of the lookback, with global/local context pairs, residual links and two projections. Use for long-term multivariate forecasting with channel-independent (or optional channel-mixing) Mamba mixing; not for covariates or probabilistic output."
---

# TimeMachine

## Idea

- `embed1` and `embed2` map the lookback to an `n1` and then an `n2` embedding; `mamba3`/`mamba4` act on the `n1` scale (outer pair) and `mamba1`/`mamba2` on the `n2` scale (inner pair), all `MambaBlock` mixers from the `mamba` component.
- With `ch_ind=True` each variate is a separate batch item; `mamba4`/`mamba1` see the embedding as a length-`n` sequence of width 1 (global context), `mamba3`/`mamba2` one token of width `n` (local context). With `ch_ind=False` the channels are the tokens.
- Residual links add the `n2` embedding before `proj1` and the `n1` embedding after it; `proj2` maps the concatenation of that result and the outer-pair output to the horizon.
- RevIN (`revin` component) normalizes and denormalizes per variate.

## When to use

- Long-term multivariate forecasting where a compact state-space model should capture both global and local temporal context at two scales.
- Channel-independent by default; `ch_ind=False` mixes channels as Mamba tokens when channels are strongly related.
- The portable Mamba recurrence is sequential, so very wide embeddings are slow on CPU; marks, covariates and probabilistic output are not supported.

## Configure

- `enc_in`: number of channels; with `ch_ind=False` it fixes the Mamba token width, so the model only accepts that channel count.

Other hyperparameters: preset defaults in `configs/models/TimeMachine.toml`; tune generically.

## Differences

- Independent implementation from the paper after reading the official code (Apache-2.0); nothing copied.
- The four mixers use the `mamba` component's portable PyTorch recurrence instead of `mamba_ssm`'s fused CUDA kernels (same selective-scan math; reference step-size initialization).
- `revin=False` keeps instance mean/std normalization without the affine, as the official fallback does.
- Preset `n1=256`, `n2=128`, `dropout=0.05` is a TSFLab default; official scripts tune `n1`/`n2`/`fc_drop` per dataset and horizon.
- The official dataflow (axis swaps, residual links, concatenation with the outer pair) is followed exactly.
- Full detail: reference.md, `## Differences in detail`.
