---
name: "energy_frequency_pooling"
description: "Per-frequency softmax-of-energy pick of one token's complex spectral value, broadcast to all tokens (sampled in training, argmax in eval). Use for a cross-channel key-frequency spectrum on [B, N, F] complex input; not for a differentiable weighted average, identical train/eval behaviour, or real input."
---

# energy_frequency_pooling

## What it does

`EnergyBasedFrequencyPooling()` takes a complex spectrum `S[b, n, f]` with a
token axis `n` (channels or variables) and for every `(b, f)` selects one token
`n*` and returns `S[b, n*, f]` copied to all token positions:

- `energy = |S|^2`, `p = softmax(energy, dim=tokens)`.
- training: `n* ~ Categorical(p[b, :, f])` (`torch.multinomial`).
- eval: `n* = argmax_n p[b, n, f]`.

The output is a single shared "key-frequency" spectrum that tokens can be fused
with. The softmax is over raw (unnormalized) energies, so it is peaked whenever
energies differ by more than a few units.

## When to use

Use to build a cross-token shared spectrum where each frequency is taken from
its strongest token, on `[B, N, F]` complex input. Do not use when a
differentiable weighted average over tokens is wanted (this selects, it does not
average), when eval must equal the training expectation (eval is argmax, train
is a sample), or when the input is real-valued.

## Interface

`EnergyBasedFrequencyPooling()` takes no arguments.

- `forward(spectrum)`: `spectrum` must be a complex tensor of shape
  `[batch, tokens, freq]`; otherwise `ValueError` (non-complex, or `ndim != 3`).
  Returns a complex tensor of the same shape and device (an expanded view of a
  gathered `[batch, 1, freq]` tensor, so it is not contiguous in memory).
- No parameters, buffers, or state-dict keys. Behaviour depends on
  `self.training`; the training path consumes the global RNG
  (`torch.multinomial`), so seeds affect it.
- Gradient flows to the selected complex values through `gather`; the selection
  itself is not differentiated (the energy softmax receives no gradient).
