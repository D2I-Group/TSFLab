---
name: "StateFlow"
description: "Archived dual-state recurrent reproduction with encoder pretraining and a frozen-encoder horizon decoder. Use for controlled two-stage workflow tests; official implementation parity is not established."
---

# StateFlow

## Idea

- A hidden state predicts the next observation; its residual updates a memory state.
- Width-five, stride-two chunks summarize both states for the horizon decoder.
- Train the encoder on shifted normalized history, freeze it, then train the decoder.

## When to use

- Controlled multivariate forecasting and two-stage execution experiments.
- Channels are modeled independently with shared parameters; this is not channel attention.

## Configure

- `enc_in`: input channels; lookback must be at least six.
- `encoder_epochs` and `encoder_lr`: explicit pretraining budget; decoder uses ordinary training settings.

## Differences

- Default hidden/memory widths 32/16 and chunk width/stride 5/2 preserve the archived reproduction.
- Search one factor among `hidden_size` (16/32/64), `memory_size` (8/16/32),
  `chunk_width` (3/5/7), `chunk_stride` (1/2/4); keep instance normalization.
- Nondefault structural settings are StateFlow-derived variants, not the archived reproduction.
- The adapter maps `[B,L,C]` to independent univariate windows and uses the runner pretraining stage.
- Each sweep cell pretrains its own encoder; original cross-horizon encoder reuse is not implemented.
- Runner validation checkpoint selection differs from the historical fixed-final-epoch loop.
- Numerical parity to that archived core is tested; official-code or full paper fidelity is not claimed.
