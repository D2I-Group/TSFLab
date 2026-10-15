---
name: "mixer_block"
description: "TSMixer basic block: pre-LayerNorm residual time-mixing Linear, then pre-LayerNorm residual two-layer feature MLP. Use for all-MLP forecasters on fixed-length multivariate windows where channels inform each other; not for variable lengths, static or auxiliary features, or channel-specific time mixing."
---

# mixer_block

## What it does

`MixerBlock(seq_len, channels, hidden, dropout)` applies two residual updates to
`x [B, L, C]`:

1. Time mixing: `x = x + drop(gelu(Linear_L(LN_time(x)^T)))^T`, where `LN_time` is a LayerNorm over the joint `(L, C)` shape, and the linear acts on the time axis (shared across channels).
2. Feature mixing: `x = x + drop(Linear_out(drop(gelu(Linear_in(LN_feat(x))))))` with `Linear_in: C -> hidden`, `Linear_out: hidden -> C`.

Shape is preserved, so blocks stack.

## When to use

Use for multivariate windows of fixed length where cross-channel information
helps: the time-mixing linear is shared across channels and the feature MLP mixes
channels at every step, so stacked blocks model both without attention. Do not
use for variable-length inputs, for the auxiliary/static-feature TSMixer
extension, for BatchNorm or post-norm variants, or when the time-mixing linear
must be channel-specific; on weakly correlated channels the feature MLP adds
capacity without signal.

## Interface

`MixerBlock(seq_len: int, channels: int, hidden: int, dropout: float)`

- `seq_len` (int >= 1): fixed; must equal the input time length (LayerNorm shape and time Linear are tied to it).
- `channels` (int >= 1): must equal the input channel width.
- `hidden` (int >= 1): feature-MLP width.
- `dropout` (float in [0, 1)): one `nn.Dropout` applied after the time activation, after the feature activation, and on the feature delta.
- `forward(x [B, seq_len, channels]) -> same shape`; float tensors, no explicit validation (shape errors come from the layers).
- State-dict keys: `time_norm.weight/bias` and `feature_norm.weight/bias` (each `[seq_len, channels]`), `time_projection.weight/bias` (`[seq_len, seq_len]`), `feature_in.weight/bias`, `feature_out.weight/bias`. GELU and dropout have no parameters. Stateless apart from dropout.
