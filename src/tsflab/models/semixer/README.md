---
name: "SEMixer"
description: "Lightweight channel-independent multiscale patch MLP-Mixer chain with a Random Attention Mechanism (Bernoulli patch-link masks, averaged at evaluation). Use for long-horizon forecasting with multiscale patterns and noise under a small budget; not for cross-channel dependencies or probabilistic output."
---

# SEMixer

## Idea

- Multiscale patterns matter for long-term forecasting, but noise and semantic gaps between non-adjacent scales make them hard to integrate; SEMixer mixes scales progressively and only pairwise.
- `ScaleEmbedding` patchifies the normalized history at each scale (`scale_factors`, default 1x2x4x8) with patch length and stride multiplied by the scale, plus a learnable position table (`positional_encoding`).
- `TemporalMixingBlock._random_attention` (RAM) replaces learned attention with a Bernoulli patch-to-patch mask in training and its closed-form `(1 - connection_probability)` average at evaluation; inter- and intra-patch MLPs follow.
- The progressive chain (MPMC) mixes the finest scale, then repeatedly concatenates the previous mixed tokens with the next scale's raw tokens and keeps the new-scale suffix.
- Scale outputs are concatenated, reduced to `reduce_dim` tokens, and mapped to the horizon by `flatten_forecast_head`; `revin` wraps the model.

## When to use

- Long-horizon forecasting with patterns at several temporal scales; pairwise mixing of adjacent scales limits memory and resists noise.
- Small model budgets (MLP-Mixer backbone, no learned attention).
- Channels share weights independently: suits weakly correlated channels, not data whose signal lies in channel interactions. Point forecasts only.

## Configure

- `enc_in` and `c_out` follow the dataset channel count; both must equal it.
- `patch_len`, `stride`, `scale_factors` follow `seq_len`: each scale `s` uses patch `patch_len * s` and stride `stride * s`, and the largest scale must still leave patches (`(seq_len - patch_len*s) // (stride*s) + 2 >= 1`); scales are strictly increasing integers starting at 1.
- Other hyperparameters: preset defaults in `configs/models/SEMixer.toml`; tune generically.

## Differences

Clean-room rewrite; the official repository (no license, `NOASSERTION`) was read only to resolve paper omissions.

- Generalizes the official hard-coded four scales `{1, 2, 4, 8}` to any strictly increasing `scale_factors` schedule; identical for the default.
- Only the shared (non-individual) flatten head and only the RAM attention path (the reported configuration) are implemented; official alternative backbones (ProbAttention, AutoCorrelation, etc.) are omitted.
- `connection_probability` (0.85) and dropout (0.1), hard-coded upstream, are parameters with the same defaults.
- Reduced representation is `[patch, d_model]` before the flatten head instead of `[d_model, patch]`; the function class is unchanged.
- No data pipeline, optimizer schedule, checkpoint, or published-metric comparison.

Full detail in `reference.md`.
