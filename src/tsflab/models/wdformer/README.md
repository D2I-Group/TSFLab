---
name: "WDformer"
description: "Inverted Transformer over Haar-subband variate tokens with differential attention (difference of two softmax maps) and an inverse-wavelet output head. Use for multivariate data with correlated channels and multi-scale structure; not for weakly related channels or probabilistic output."
---

# WDformer

## Idea

- `DecimatedWaveletTransform` (Haar, `wave_size` levels) splits each variate's history; each subband gets its own linear embedding, concatenated into one token per variate.
- `DifferentialAttention` layers subtract two softmax attention maps to cancel common-mode noise, with RMSNorm, a SwiGLU feed-forward, and a layer-dependent lambda schedule.
- A linear projector emits pseudo-subband coefficients and the inverse wavelet transform reassembles the forecast.
- Calendar marks, if given, are appended as extra variate tokens; instance standardization wraps the model.

## When to use

- Variate tokens attend to each other, so the model targets channels that inform each other.
- The Haar subbands separate coarse and fine scales of the lookback before mixing.
- Differential attention is meant to suppress attention noise shared by both maps.
- Point output only.

## Configure

- `enc_in`: must equal the dataset channel count (marks add tokens on top).
- `wave_size`: Haar levels; when `seq_len` or `pred_len` is not a multiple of `2 ** wave_size` one sample is padded at some level and trimmed on reconstruction, so multiples avoid padding. At most 12.

Other hyperparameters: preset defaults in `configs/models/WDformer.toml`; tune generically.

## Differences

- Paper: arXiv 2509.25231 (2025). Official code `xiaowangbc/WDformer` at `c741e4e4` (`model/WDformer.py`, `layers/SelfAttention_Family.py`, `layers/Embed.py`) has no license file, recorded as `NOASSERTION`.
- Clean-room derivation: the wavelet-subband embedding, inverse-wavelet output head, and differential-attention encoder loop were re-derived from the pinned files' module boundaries, default hyperparameters, and docstrings; no lines were copied. Differential attention was extracted into the paper-neutral `differential_attention` component, since subtracting two softmax maps with a learned mixing scalar is not specific to this paper.
- Wavelet: the official code uses `ptwt` `wavedec`/`waverec`. The cataloged `DecimatedWaveletTransform` is a from-scratch strided-convolution Haar analysis/synthesis pair, the same critically sampled orthogonal transform, with exact round trip.
- RMSNorm: the official `EncoderLayer` (`layers/Transformer_EncDec.py`) applies three independently parameterized `RMSNorm` instances to the same input before the query, key, and value projections. Because the projections are linear, the weights absorb any fixed per-branch rescaling, so one shared `RMSNorm` has the same representational capacity.
- Feed-forward: the official `EncoderLayer` builds `conv1`/`conv2` sized by `d_ff` but never calls them; the feed-forward is always `SwiGLU(d_model)` with width `2 * d_model`. `d_ff` is kept for preset parity with no effect.
- `lambda_init`: the paper uses the Differential-Transformer schedule `0.8 - 0.6 * exp(-0.3 * (l - 1))` with 1-based `l`; the official `model/WDformer.py` uses `0.7 - 0.5 * exp(-0.3 * l)` with 0-based `l`. TSFLab keeps the code's values.
- Padding: Haar decomposition of a length that is not a multiple of `2 ** wave_size` needs one extra padded sample at some level. The projection to pseudo-subband coefficients is sized to the exact padded coefficient lengths (not `pred_len`), and `DecimatedWaveletTransform.reconstruct` trims the padding, returning exactly `pred_len` samples.
- Calendar marks are concatenated to the variate axis before embedding and the forecast is read from the data variates.
