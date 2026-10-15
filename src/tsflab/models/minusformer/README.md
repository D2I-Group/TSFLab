---
name: "Minusformer"
description: "Inverted variate-token Transformer whose blocks progressively subtract what they explain and sum outputs with alternating signs (deep boosting). Use for correlated multivariate data where deep Transformers overfit; not for univariate series or very many channels (full variate attention)."
---

# Minusformer

## Idea

- `SeriesScaler` standardizes each variate over the lookback window and `embed.DataEmbedding_inverted` maps every whole series (and, with `use_marks`, every `timeF` calendar feature) to one `d_model` token (Algorithm 1, lines 1-3).
- `MinusBlock` (Eqs. 5-9): full attention over variate tokens is subtracted from the input (Eq. 7), the LayerNorm-ed result passes a feed-forward network whose output is subtracted again (Eq. 5b), and a sigmoid gate on the residual gives the next block's input (Eq. 8).
- Each block also emits a sigmoid-gated linear map of the concatenated attention and feed-forward outputs to `d_block` steps (Eq. 9).
- `subtract_outputs` forms `O_{l+1} = O_hat_{l+1} - O_l`, so the forecast is Eq. (2)'s alternating-sign ensemble of block outputs; an optional linear map aligns `d_block` with the horizon before de-standardization.

## When to use

- Designed to make deep Transformer forecasters learn residuals progressively (boosting-style), reducing redundancy and overfitting as depth grows.
- Attention runs across variate tokens, so it suits correlated multivariate data; per-window standardization handles level drift.
- Full attention over variates grows quadratically with channel count; for univariate series there is nothing to attend across.
- Calendar tokens exist only for hourly data.

## Configure

- `enc_in`: number of variates (one token each).
- `freq`: only hourly (`h`) calendar tokens are supported; set `use_marks = false` for other sampling rates.
- `d_block`: block output length; defaults to `pred_len`, otherwise a linear map aligns it to the horizon.

Other hyperparameters: preset defaults in `configs/models/Minusformer.toml`; tune generically.

## Differences

- Independent rewrite from Section 2.4 and Appendix H after reading the pinned official code (no license file; nothing copied).
- The residual-stream gate is followed by a second LayerNorm, as in the code (not shown in Algorithm 1).
- With `use_marks = false` calendar tokens are removed; the official code appends them whenever marks are given.
- FlashAttention/ProbAttention variants and the Minus-Autoformer/Informer/Flowformer/Periodformer generalizations (Section 3.5) are not implemented.
- The official encoder's unused final LayerNorm is omitted; pointwise `Conv1d` layers are written as equivalent `Linear` layers.
