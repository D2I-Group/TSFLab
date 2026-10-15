# EMAformer — reference

## Inputs and marks

`forward(x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None)` takes a history
`[batch, seq_len, enc_in]` and the raw six-column calendar marks
`[batch, seq_len, 6]` (`year, month, day, weekday, hour, minute`); only the last
two history marks are read (time and sampling interval). `x_dec`/`x_mark_dec`
are ignored. Without marks the phase is 0.

## Paper-to-code map

- Eq. (4): `DataEmbedding_inverted(seq_len, d_model)` from the shared `embed`
  component, called without marks.
- Eqs. (5)-(8): `channel_embedding [C, d]`, `phase_embedding [cycle, d]`,
  `joint_embedding [cycle, C * d]`, all Xavier-normal initialized as in the
  official code, summed in `armored_tokens`.
- Eqs. (9)-(12): `Encoder` of `e_layers` `EncoderLayer`s with
  `AttentionLayer(FullAttention(mask_flag=False))`, kernel-1 convolutional FFN
  and a final `LayerNorm`.
- Head (paper: "an MLP"; layout from the official code): `d -> 2d -> 4d ->
  pred_len` with GELU and `output_proj_dropout`, applied to `encoder output + Z0`.
- Optional normalization sandwich (`use_norm`): per-window mean and biased
  standard deviation with `1e-5`, as in iTransformer.

## Differences in detail

- Sources read: `model/EMAformer.py`, `layers/Embed.py`,
  `layers/SelfAttention_Family.py`, `layers/Transformer_EncDec.py`,
  `data_provider/data_loader.py`, `run.py`, `scripts/`.
- Phase index: the step index is elapsed minutes since 1970-01-01 divided by the
  interval of the last two marks; the official loader counts rows from the
  first CSV row. Assumes a constant sampling interval of at least one minute.
- The official scripts tune `d_model`, `n_heads`, `e_layers`, `d_ff`,
  `output_proj_dropout` and `cycle` per dataset (ETTh1: 256 / 4 / 3 / 256 / 0.3
  / 24).

## Citation

Zhang, Z., Du, X., Guo, X., Wang, W., Han, W. "EMAformer: Enhancing Transformer through Embedding Armor for Time Series Forecasting." AAAI 2026.
