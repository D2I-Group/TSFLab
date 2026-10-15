# PRformer — reference

## Differences in detail

- Implementation: independent rewrite from the Method section (Eqs. 1-5, Figs. 1-2) of the paper after reading the pinned official code (`usualheart/PRformer`, revision `df718ad837db1db5fe6ab3d085119eb116b7d241`, Apache-2.0): `model/PRformer.py`, `layers/Embed.py` (`PyramidalRNNEmbedding`, `ConvRNNBlock`), `layers/WindowListSplit.py`, `layers/Transformer_EncDec.py`, `layers/SelfAttention_Family.py`, `run.py`, and `scripts/multivariate_forecast/*.sh`. Nothing was copied or imported.
- Chain construction (from the code): largest window first, appending each smaller divisor, the smallest window reusable across chains; `scale_hidden_sizes` splits a scale's GRU units between chains that share a window.
- Convolutions: the first convolution's kernel and stride equal the first period; every convolution outputs 128 channels with no activation.
- Upsampling: nearest neighbour; the upsampled stream is itself re-upsampled (the fused sum feeds only the GRU), with the last step repeated when the lengths differ. When an upsampled map is shorter than the target scale by more than one step it is padded to length (the official code pads exactly one step and fails otherwise).
- RNN block: each GRU has one layer and its last hidden state is the scale embedding; `alpha` is initialised to `1/l`; a `Linear(d_model, d_model)` follows the concatenation. The output linear layer reads the actual concatenated width, which equals `d_model` whenever `d_model` divides evenly as in the official scripts.
- Scale weights follow the paper (`softmax(alpha / T)` across the `l` scales); the official code stores `alpha` as an `[l, 1]` tensor and takes the softmax over its singleton axis, so every official weight is exactly 1 and `alpha` never trains.
- Attention follows Eq. (5) without a mask; the official encoder builds `FullAttention(mask_flag=True)` and therefore applies a triangular causal mask over the variate order; set `causal_variate_mask = true` to reproduce that.
- Calendar marks are not appended as extra tokens (the official run feeds time features as additional PRE tokens and drops their outputs).
- Normalization and encoder: RevIN is affine and uses the cataloged `sqrt(var + eps)` scale instead of the official `rsqrt(clamp(var, eps))`; the encoder is the Time-Series-Library post-norm layer (conv-FFN, GELU) with a final LayerNorm.
- Defaults (`n_heads = 8`, `dropout = 0.1`, `rnn_mix_temperature = 0.002`, `conv_windows = 24 48 72 96 144`) follow `run.py`; the preset follows `PRformer-ETTh1.sh` (`seq_len = 720`, windows 24 48 72 144, `e_layers = 5`, `d_model = d_ff = 720`).
- Checked behaviour: the chain configuration of the official scripts and the hidden-size split, kernel = stride = period ratio and the level lengths of Eq. (2), the recursive upsampling and lateral addition of Eq. (3), last-step padding, the temperature softmax and weighted concatenation of Eq. (4), per-variate independence of the embedding, the full forward pipeline, full versus causal variate attention, RevIN shift equivariance, gradient flow to the scale logits, and the parameter schema. Reported benchmark numbers are not reproduction claims of this implementation.

## Paper

PRformer: Pyramidal Recurrent Transformer for Multivariate Time Series Forecasting, arXiv preprint 2408.10483 (2024-08).

## Citation

```bibtex
@misc{yu2024prformer,
  title         = {PRformer: Pyramidal Recurrent Transformer for Multivariate Time Series Forecasting},
  author        = {Yu, Yongbo and Yu, Weizhong and Nie, Feiping and Li, Xuelong},
  year          = {2024},
  eprint        = {2408.10483},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```
