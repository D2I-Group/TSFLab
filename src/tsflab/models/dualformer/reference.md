# Dualformer — reference

## Differences in detail

- **Depth-wise chaining.** In the pinned revision, every layer's band-limited
  input is recomputed from the original embedding's spectrum and each layer's
  output overwrites the previous one without being consumed by it, so only the
  last encoder layer of each branch reaches the fused output or receives a
  gradient, and the paper's depth-wise curriculum never composes. Here the
  spectrum is re-derived from each branch's running state every layer.
- **Band-pass reconstruction.** The pinned code calls `torch.fft.irfft` on the
  sliced bins with `n=seq_len`, reinterpreting them as starting at frequency 0.
  Zero-padding the unselected bins back to their positions is the alias-free
  reconstruction the paper's "Padding" step implies.
- **Embedding.** `DataEmbedding_wo_pos` (learned token embedding plus a
  learned/linear calendar embedding selected by `embed`/`freq`) is replaced by
  `forecast_embedding` (linear value projection plus a fixed-scale six-column
  raw-calendar projection); not asserted numerically equivalent.
- **Autocorrelation attention.** An independent rewrite of the
  Wiener-Khinchin autocorrelation attention (popularized by Autoformer) used
  unmodified by the official code; no source, including this catalog's
  Autoformer, was imported. Only batch-shared top-lag selection runs, so the
  model behaves the same on CPU in training and evaluation.
- **Band tiling.** Non-overlapping bands from `floor(n*k/L)` tile the spectrum
  exactly; earlier integer truncation left gaps (for example the highest bin).
  Outputs change only in the tiling regime (`alpha <= 1/e_layers`) or when
  `alpha * n_bins` is not an integer in the sliding regime; the default
  `alpha = 1.0` is unchanged. The paper (Sec. 3.1) asks case 1 to avoid gaps but
  gives no rounding.
- **Inputs.** `x_mark_enc` defaults to zero-filled six-column marks; decoder
  arguments are accepted and ignored (no decoder).
- **Evidence.** Structure and runtime contract only.

## Citation

Bai, J., Kawahara, Y. "Dualformer: Time-Frequency Dual Domain Learning for Long-term Time Series Forecasting." arXiv:2601.15669 (2026).
