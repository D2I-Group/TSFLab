# FreDF — reference

## Paper identity

"FreDF: Learning to Forecast in the Frequency Domain" (arXiv 2402.02399, Wang
et al., ICLR 2025) proposes a training loss that aligns forecasts and labels in
the frequency domain to reduce label autocorrelation. It is a different method;
nothing in this entry is taken from it.

## Differences in detail

- Resolved from the official code (Algorithm 1 and Eqs. 6, 14 leave it open):
  the history is instance-normalized, passed through `Linear(seq_len, seq_len)`
  + ReLU + dropout over time, zero-padded by the horizon, embedded with
  `DataEmbedding` (`timeF`) over the concatenated history and horizon marks;
  each block is `x + dropout(FDBlock(x))`; the last `pred_len` steps pass
  through `Linear(pred_len, pred_len)` + ReLU + dropout and a
  `d_model -> channels` projection before de-normalization. The per-bin complex
  maps are bias-free.
- Equivalent reformulation: Algorithm 1 transforms K masked spectrum copies back
  and sums them with the frequency weights; since the inverse FFT is linear and
  each weight is a real scalar, one weighted inverse transform is the same.
- Fusion weights: the official `ConcatenationWithLearnableCoefficient` is
  initialised from `randn` and overwritten with `softmax(self.weights)` on every
  forward, iterating the softmax on the stored parameter instead of
  differentiating through it. Here `frequency_weight` holds `randn`-initialised
  logits and the forward uses `softmax(frequency_weight)`, always on the
  simplex.
- `irfft` default length `2 * (bins - 1)` equals `seq_len + pred_len` only when
  that sum is even.
- Normalization: biased variance plus `1e-5`, affine off.
- Calendar: official `--freq` accepts other values; the embedding is always on
  (the official `is_embed` is always true in practice).
- `d_model = 128`, `e_layers = 1`, `dropout = 0.1` match the official
  defaults; its batch size 4 and learning rate 1e-4 are runner settings, not in
  the preset. Reported benchmark numbers are not reproduction claims; no
  training was run.

## Citation

Zhang, X., Zhao, S., Song, Z., Guo, H., Zhang, J., Zheng, C., Qiang, W. "Not All Frequencies Are Created Equal: Towards a Dynamic Fusion of Frequencies in Time-Series Forecasting." ACM MM 2024. arXiv:2407.12415.
