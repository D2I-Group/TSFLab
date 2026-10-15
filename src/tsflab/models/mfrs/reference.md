# MFRS — reference

## Paper

Yu, Tu, Bai, "MFRS: A Multi-Frequency Reference Series Approach to Scalable and Accurate Time-Series
Forecasting", arXiv 2503.08328 (2025-03). Sources used: Sec. 3.1-3.4, Algorithms 1-3, Sec. 4.1-4.3, App. A-C.

## Differences in detail

- **Shared components.** `self_attention_family` (`AttentionLayer` around a non-causal `FullAttention`) provides
  the multi-head variate-to-reference cross-attention, `A = QK^T / sqrt(D)` with queries from variates and keys and
  values from reference series.
- **Algorithm 1 details.** After keeping `T`, the interval is zeroed; mapping each rFFT bin to its nearest integer
  period avoids repeating one long period across the neighbouring integers that share a coarse bin.
- **Algorithm 2 details.** `k` is bounded so each harmonic stays below half the frequency of the next shorter
  pattern; the channel-mean amplitude spectrum feeds Algorithm 1 and Algorithm 2 reads the nearest bin `round(L k / T)`.
- **Time steps.** With timestamps, the step counts sampling intervals from the Unix epoch at the window's own
  interval (a constant phase shift). Algorithm 3 uses the last `T_M + S` steps of the training split; the window's
  step is the stretch start plus `xi`.
- **Normalization.** The paper's "LayerNorm" DC blocking is implemented as per-window standardization of variates
  without affine parameters, restored on the output (as its inverse formula implies).
- **Training.** The extractor needs the training split and an absolute-time reference for phases. Training uses the
  configured loss through the catalog trainer; reported benchmark numbers are not reproduction claims.
- **Verified properties.** The calendar-to-minute conversion; Algorithm 1 against a brute-force evaluation of its
  rule on a spectrum with daily, weekly and harmonic content; the Algorithm 2 harmonic bounds and scores; the four
  App. B waveforms; pattern collection with manual periods and deduplication; Algorithm 3 recovering a window's step;
  absolute steps from hourly and 15-minute timestamps; the forward pass against a direct composition of the
  cross-attention layers; channel independence; phase dependence (and invariance to a shift by a whole common
  period); the state round trip of the data-sized buffers; gradients; `training_setup`; the strict schema.

## Citation

```bibtex
@article{yu2025mfrs,
  title   = {MFRS: A Multi-Frequency Reference Series Approach to Scalable and Accurate Time-Series Forecasting},
  author  = {Yu, Liang and Tu, Lai and Bai, Xiang},
  journal = {arXiv preprint arXiv:2503.08328},
  year    = {2025}
}
```
