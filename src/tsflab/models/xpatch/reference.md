# xPatch — reference

## Paper

xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend Decomposition (AAAI 2025, arXiv 2412.17323).

Transformers struggle to fully exploit temporal relations because of attention. xPatch is a dual-stream, non-Transformer architecture built on a seasonal-trend exponential decomposition inspired by classical exponential smoothing, with an MLP-based linear stream and a CNN-based non-linear stream, studying patching and channel independence without attention. It also proposes a robust arctangent loss and a sigmoid learning-rate schedule against overfitting.

## Implementation notes

The EMA follows Eq. (2): `s[0] = x[0]`, `s[t] = alpha * x[t] + (1 - alpha) * s[t-1]`, with `seasonal = x - trend`. The seasonal branch unfolds channel-independent patches and applies an embedding, depthwise convolution, residual pooling, pointwise convolution, and an MLP head; the two horizon representations are fused by a learned MLP.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/StitsyukC25,
  author    = {Artyom Stitsyuk and Jaesik Choi},
  editor    = {Toby Walsh and Julie Shah and Zico Kolter},
  title     = {xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend
               Decomposition},
  booktitle = {Thirty-Ninth {AAAI} Conference on Artificial Intelligence, Thirty-Seventh
               Conference on Innovative Applications of Artificial Intelligence,
               Fifteenth Symposium on Educational Advances in Artificial Intelligence,
               {AAAI} 2025, Philadelphia, PA, USA, February 25 - March 4, 2025},
  pages     = {20601--20609},
  publisher = {{AAAI} Press},
  year      = {2025},
  url       = {https://doi.org/10.1609/aaai.v39i19.34270},
  doi       = {10.1609/AAAI.V39I19.34270}
}
```
