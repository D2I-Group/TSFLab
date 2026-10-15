# TimeKAN — reference

## Paper

TimeKAN: KAN-based Frequency Decomposition Learning Architecture for Long-term Time Series
Forecasting (Huang, Zhao, Li, Bai; ICLR 2025; arXiv:2502.06910).

Real-world series mix multiple frequency components whose pattern information density differs, so a
uniform model characterizes them poorly. TimeKAN uses Cascaded Frequency Decomposition blocks (bottom-up
band extraction), Multi-order KAN Representation Learning blocks that exploit KAN flexibility to model
each band, and Frequency Mixing blocks that recombine the bands. It reaches state-of-the-art results as
an extremely lightweight architecture.

## Implementation notes

- Levels: `down_sampling_layers + 1`; level `i` has Chebyshev order `begin_order + levels - i - 1`, so the finest level gets the highest order.
- Chebyshev coefficients `[width, width, order + 1]` are initialized `N(0, (width * (order + 1))^-0.5)`.
- Each scalar series is embedded by `Linear(1, d_model)` before the blocks; `e_layers` blocks are stacked.
- FFT-padding upsampling: rFFT of the coarse level, zero-pad to `target_length // 2 + 1` bins, inverse rFFT.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/HuangZL025,
  author    = {Songtao Huang and Zhen Zhao and Can Li and Lei Bai},
  title     = {TimeKAN: KAN-based Frequency Decomposition Learning Architecture for Long-term Time Series Forecasting},
  booktitle = {The Thirteenth International Conference on Learning Representations, {ICLR} 2025},
  publisher = {OpenReview.net},
  year      = {2025},
  url       = {https://openreview.net/forum?id=wTLc79YNbh}
}
```
