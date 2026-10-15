# COSA — reference

## Paper

- **Title**: COSA: Context-aware Output-Space Adapter for Test-Time Adaptation in Time Series Forecasting
- **Venue**: ICLR 2026
- **Published**: 2026
- **OpenReview**: https://openreview.net/forum?id=L7Z5wBMPrW

## Abstract

Deployed forecasters degrade under non-stationarity and distribution shift, and in forecasting the ground
truth arrives shortly after prediction. COSA is a minimal plug-and-play adapter that directly corrects a
frozen model's predictions by a gated residual over the original prediction and a lightweight context of
recently observed ground truth; only the adapter is updated at test time under a leakage-free protocol with
an adaptive learning rate. It reports 13.91-17.03% gains over no TTA and 10.48-13.05% over SOTA TTA
methods, largest at long horizons, with negligible overhead.

## Citation

```bibtex
@inproceedings{im2026cosa,
  author    = {Jeonghwan Im and Hyuk-Yoon Kwon},
  title     = {{COSA}: Context-aware Output-Space Adapter for Test-Time Adaptation in Time Series Forecasting},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=L7Z5wBMPrW}
}
```
