# Kronos — reference

## Paper

Shi et al., AAAI 2026 (arXiv 2508.02739, https://arxiv.org/abs/2508.02739).

Time-series foundation models underperform on financial candlestick (K-line) data and overlook tasks such as
volatility prediction and synthetic data generation. Kronos is a pre-training framework for K-line modelling with
a tokenizer that discretizes continuous market data into token sequences, keeping price dynamics and trade
activity. It is pre-trained autoregressively on over 12 billion K-line records from 45 exchanges and excels
zero-shot: RankIC +93% over the leading TSFM, 9% lower volatility MAE, and 22% better synthetic-sequence fidelity.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/ShiFCZXZL26,
  author    = {Yu Shi and Zongliang Fu and Shuo Chen and Bohan Zhao and Wei Xu and
               Changshui Zhang and Jian Li},
  editor    = {Sven Koenig and Chad Jenkins and Matthew E. Taylor},
  title     = {Kronos: {A} Foundation Model for the Language of Financial Markets},
  booktitle = {Fortieth {AAAI} Conference on Artificial Intelligence, {AAAI} 2026, Singapore},
  pages     = {25366--25373},
  publisher = {{AAAI} Press},
  year      = {2026},
  url       = {https://doi.org/10.1609/aaai.v40i30.39730},
  doi       = {10.1609/AAAI.V40I30.39730}
}
```
