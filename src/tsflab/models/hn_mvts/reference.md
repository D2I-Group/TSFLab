# HN_MVTS — reference

## Paper

- **Title**: HN-MVTS: HyperNetwork-based Multivariate Time Series Forecasting
- **Venue**: AAAI 2026, pages 25200-25208
- **arXiv**: https://arxiv.org/abs/2511.08340

## Abstract (shortened)

Complex channel-dependent models often underperform channel-independent ones, which are robust because of their small capacity. HN-MVTS adds a hypernetwork-based generative prior to an arbitrary forecasting model: a learnable embedding matrix of the series components is mapped to the weights of the target network's last layer, a data-adaptive regularizer that improves generalization and long-range accuracy. The hypernetwork is used only during training, so inference time equals the base model's. On eight benchmarks it typically improves DLinear, PatchTST, TSMixer and other models.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/SavchenkoK26,
  author    = {Andrey V. Savchenko and Oleg Kachan},
  title     = {{HN-MVTS:} HyperNetwork-based Multivariate Time Series Forecasting},
  booktitle = {Fortieth {AAAI} Conference on Artificial Intelligence, {AAAI} 2026},
  pages     = {25200--25208},
  publisher = {{AAAI} Press},
  year      = {2026},
  doi       = {10.1609/AAAI.V40I30.39711}
}
```
