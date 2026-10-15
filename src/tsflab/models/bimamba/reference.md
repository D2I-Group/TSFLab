# BiMamba — reference

## Paper

- **Title**: Bi-Mamba+: Bidirectional Mamba for Time Series Forecasting
- **Venue**: arXiv preprint (2404.15772, 2024-04)
- **arXiv**: https://arxiv.org/abs/2404.15772

## Abstract

Long-term forecasting must capture long-range dependencies in semantically sparse series; Mamba balances accuracy and efficiency through selective, hardware-aware scans.
Mamba+ adds a forget gate that combines new and historical features complementarily, and Bi-Mamba+ applies it forward and backward to capture interactions among series elements.
A series-relation-aware decider chooses between channel-independent and channel-mixing tokenization per dataset; the model improves over prior methods on 8 real-world datasets.

## Citation

```bibtex
@misc{liang2024bimamba,
  author        = {Aobo Liang and Xingguo Jiang and Yan Sun and Xiaohou Shi and Ke Li},
  title         = {Bi-Mamba+: Bidirectional Mamba for Time Series Forecasting},
  year          = {2024},
  eprint        = {2404.15772},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2404.15772}
}
```
