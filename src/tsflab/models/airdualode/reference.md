# AirDualODE — reference

## Paper

- **Title**: Air Quality Prediction with Physics-Guided Dual Neural ODEs in Open Systems
- **Venue**: ICLR 2025 (arXiv 2410.19892, 2024-10)
- **arXiv**: https://arxiv.org/abs/2410.19892

## Abstract

Physics-based air-quality models are costly and assume closed systems, while data-driven models may miss essential physical dynamics; earlier hybrids mismatch explicit equations and learned representations.
Air-DualODE uses two neural-ODE branches: one applies open-system physical equations to learn physics dynamics, the other learns the remaining dependencies in a fully data-driven way.
The two representations are temporally aligned and fused, giving state-of-the-art pollutant forecasts across spatial scales.

## Runtime contract

Inputs are `x_enc [B, seq_len, N]` plus raw or node-structured meteorology; the distance adjacency and the directed wind/flow adjacency are construction inputs. Output is `[B, pred_len, N]`.

## Citation

```bibtex
@inproceedings{tian2025airdualode,
  author    = {Jindong Tian and Yuxuan Liang and Ronghui Xu and Peng Chen and Chenjuan Guo and Aoying Zhou and Lujia Pan and Zhongwen Rao and Bin Yang},
  title     = {Air Quality Prediction with Physics-Guided Dual Neural ODEs in Open Systems},
  booktitle = {The Thirteenth International Conference on Learning Representations (ICLR 2025)},
  year      = {2025},
  url       = {https://openreview.net/forum?id=kOJf7Dklyv}
}
```
