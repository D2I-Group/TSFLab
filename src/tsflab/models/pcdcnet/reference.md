# PCDCNet — reference

## Paper

- **Title**: PCDCNet: A Surrogate Model for Air Quality Forecasting with Physical-Chemical Dynamics and Constraints
- **Venue**: arXiv preprint 2505.19842 (2025-05)
- **Abstract (shortened)**: Numerical air-quality models (CMAQ, WRF-Chem) are physically grounded but expensive and depend on uncertain emission inventories; deep models are cheap but generalize poorly without physical constraints. PCDCNet is a surrogate that models pollutant formation, transport, and dissipation with emissions, meteorology, and domain-informed constraints, combining graph-based spatial transport, recurrent temporal accumulation, and local-interaction representations. It reports state-of-the-art 72-hour station-level PM2.5 and O3 forecasts at much lower cost and is deployed in an online platform.

## Differences in detail

An author implementation exists at `src/models/pcdcnet.py` in CauAir at the
pinned revision (class `PCDCNet`). The local model keeps the LID/STD/TAD update
order and exposes `domain_informed_constraint()` after a forward pass, but
differs from that file: the official transport is a two-layer residual GELU
graph convolution with DropEdge over the supplied `gso`, whereas the local
transport is one projection of the symmetric-normalized Laplacian message; the
official feature-mixing MLP is 4x wide (local 2x); the official update reads the
increment only from the fused hidden state, whereas the local update also adds a
zero-mean transport flux; and the official forward returns a spatial/temporal
regularizer on graph-branch readouts (weight 10) with the forecast, whereas the
local penalty is exposed but not called by the trainer. Standard calendar
covariates are a reduced substitute for the paper's meteorology and emissions
inputs, which the official loop feeds over history and horizon.

## Citation

```bibtex
@misc{wang2025pcdcnet,
  author        = {Shuo Wang and Yun Cheng and Qingye Meng and Olga Saukh and Jiang Zhang and
                   Jingfang Fan and Yuanting Zhang and Xingyuan Yuan and Lothar Thiele},
  title         = {PCDCNet: A Surrogate Model for Air Quality Forecasting with Physical-Chemical Dynamics and Constraints},
  year          = {2025},
  eprint        = {2505.19842},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2505.19842}
}
```
