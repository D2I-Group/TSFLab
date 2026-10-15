# CauAir — reference

## Paper

- **Title**: Causal Learning Meet Covariates: Empowering Lightweight and Effective Nationwide Air Quality Forecasting
- **Venue**: IJCAI 2025
- **Published**: 2025
- **arXiv**: N/A

## Abstract

Existing air-quality models scale poorly to nationwide data and treat weather covariates as optional,
although weather causally affects AQI. The paper releases a nationwide air-quality dataset and proposes
CauAir, a Transformer that explicitly models the causal association between weather covariates and AQI.
Its CachLormer removes redundant components and uses cache-attention with learnable embeddings to
perceive that association coarsely, giving competitive accuracy with high training efficiency and low memory.

## Implementation mapping

- Inputs: `x_enc [B, seq_len, N]` plus historical and future weather `[B, time, N, cov_dim]`; raw
  timestamps provide a two-calendar-feature fallback. Output: `[B, pred_len, N]`.
- Adjacency is deliberately ignored because cache-attention is graph-free.
- Cache assignment and reconstruction map to equations (6)-(11); the two causal stages stay explicit
  in `Model.forward`.
- `src/models/cauair.py` of the official repository was inspected at the pinned revision to confirm
  details; no external source file was copied.

## Citation

```bibtex
@inproceedings{DBLP:conf/ijcai/MaCW0ZZW25,
  author    = {Jiaming Ma and Zhiqing Cui and Binwu Wang and Pengkun Wang and Zhengyang Zhou and Zhe Zhao and Yang Wang},
  title     = {Causal Learning Meet Covariates: Empowering Lightweight and Effective Nationwide Air Quality Forecasting},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, {IJCAI} 2025, Montreal, Canada, August 16-22, 2025},
  pages     = {3171--3179},
  publisher = {ijcai.org},
  year      = {2025},
  url       = {https://doi.org/10.24963/ijcai.2025/353},
  doi       = {10.24963/IJCAI.2025/353}
}
```
