# CrossGNN — reference

## Paper

- **Title**: CrossGNN: Confronting Noisy Multivariate Time Series Via Cross Interaction Refinement
- **Venue**: NeurIPS 2023
- **Published**: 2023
- **arXiv**: N/A

## Abstract

Existing Transformer and GNN forecasters handle temporal fluctuations and heterogeneity between variables
poorly. CrossGNN is a linear-complexity GNN that refines cross-scale and cross-variable interaction: an
adaptive multi-scale identifier (AMSI) builds multi-scale series with reduced noise, a Cross-Scale GNN
extracts scales with clearer trend and weaker noise, and a Cross-Variable GNN uses homogeneity and
heterogeneity between variables. Keeping only salient edges makes time and space complexity O(L).

## Implementation mapping

Paper Eqs. 1-5 map to FFT-selected period pooling; Eqs. 6-10 to scale-sensitive and trend-preserving
temporal edges; Eqs. 11-13 to positive/negative variable edges; Eq. 14 to the direct multi-step head.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/HuangSZDWZW23,
  author    = {Qihe Huang and Lei Shen and Ruixin Zhang and Shouhong Ding and Binwu Wang and Zhengyang Zhou and Yang Wang},
  title     = {CrossGNN: Confronting Noisy Multivariate Time Series Via Cross Interaction Refinement},
  booktitle = {Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023},
  year      = {2023},
  url       = {http://papers.nips.cc/paper\_files/paper/2023/hash/9278abf072b58caf21d48dd670b4c721-Abstract-Conference.html}
}
```
