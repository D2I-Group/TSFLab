# Informer — reference

## Paper

Zhou et al., AAAI 2021 (arXiv 2012.07436, https://arxiv.org/abs/2012.07436).

Long sequence time-series forecasting (LSTF) needs efficient long-range dependency modelling, but vanilla
Transformers suffer quadratic time and memory and slow step-by-step decoding. Informer adds (i) ProbSparse
self-attention with O(L log L) time and memory, (ii) self-attention distilling that halves each layer's input
to handle very long inputs, and (iii) a generative decoder that predicts the whole horizon in one forward
pass. It outperforms prior methods on four large-scale datasets.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/ZhouZPZLXZ21,
  author    = {Haoyi Zhou and Shanghang Zhang and Jieqi Peng and Shuai Zhang and
               Jianxin Li and Hui Xiong and Wancai Zhang},
  title     = {Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting},
  booktitle = {Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI} 2021},
  pages     = {11106--11115},
  publisher = {{AAAI} Press},
  year      = {2021},
  url       = {https://doi.org/10.1609/aaai.v35i12.17325},
  doi       = {10.1609/AAAI.V35I12.17325}
}
```
