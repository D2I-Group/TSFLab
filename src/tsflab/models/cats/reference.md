# CATS — reference

## Paper

- **Title**: Are Self-Attentions Effective for Time Series Forecasting?
- **Venue**: NeurIPS 2024
- **Published**: 2024 (arXiv: 2024-05)
- **arXiv**: https://arxiv.org/abs/2405.16877

## Abstract

The paper asks whether self-attention itself is effective for time-series forecasting, given that
simpler linear models can outperform Transformers. It proposes CATS (Cross-Attention-only Time
Series transformer), which removes self-attention, uses future horizon-dependent parameters as
queries, and shares parameters more aggressively, improving long-term accuracy while reducing
parameter count and memory use.

## Differences in detail

**Component-extraction note.** CATS centers on `x_enc[:, -1:, :]` without `.detach()`, unlike the
`last_value_center` component (used by NLinear, SegRNN, CrossGNN), so gradients flow through the
additive level term back into the encoder. This differs from the component's fixed detach contract,
so the centering stays model-local.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/Kim00K24,
  author    = {Dongbin Kim and Jinseong Park and Jaewook Lee and Hoki Kim},
  title     = {Are Self-Attentions Effective for Time Series Forecasting?},
  booktitle = {Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024},
  year      = {2024},
  url       = {http://papers.nips.cc/paper\_files/paper/2024/hash/cf66f995883298c4db2f0dcba28fb211-Abstract-Conference.html}
}
```
