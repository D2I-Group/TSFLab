# TimeCAP — reference

## Paper

TimeCAP: A Channel-Aware Pre-Training Framework for Multivariate Time Series Forecasting (Ren et al.,
AAAI 2026, Oral, pp. 25108-25116).

TimeCAP is a purely channel-aware pre-training framework that internalizes latent relationships among
variables in multi-domain data and transfers them downstream. It combines flexible channel grouping
with adaptive meta-routing to capture intra-group local patterns and global coherence, uses self- and
cross-attention with a channel-aware mask confined to time-aligned tokens, and reconciles
autoregressive and one-shot generation. Few-shot evaluation shows 11.8% / 6% average MSE / MAE
reductions, with gains in full-shot and zero-shot settings too.

## Implementation notes

- Group `g` covers channels `(g * group_stride + o) mod enc_in` for `o < group_size`; each group's tokens are `group_size` channel tokens plus one router token per patch.
- `fusion_midpoint` defaults to `(pred_len + 1) / 2`; the one-shot weight is `sigmoid(fusion_alpha * (step - midpoint))`.
- RevIN wraps the model; `d_model` must be divisible by `num_heads`.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/RenLHZZLLL26,
  author    = {Chuanru Ren and Yao Lu and Tianjin Huang and Haowen Zheng and Hengde Zhu and Yunyin Li and Hengxiao Li and Lu Liu},
  title     = {TimeCAP: {A} Channel-Aware Pre-Training Framework for Multivariate Time Series Forecasting},
  booktitle = {Fortieth {AAAI} Conference on Artificial Intelligence, {AAAI} 2026},
  pages     = {25108--25116},
  publisher = {{AAAI} Press},
  year      = {2026},
  doi       = {10.1609/AAAI.V40I30.39700}
}
```
