# NSTransformer — reference

## Paper

- **Title**: Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting
- **Venue**: NeurIPS 2022 (arXiv 2205.14415, 2022-05)
- **Abstract (shortened)**: Transformers degrade on non-stationary real-world data whose joint distribution changes over time. Stationarization helps predictability but causes over-stationarization: indistinguishable temporal attentions for different series. Non-stationary Transformers is a generic framework with Series Stationarization (unify input statistics, restore them on output) and De-stationary Attention (approximate attentions learned from raw series). It reduces MSE by 49.43% on Transformer, 47.34% on Informer and 46.89% on Reformer.

## Implementation mapping

- Stationarization: per-window mean and population standard deviation (`+1e-5`), both detached; output is `projection(decoded) * std + mean`.
- `Projector`: a linear temporal pooling of the raw series (`seq_len -> 1` per channel), concatenated with the statistic, then an MLP with GELU (`p_hidden_dims[:p_hidden_layers]`). `tau_learner` reads `std` and outputs one value; `delta_learner` reads `mean` and outputs `seq_len` values.
- `DeStationaryAttention`: fused QKV projection, scores scaled by `tau` per sample and shifted by `delta` over key positions, then `1/sqrt(d_head)` and softmax.
- `NSBlock`: post-norm residual attention and GELU feed-forward; the encoder uses self-attention, the decoder uses cross-attention from future queries to the encoder output, both with the same `tau` and `delta`.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/LiuWWL22,
  author    = {Yong Liu and Haixu Wu and Jianmin Wang and Mingsheng Long},
  title     = {Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting},
  booktitle = {Advances in Neural Information Processing Systems 35, NeurIPS 2022},
  year      = {2022},
  url       = {http://papers.nips.cc/paper\_files/paper/2022/hash/4054556fcaa934b0bf76da52cf4f92cb-Abstract-Conference.html}
}
```
