# SST — reference

## Differences in detail

- Sources read at revision `b39292bfc54536c48a7932486d797116701083b8`: `models/SST.py`, `layers/Long_encoder.py`, `layers/Short_encoder.py`, `layers/LWT_layers.py`, `layers/RevIN.py`, `run.py`, `scripts/etth1.sh`; paper Sec. 4 (Definition 4.1, Eq. 6, Figs. 5-6).
- RevIN is non-affine and subtracts the mean; the router and both experts read the normalized window.
- The short range is the last `label_len` steps officially; here `short_len`, default `seq_len // 2` because the experiments use `L = 2S = 672`.
- Patches are end-padded by replicating the last value `stride` times.
- Each Mamba layer is `Mamba(d_model, d_state, d_conv)` with expansion 2 followed by a GELU feed-forward network, with no residual connection or normalization.
- The LWT is the PatchTST encoder: learnable uniform(+-0.02) position table, post-norm BatchNorm, GELU feed-forward, residual attention scores, dropout on the embedding, attention output, and sublayers.
- The router projects the variate axis with a linear layer to `d_model` and flattens over time; the fusion head concatenates the two weighted embeddings (`concat = 1`), shared across variates.
- Defaults (`m_layers = 1`, `d_state = 16`, `d_conv = 4`, `e_layers = 3`, `n_heads = 4`, `d_model = 16`, `d_ff = 128`, `dropout = 0.3`, `local_ws = 7`) follow the ETTh1 script.
- Local mask: the official `get_local_mask` builds the in-window mask and then fills those positions with `-inf`, so attention reaches only patches outside the window and a row is fully masked when the patch count is at most the half window. Here each patch attends only within `local_ws // 2` positions.
- The selective SSM uses the reference `dt` initialisation.
- Reported benchmark numbers are not reproduction claims of this implementation.

## Citation

```bibtex
@inproceedings{xu2025sst,
  title     = {SST: Multi-Scale Hybrid Mamba-Transformer Experts for Time Series Forecasting},
  author    = {Xu, Xiongxiao and Chen, Canyu and Liang, Yueqing and Huang, Baixiang and Bai, Guangji and Zhao, Liang and Shu, Kai},
  booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM)},
  year      = {2025}
}
```
