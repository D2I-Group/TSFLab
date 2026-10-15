# STHD — reference

## Differences in detail

- Sources read at revision `97a36a59597ec26a7bd668e1fd240a5de1259cd0`: `models/STHD.py`, `layers/Embed.py` (`PatchEmbedding_Multidim`, `PositionalEncodingTC`), `layers/Transformer_EncDec.py`, `layers/SelfAttention_Family.py`, `data_provider/data_loader.py`, `datasets/top-k-train/corr-compute.py`, `run_crime.py`, `run_wiki.py`; paper Sec. 4 (Eqs. 1-5, Figs. 3-4), arXiv v1 of the CIKM 2024 paper.
- Correlations are computed on the z-normalized training split (first 70% of time steps) and ranked in descending order with the series itself removed; each sample stacks the target first, then its related series in rank order.
- Inputs are normalized per series (mean, biased standard deviation, `eps = 1e-5`) and the target forecast is de-normalized with the target's statistics, which equals the catalog `revin` with `affine=False` applied before grouping.
- Padding is one stride of end replication; the time code is the base-10000 sinusoid and the channel code a base-5000 sinusoid added after reshaping to the channel axis, followed by one dropout.
- The encoder is the Time-Series-Library post-norm layer (full non-causal attention, 1x1-convolution feed-forward, GELU) with a final LayerNorm; the head flattens `(d_model, P)` and applies a linear layer with dropout to the target only.
- Defaults follow `run_crime.py`: `k = 13`, `patch_len = 12`, `stride = 6`, `d_model = 256`, `n_heads = 4`, `e_layers = 2`, `d_ff = 384`, `dropout = 0.2`.
- ReIndex (Sec. 4.2) draws batches of single (target, window) samples from the `M x S` pool; the catalog loader yields windows of all channels, so a batch is processed as `b x N` target groups (the conventional batching the paper compares against).
- DeepGraph only accelerates the correlation computation; the dense computation gives the same values.
- Before `training_setup` runs (for example in a contract check), `related` holds the cyclically next `k` channels so the model stays runnable.
- The official loader reshapes the stacked `(time, 1 + K, M)` array to `(time, M, 1 + K)` without transposing.
- Training uses the catalog trainer and configured loss (the paper trains with MSE and Adam). Reported benchmark numbers are not reproduction claims of this implementation.

## Citation

```bibtex
@inproceedings{zhou2024sthd,
  title     = {Scalable Transformer for High Dimensional Multivariate Time Series Forecasting},
  author    = {Zhou, Xin and Wang, Weiqing and Buntine, Wray and Qu, Shilin and Sriramulu, Abishek and Tan, Weicong and Bergmeir, Christoph},
  booktitle = {Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (CIKM)},
  pages     = {3515--3526},
  year      = {2024}
}
```
