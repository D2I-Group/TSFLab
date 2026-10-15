# GMRL — reference

## Differences in detail

- Sources read: arXiv 2306.00390 (Secs. 3-4) and the official code at `ef37b8a4` (`models/GMRL.py`, `main.py`, `utils/data_utils_mutilsource.py`, `utils/tester_mutilsource.py`, `data/NYC.h5`).
- Tensor input: TSFLab's spatiotemporal mode supplies `[B, T, N]`; the `(location, source)` tensor is carried as `N = num_sources * L` source-major channels (`Model.to_tensor`). With `num_sources = 1` the model runs on ordinary node datasets. Marks and adjacency are not used (the method has none).
- Embedding: a 1x1 projection of the values concatenated with `e_t + e_l + e_s`, followed by a 1x1 conv and ReLU (the official `proj3`, absent from the paper).
- GMRE: the mixture projections are one kernel-1 convolution shared across channels (Eqs. 3-4 index weights per channel); `GMRETELayer` folds the `S * C` outputs back to the source axis.
- Cluster Norm uses the cluster means; the official code aliases `new_mu` and `new_sigma` so the means never enter. The cluster objective follows Eq. 7 (the official `F.kl_div` call computes the reverse direction with a geometric-mean `Q`) and is averaged over layers (the official loop keeps only the last layer's loss).
- The TTSE embeddings, memory `M`, `W_q` and `W_fc` are registered and trained (officially `nn.Parameter(...).to(device)` makes them plain tensors on CUDA); HRA projections carry the paper's biases `b_Q`, `b_V`.
- Dilation rates follow the code (1, 2, 4, 8); the paper's {2, 4, 8, 16} with kernel 2 would need 31 input steps, not 16.
- The scaling of the inputs (the official global z-score) belongs to the run's data pipeline; metrics are the run's.
- Initialization follows the official training script: Xavier-uniform for every multi-dimensional parameter and `U(0, 1)` for every vector parameter, applied to all parameters.
- Defaults are the paper's (`K = 17`, `d_z = 24`, 4 layers, kernel 2, `m = 8`, `d_m = 2 d_z = 48`, `lambda = 1`); the official optimizer (Adam, lr 1e-4, batch 8, gradient clipping 10, early stopping 7) belongs to the run configuration.
- Preset: the NYC layout of 98 locations x 4 sources, 16 steps in and 3 out. The shipped `data/NYC.h5` differs from the paper's Table 2 (half-hour MoSSL data, not the stated hourly 2015-2016 span).
- No reference comparison was run: the official code cannot train as shipped (`lam` undefined).

## Verification

Checked properties: the source-major layout; the receptive-length constraint; the GMRE mixture parameters and Cluster Norm against an explicit per-scalar Bayes derivation; the Eq. 7 objective; the gated source-spanning convolution and channel fold; HRA memory attention and predictor; `aux_loss` exists only in training and reaches every parameter.

## Citation

```bibtex
@inproceedings{deng2023gmrl,
  title     = {Learning {G}aussian Mixture Representations for Tensor Time Series Forecasting},
  author    = {Deng, Jiewen and Deng, Jinliang and Jiang, Renhe and Song, Xuan},
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, {IJCAI-23}},
  year      = {2023}
}
```
