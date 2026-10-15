# STDMAE — reference

## Differences in detail

- Checked official files: `stdmae/stdmae_arch/stdmae.py`, `mask/mask.py`, `mask/patch.py`, `mask/positional_encoding.py`, `mask/transformer_layers.py`, `graphwavenet/model.py`, `stdmae_runner/stdmae_runner.py`, `stdmae_data/forecasting_dataset.py`, `stdmae/STDMAE_PEMS04.py`; no code copied.
- Inputs are `[B, seq_len, nodes]` values plus calendar marks; `seq_len` is the long-history length (a multiple of `patch_size`, 864 for the official PEMS04 setting) and the short history is its last `history_len` steps (12), as the official dataset builds both windows. Output is `[B, pred_len, nodes]`.
- **Pre-training.** The official forecasting model loads two frozen encoders from separately trained checkpoints. Masked-autoencoder training (random token masking at ratio 0.25 in the released configs, mask tokens, a one-layer decoder, the reconstruction layer and loss) is a different task the runner cannot express, so decoder, mask-token, and reconstruction parameters are absent and released checkpoints are not loaded. `freeze_encoders=True` freezes `temporal_encoder` and `spatial_encoder` (kept in evaluation mode) for externally loaded weights.
- **Encoders.** Patch embedding by a `(patch_size, 1)` strided convolution, sinusoidal 2-D position over (node, patch) added to the tokens, input scaled by `sqrt(embed_dim)`, post-norm ReLU Transformer layers (4 heads, feed-forward 4x, dropout 0.1), final LayerNorm, no masking in the forecasting stage. `sincos_2d` uses interleaved sin/cos, row code then column code. The official runner leaves encoders in training mode under no-grad (dropout active).
- **Backend.** Graph WaveNet with kernel 2, 4 blocks of 2 layers, 32/32/256/512 channels, input value and time-of-day, left padding of one step then padding to the receptive field (13), valid dilated convolutions, per-layer batch norm, summed skips cropped to the shortest length, final length one step. The official always-created but unused residual 1x1 convolutions are omitted. The adaptive adjacency is initialised from standard normals; the predefined supports are the double-transition matrices of the dataset adjacency (identity when none is given).
- **Windows.** The official dataset gives early samples an all-zero long history; the runner's windows start where a full window exists. Official time features are BasicTS scaled values; here calendar features come from the repository's marks (time-of-day, day-of-week), first `in_dim = 2` used.
- **Training recipe.** The official run uses masked MAE with null value 0, curriculum learning over the horizon (6 epochs), gradient clipping 3.0, Adam with weight decay and a multi-step schedule; these are runner configuration here (`masked_mae` is the smoke loss, without a validity mask).

## Citation

```bibtex
@inproceedings{gao2024stdmae,
  title     = {Spatial-Temporal-Decoupled Masked Pre-training for Spatiotemporal Forecasting},
  author    = {Haotian Gao and Renhe Jiang and Zheng Dong and Jinliang Deng and Yuxin Ma and Xuan Song},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence (IJCAI)},
  year      = {2024},
  url       = {https://arxiv.org/abs/2312.00516}
}
```
