# VLBM — reference

## Paper

VLBM: Variational Latent Basis Modeling for OOD Robust Multivariate Time Series Forecasting (arXiv 2606.02138, 2026). Proofs, implementation details, and complete results are referred to a supplementary material not in arXiv v1 or the repository.

```bibtex
@article{zhang2026vlbm,
  title   = {VLBM: Variational Latent Basis Modeling for OOD Robust Multivariate Time Series Forecasting},
  author  = {Zhang, Xudong and Lei, Jierui and Li, Jiacheng and Shen, Lingdong and Cui, Jian and Tang, Haina},
  journal = {arXiv preprint arXiv:2606.02138},
  year    = {2026}
}
```

## Implementation mapping

Independent rewrite from Section IV (Eqs. 17-25, 28-33) and Section V-A after reading the pinned official code (`leijieruilq/VLBM_OOD_forecast` at `4786753f`, MIT): `models/VLBM.py`, `exp/vlbm_loss.py`, `exp/exp_long_term_forecasting_vlbm.py`, `exp/exp_ood_forecasting_vlbm.py`, `data_provider/data_loader.py`, `run_vlbm.py`, and `scripts/**/VLBM.sh`. Nothing was copied or imported.

Resolved from the official code:

- Instance normalization: mean and `sqrt(var + 1e-5)` (population variance) over the history per variable.
- Value projection, fusion, and block layers are `1x1` convolutions, implemented as equivalent linear maps with the same default initialization.
- Embedding concatenates `[time, day, value, node]` with Xavier-initialized tables of sizes `int(1440 / interval)`, 7, and `N`, using the last step's slots.
- Encoders: `ReLU` projections, `LeakyReLU(0.2)` scores, `-1e9` masking, no sampling for the posterior. The basis is Xavier-initialized.
- Residual and base blocks use dropout 0.15; the latent graph uses cosine similarity, topology weight 0.2, top-`k` (`--heads`) per row, softmax over kept entries.
- Loss: MAE plus `0.1` KL (sum over the basis, mean over batch and variables); Adam with learning rate halving each epoch.
- Preset: `scripts/long_term_forecast/Weather_script/VLBM.sh` for horizon 96 (`seq_len = 96`, `hiddens = 128`, `num_Period_Block = 1`, `m = 100`, `heads = 1`, `interval = 10`, `use_norm = 1`, learning rate `1e-3`, batch 16). Spec defaults are the `run_vlbm.py` parser defaults.

## Differences in detail

- Projection (Eqs. 26-28, 30): the paper uses `P_B = B^T (B B^T + eta I)^-1 B` with base input `[Z, X P_B]` and residual input `[X (I - P_B), X]`; the code (followed) has no projection, and `eta` is unspecified.
- Shared basis: the paper shares `B` between posterior and prior; the code gives each encoder its own, but the posterior's sampled `z_q` is never used, so its basis gets no gradient. One basis (the prior's) is used.
- Topology: the paper masks attention with the physical topology and uses it as `A_topo` (Eq. 31); the code always uses `torch.eye(N)`, so each encoder returns `ReLU(Conv_v(V))_i` and query/score layers get no gradient. The identity is used; a dataset adjacency is ignored.
- Heads (Eq. 32): the paper predicts `Y_base + R_res` separately; the code uses one `1x1` convolution on the summed streams and leaves `base_pred`, `res_pred`, `prediction1`, `prediction2`, `pos_emb`, `conv2`, `score_proj` unused. These are omitted.
- Loss: `masked_mae_loss(Y, Y_pred)` swaps arguments, so training is plain MAE while validation masks zero targets. The configured criterion is used unmasked. `beta` (Eq. 33) is hard-coded 0.1 in the code: `kl_weight = 0.1`.
- `--use_norm` is not in the paper and the posterior's future window is embedded without it; kept as in the code (`use_norm`).
- Calendar: the official loaders index time-of-day and weekday by row position from the first row, and the Flight script's `--interval 5` on hourly data wraps slots every 12 days. Here slots come from the last step's marks (raw marks, or the two normalized calendar covariates of spatio-temporal datasets; slot 0 without marks) and `interval` must be set to the sampling interval.
- The posterior runs through `training_objective` with the future window and marks; `forward` evaluates only the prior path. `top_k <= enc_in` is validated at construction (the official `torch.topk` fails at run time). MS targets use the last forecast channel while the posterior embeds every variable's future.
- The official `inverse_transform` raises `NameError`; not applicable here. The README's `requirements.txt` and `scripts/ood_forecast/pemsbay0/VLBM.sh` are missing upstream; only PyTorch is needed.
