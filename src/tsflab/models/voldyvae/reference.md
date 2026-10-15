# VolDyVAE — reference

## Paper

Beyond Static Uncertainty: Modeling Temporal Uncertainty Dynamics for Probabilistic Time Series Forecasting (Yijun Wang, Qiyuan Zhuang, Larysa Marchanka, Xiu-Shen Wei; arXiv 2603.24254v3, 2026). Sources used: Sec. III, Eqs. 1-10, Sec. IV Implementation Details, Sec. V-A, supplementary Table XVI.

```bibtex
@article{wang2026voldyvae,
  title   = {Beyond Static Uncertainty: Modeling Temporal Uncertainty Dynamics for Probabilistic Time Series Forecasting},
  author  = {Wang, Yijun and Zhuang, Qiyuan and Marchanka, Larysa and Wei, Xiu-Shen},
  journal = {arXiv preprint arXiv:2603.24254},
  year    = {2026}
}
```

## Implementation mapping

Independent rewrite after reading the pinned official code (`wangyijunlyy/VolDy-VAE` at `e2d019b7`, no license file, recorded as `NOASSERTION`): `VolDy_VAE_nn.py`, `VolDy_VAE_Forecaster.py`, `VolDy_VAE.yaml`, `run.sh`. The repository imports `MLP` and `RevIN` from `probts.model.nn.prob.k2VAE`, which it does not ship; their semantics come from `probts/model/nn/prob/k2VAE/koopman.py` and `RevIN.py` of `decisionintelligence/K2VAE` at `19c602a8` (MIT). Nothing was copied or imported.

Resolved from the official code:

- RevIN with `eps = 1e-5` and learnable affine (the catalog `revin` has the same arithmetic).
- Patches flatten time and channels (`P*C` features); a length not divisible by `P` is handled by prepending the last `N*P - L` normalized steps.
- MLP depth counts linear layers (`hidden_layers = 3`), ReLU, dropout 0.05 (also in encoder and location MLPs). The latent projection has two layers with hidden width `N*D/2`.
- One-layer GRU; `sigma = Softplus(Linear(h)) + 1e-6` on the patch-flattened output, cut to the horizon. The GRU runs over the `N` past patches then the `M` future ones with the state carried over (the paper text says it runs over the `M` latents).
- NLL clamps `sigma` to `[1e-6, 1e3]` (zero gradient outside) and averages over time and channels; KL sums over latent dimensions and averages over batch and patches.
- Defaults `P = 24`, `D = 128`, `hidden_dim = 256`, `vol_hidden_dim = 128`, `beta = 0.01`, 100 inference samples.

## Differences in detail

- Eqs. (5), (7), (9): the mean is RevIN-inverted but `sigma` is not, so the NLL divides raw-scale residuals by a scale independent of the window's standard deviation. Kept as in paper and code.
- Paper Table XVI gives GRU hidden size and `D = 256`, batch 32; the YAML uses the values above and batch 64 (batch size is a run setting here).
- Unused official keys (`sample_schedule`, `init_koopman`, `init_kalman`, `d_model`, `d_ff`, `e_layers`, `n_heads`, `factor`, `multistep`) are excluded by a strict parameter schema.
- Evaluation: the paper computes CRPS from 100 samples and NMAE from their median or mean (supplementary Sec. B-B); TSFLab returns quantiles at the configured levels.
- Training: the paper uses Adam with learning rate in `[1e-4, 1e-3]`, gradient clipping 0.5, at most 50 epochs; TSFLab uses the catalog optimizer and schedule.
