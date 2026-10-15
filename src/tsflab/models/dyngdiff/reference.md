# DynGDiff — reference

## Implementation mapping

- `S4NoiseBackbone` (App. B.1) on `[B, D, seq_len + pred_len]`: linear+ReLU
  input map, sinusoidal step embedding through a two-layer SiLU MLP,
  `S4ResidualBlock`s (step added, pre-LayerNorm bidirectional `S4` residual,
  `tanh * sigmoid` gate, residual and skip 1x1 heads), global skip sum, MLP head
  and the input added back. `NPLRKernel`: HiPPO-LegS diagonal-plus-low-rank
  (`hippo_legs_nplr`), bilinear discretization, conjugate-pair Cauchy sums with
  a rank-one Woodbury correction.
- `StateAwarePolicyNetwork` (Sec. 4.1, Fig. 2): Conv1d/BatchNorm/SiLU layers
  with a fused step embedding output `Z_t` (Eq. 6); `policy_loss` regresses it
  onto `proxy_precision_target` (Eqs. 8-9).
- `Model.guidance_score` / `reverse_step` (Sec. 4.2): detached `A_t` weights
  `rho_kappa(y - x0_hat)` over the observed context (Eq. 13); the mean shift is
  `s sigma_t^2 grad log p(y_obs | x_t)`. `Model.sample` uses
  `kappa_s = s / (S + 1)` and rescales trajectories.

## Differences in detail

- Sources read: `src/dyng_diff/arch/{backbones,s4,policy_net}.py`,
  `src/dyng_diff/model/diffusion/{_base,dyng_diff}.py`,
  `src/dyng_diff/sampler/observation_guidance.py`, `src/dyng_diff/configs.py`,
  `src/dyng_diff/utils.py`, `scripts/{train_backbone,train_weight_policy,guidance_experiment,prepare_custom_dataset}.py`, `configs/`.
- Inputs: `x_enc` only; marks and `x_dec` are ignored (official runs use no
  covariates or lags). Quantiles are linear-interpolated at
  `evaluation.quantile_levels` (or `quantile_levels`). For `MS` the runner
  selects the target channel.
- Training: `pretrain` runs once (the `pretraining-stage` capability) with Adam
  at `backbone_lr`, gradient-norm clipping 1.0, ReduceLROnPlateau (factor 0.5,
  patience 10) on the epoch-mean loss, as in the official Lightning module; the
  policy then trains through `training_objective` with the configured optimizer
  (official: Adam `1e-3`, 30 epochs of 100 batches, AMP).
- Resolved from the official code: windows scaled by the per-variable mean
  absolute context value plus `1e-5`, clipped to `[-10, 10]`; linear beta
  schedule `1e-4` to `0.1` over 100 steps (Table 2); sin-then-cos step embedding
  with `log(10000) / (half - 1)` frequencies; every step guided
  (`stop_guidance_ratio = 0`); guidance energy summed over observed positions
  with the context-only mask; DDPM posterior standard deviation as reverse noise
  and no noise or guidance at the final step; guidance added to the DDPM mean as
  in the released DDPM sampler (Eq. 5 form exists only in the unused DDIM code);
  Eq. 9 statistics per sample over variables and time; policy widths (hidden 64,
  step embedding 32), kernels 3 and 1, BatchNorm after the first convolution.
- Preset: ETTh1 (Table 3, `configs/guidance/eval_etth1.yaml`): hidden 64, 3 S4
  blocks, guidance scale 3, 100 samples.
- Other: policy weights reuse the backbone pass of the guidance gradient (the
  official sampler runs it three times per step with identical results); the
  unused covariate encoder (`S4Block.feature_encoder`) is omitted; trajectories
  are drawn in chunks of `sample_batch_size`.
- Official evaluation leaks training windows into the test set; TSFLab uses its
  own held-out split and metrics. Reported benchmark numbers are not
  reproduction claims of this implementation.

## Citation

Zhang, Z., Ni, Z., Fan, W. "DynG-Diff: A State-Aware Dynamic Guidance Diffusion Framework for Probabilistic Time Series Forecasting." arXiv:2609.02068 (2026).
