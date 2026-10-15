# StaTS — reference

## Differences in detail

- Sources read at revision `5f0b50ee2aca397c4abf08ad4458d950e33a151d`: `src/models/StaTS.py`, `src/experiments/StaTS.py`, `src/experiments/prob_forecast.py`, `src/experiments/Setup.py`, `scripts/StaTS/*.sh`, `requirements.txt`; paper Sections 3-4 and Appendices A-C (arXiv v1).
- Inputs and outputs: `x_enc` is `[B, seq_len, enc_in]`; marks and `x_dec` are ignored (the official denoiser never reads them). The output is `[B, pred_len, enc_in, K]`, the linear-interpolated empirical quantiles of `num_samples` DDIM trajectories at `evaluation.quantile_levels` (or `quantile_levels`). The diffusion is joint over all `enc_in` channels; for `MS` the runner selects the target channel.
- Training: the runner calls `pretrain` once (the `pretraining-stage` capability) for stage I with two internal Adam optimizers (learning rate `1e-3`, weight decay `5e-4`; gradient-norm clipping 5 on the denoiser epochs), then trains the denoiser with the configured optimizer through `training_objective` (stage II, `L_obj`). The official stage-II epochs continue with the stage-I denoiser optimizer state.
- Resolved from the official code: instance statistics are the history mean (detached) and biased standard deviation with a `1e-5` variance floor, applied to history and target and inverted on the output; the corrupted history uses the same `alpha_bar_t` with independent noise.
- The anchor's frequency gate `sigmoid(a log(1 + mean_ch |C|) + b)` starts at 0.5 (zero `a`, `b`) and the fusion weight at 0.5; the distortion ratio has a `1e-4` floor and `[-10, 10]` clipping; FiLM tables have `T + 1` rows; the DDIM noise floor is `1e-8`.
- Code-only values: hidden width 256, bottleneck `max(8, N // 4)`, the FiLM table form and `U(-0.02, 0.02)` initialization, the single `Linear(256, 256)` refinement, and `0.02 randn` complex gains.
- STS objective: barrier (Eq. 6), terminal spectral KL (Eqs. 7-8), flatness progression toward `(1 - gamma_t) SF(x0) + gamma_t SF(x_T)` (Eqs. 9-11, 69), smoothness and initial-step penalties (Eq. 12), weights `lambda_smooth = 5`, `lambda_init = 0.5`, `lambda_end = 0.5`, `lambda_bar = 5e-3`, `lambda_prog = 0.5`, `lambda_obj = 0.01`.
- Only the linear schedule initialization is provided (the code's alternatives are unused by the scripts).
- The official validation and test use reduced `torch_timeseries` fast loaders; TSFLab evaluates its own test windows. Per-dataset official scripts use other batch sizes and epochs than Sec. 4.1; the preset follows the paper's ETTh1 setting.
- Reported benchmark numbers are not reproduction claims of this implementation.

## Citation

```bibtex
@article{zhang2026stats,
  title   = {StaTS: Spectral Trajectory Schedule Learning for Adaptive Time Series Forecasting with Frequency Guided Denoiser},
  author  = {Zhang, Jintao and Liu, Zirui and Cheng, Mingyue and Wang, Xianquan and Liu, Zhiding and Liu, Qi},
  journal = {arXiv preprint arXiv:2603.00037},
  year    = {2026}
}
```
