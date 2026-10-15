# ARMD — reference

## Citation

```bibtex
@inproceedings{gao2025armd,
  title     = {Auto-Regressive Moving Diffusion Models for Time Series Forecasting},
  author    = {Jiaxin Gao and Qinglong Cao and Yuntian Chen},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {39},
  number    = {16},
  pages     = {16727--16735},
  year      = {2025},
  doi       = {10.1609/aaai.v39i16.33838},
  url       = {https://arxiv.org/abs/2412.09328}
}
```

## Differences in detail

Independent implementation; no official code was copied (checked `Models/autoregressive_diffusion/armd.py`, `Models/autoregressive_diffusion/linear.py`, `engine/solver.py`, `main.py`, the configs, and the paper's Eq. 1-10). Inputs are `[B, seq_len, enc_in]` with `seq_len >= pred_len`; marks and decoder inputs are ignored; the output is the `[B, pred_len, enc_in]` point forecast (deterministic, so no distribution output is declared). No shared component matches: the schedules, the devolution network and the sampler are model-local (`Model.intermediate_state`, `Model.devolve` with `step_weight`, `Model.training_loss`).

- **History length.** The paper and official code use history length equal to the horizon T (a 2T window); the model takes the last `pred_len` steps of a longer history (the official sampler slices the first `pred_len` steps of a 2T window, which is the history for `seq_len == pred_len`).
- **Diffusion length.** Official code hard-codes 96 steps in the network schedules and a global `pred_len = 96`; here the number of diffusion steps and every schedule length equal `pred_len`, as the paper states.
- **Input deviation.** The official network adds the training-time deviation in place on its input, a view of the training window, so the noise also leaks into the loss target and the intermediate state; here it is applied to the network input only and the trend losses use the clean state and future (Eq. 3, 6).
- **Loss.** The official per-step weight `sqrt(alpha) sqrt(1 - abar) / beta / 100`, omitted by Eq. 7, is reproduced. One diffusion step is drawn per batch, as officially. `loss_type` `l1` or `l2` and `w_grad` follow the official configs (`w_grad=False` is used for ETTm1).
- **Sampling.** Only the deterministic DDIM update of Eq. 9-10 with `sampling_steps` steps is provided. Official code uses it when `sampling_timesteps < timesteps` and otherwise a stochastic ancestral sampler with clamping to `[-1, 1]`, not reproduced; the official configs use 1 or 2 steps.
- **Data and training.** Official data are min-max scaled to `[-1, 1]` over 2T windows with EMA weights (decay 0.995), gradient clipping and a plateau scheduler; here the repository's standard scaling, trainer and optimiser apply and there is no EMA.
