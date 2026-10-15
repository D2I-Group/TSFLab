# TALON — reference

## Paper

Adapting LLMs to Time Series Forecasting via Temporal Heterogeneity Modeling and Representation
Alignment (Sun et al., arXiv:2508.07195, 2025; revised v2 2026-08).

## Implementation mapping

- `PatchComplexity`: cues are computed on the dataset-scaled input; STL is the non-robust procedure (Cleveland et al., 1990) as fixed linear operators (`stl_operators`, period `max(S // 2, 2)`), with the defaults of the `statsmodels` implementation the official code calls (seasonal 7, default trend and low-pass windows, 5 inner and 0 robustness iterations, descriptors rounded to three decimals).
- Gate: latent scores `ReLU(s W0t) W1t` (Eq. 1) plus, in training, Gaussian noise scaled by `Softplus(ReLU(c W0c) W1c)` (Eqs. 2-3), projected by `W_H` (Eq. 4), top-k softmax (Eqs. 5-6).
- Released GPT-2 weights come from the pinned `gpt2` artifact.

## Differences in detail

- Official code read at `syrGitHub/TALON` revision `11bd7b2b8ae7aed5ade6a4e7fff7d0f48b66ae83`: `models/TALON_GPT2.py`, `models/Preprocess.py`, `layers/expert_extractor_cluster.py`, `layers/encoders.py`, `layers/pattern_extractor.py`, `layers/mlp.py`, `data_provider/complexity.py`, `data_provider/data_loader.py`, `exp/exp_long_term_forecasting.py`, `preprocess.py`, `run.py`, `scripts/time_series_forecasting/long_term/ETTh1.sh`.
- Resolved from the official code: instance normalization per channel (`revin`, no affine); routing cues from the dataset-scaled segments; bias-free two-layer gate encoders with hidden width `expert_hidden`; noise scale `Softplus(.) + 0.01`; `W_H` initialized to the identity; top-k renormalization with `1e-6`; balance loss `alpha * (cv^2(importance) + cv^2(load))` with unbiased variance; CNN expert `Conv1d(1, h, 3, padding 1) -> ReLU -> Conv1d(h, 1, 1) -> Linear(S, 768)`; LSTM expert one layer of width `expert_hidden` over the segment's steps, last state projected to 768; next-segment training target (history shifted by one segment plus the first future segment).
- A constant segment gets zero trend strength and autocorrelation (the official `var_deseasonal == 0` and `std == 0` branches), decided on the segment itself because the linear STL operators leave rounding noise in its deseasonalized variance.
- Preset defaults follow `ETTh1.sh`: segment 96, `top_k = 3`, hidden 1024, tanh MLP of width 1024, `alpha = 0.02`, dropout 0.1. The script's `--mlp_hidden_layers 1` builds two Linear layers, which is `mlp_hidden_layers = 2` here (1 is rejected).
- The alignment module needs the GPT-2 tokenizer and calendar date strings that TSFLab's numeric marks do not carry, and the official pipeline pairs prompts with the wrong segments (see `issues`).
- The released `openai-community/gpt2` `model.safetensors` (revision `607a30d`, SHA-256 pinned) is a required artifact fetched explicitly (`tsf model artifacts TALON --fetch gpt2`); without it `spec.build` constructs a randomly initialized trunk for offline structure tests only.
- The training objective uses the configured TSFLab loss on the next-segment targets (the paper uses MSE); with `features = "MS"` only the target channel is used.
- Reported benchmark numbers are not reproduction claims of this implementation.

## Citation

```bibtex
@misc{sun2025talon,
  title         = {Adapting {LLMs} to Time Series Forecasting via Temporal Heterogeneity Modeling and Representation Alignment},
  author        = {Sun, Yanru and Eldele, Emadeldeen and Xie, Zongxia and Wang, Yucheng and Niu, Wenzhe and Hu, Qinghua and Kwoh, Chee Keong and Wu, Min},
  year          = {2025},
  eprint        = {2508.07195},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```
