---
name: "MLPForecasterTS"
description: "Per-channel MLP that maps the lag window straight to the horizon over the time axis, wrapped in RevIN. Use as a cheap direct multistep baseline for multivariate forecasting with weakly related channels; not for cross-channel interaction, exogenous or calendar inputs, or probabilistic output."
---

# MLPForecasterTS

## Idea

- Transposes to `[batch, channels, seq_len]` so one `nn.Sequential` MLP (Linear, GELU, Dropout blocks) acts over time and is shared by every channel.
- Ends in a single `nn.Linear(d_model, pred_len)` that emits the whole multistep forecast directly (no recursion).
- Reversible instance normalization (`revin`) normalizes inputs and de-normalizes the forecast; `use_revin=False` disables it.

## When to use

- A cheap, fast baseline for direct multistep forecasting: a few dense layers over the lag window, low compute and memory.
- Multivariate data whose channels are weakly related: one shared per-channel network, no channel mixing.
- Not when cross-channel interaction carries the signal, when exogenous covariates or calendar marks matter (marks are ignored), or when quantiles or distributions are needed (point output only).

## Configure

- `enc_in`: number of input channels; must equal the dataset's channel count (validated against the input shape).

Other hyperparameters: preset defaults in `configs/models/MLPForecasterTS.toml`; tune generically.

## Differences

- No official code; the cited paper (Rumelhart, Hinton, Williams, Nature 323(6088):533-536, 1986, doi:10.1038/323533a0) supplies only back-propagation and feed-forward concepts, not a time-series architecture.
- The shared channel-wise lag mapping, GELU activation, direct horizon head, and optional RevIN are disclosed local choices.
