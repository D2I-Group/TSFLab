---
name: "LSTMForecasterTS"
description: "LSTM encoder over the RevIN-normalized multichannel window with a direct linear decode of its final hidden state. Use as a simple recurrent baseline on few-channel multivariate data; not for many channels (output head grows with pred_len * channels) or long lookbacks."
---

# LSTMForecasterTS

## Idea

- An `nn.LSTM` consumes all channels jointly at every step, so channels are mixed in the recurrent state.
- The last layer's final hidden state goes through one linear `head` to `pred_len * enc_in` values (direct multi-horizon, no autoregression).
- `revin` normalizes the input and denormalizes the forecast (`use_revin`).

## When to use

- A recurrent baseline for multivariate forecasting where a compact sequential model is the comparison point.
- RevIN handles level shifts between windows.
- Channels share one recurrent state, which suits a few related channels; with hundreds of channels the joint input and the `pred_len * enc_in` head become a bottleneck.
- Sequential recurrence over every step makes long lookbacks slow.

## Configure

- `enc_in`: must equal the channel count (LSTM input width and head output `pred_len * enc_in`).

Other hyperparameters: preset defaults in `configs/models/LSTMForecasterTS.toml`; tune generically.

## Differences

- Clean-room implementation from the published LSTM gate equations and the repository tensor contract; no external source was copied.
- The 1997 paper does not define the direct multi-horizon head, joint-channel setup, or optional RevIN; these are TSFLab choices, so no experimental reference comparison is claimed.
- Citation: S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory", Neural Computation 9(8):1735-1780, 1997, doi:10.1162/neco.1997.9.8.1735.
