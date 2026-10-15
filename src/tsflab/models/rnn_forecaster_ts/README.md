---
name: "RNNForecasterTS"
description: "Classical baseline: an Elman (tanh) RNN reads all channels jointly and a linear head maps its final state directly to the full multichannel horizon, inside RevIN. Use as a recurrent reference on small problems with short lookbacks; not for long histories, where plain RNNs suffer vanishing gradients."
---

# RNNForecasterTS

## Idea

- `nn.RNN` (tanh, `h_t = tanh(W_h h_{t-1} + W_x x_t + b)`) consumes all channels at each step, so channels are mixed in the hidden state.
- `head` maps the last layer's final state straight to `pred_len * enc_in` values (direct multistep, no decoding recursion).
- `revin` normalizes inputs and de-normalizes outputs; `aux_loss` is zero.

## When to use

- A recurrent baseline next to linear and gated models, with small capacity suited to short training sets.
- Short lookbacks: the vanishing-gradient problem of simple RNNs over long histories motivates gated variants (LSTM, GRU).
- Channels interact only through one shared hidden state; for many channels the head (`pred_len * enc_in` outputs) grows large.

## Configure

- `enc_in` follows the channel count: must equal the number of input channels.

Other hyperparameters: preset defaults in `configs/models/RNNForecasterTS.toml`; tune generically.

## Differences

Clean-room implementation from Elman's published recurrence; no external implementation source was copied.

- Elman (1990) does not define the direct multi-horizon head, RevIN, or this joint multivariate forecasting setup, so no experimental reference comparison is claimed.

Cite: Jeffrey L. Elman, "Finding Structure in Time", Cognitive Science 14(2):179-211, 1990, doi:10.1207/s15516709cog1402_1.
