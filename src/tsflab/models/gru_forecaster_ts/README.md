---
name: "GRUForecasterTS"
description: "Baseline GRU encoder over the normalized window with a direct linear decode of its final hidden state to the horizon. Use as a small recurrent, channel-mixing reference on short or modest data; not for long lookbacks, long horizons, or many channels."
---

# GRUForecasterTS

## Idea

- An `nn.GRU` reads all channels jointly at each step, so channels are mixed in the recurrent state.
- The last layer's final state is mapped by one linear head to `pred_len * enc_in` values (direct multi-horizon, no autoregressive decoding).
- `revin` normalizes the input and denormalizes the forecast (`use_revin`).

## When to use

- A small recurrent baseline (one layer, 64 units by default) for short training sets or as a reference point against larger models.
- Mixes channels in one hidden state, which helps only when channels are correlated.
- Not for long lookbacks (sequential recurrence, everything compressed into one final state), long horizons, or many channels (the head grows as `pred_len * enc_in`).
- Point output only.

## Configure

- `enc_in`: the dataset's channel count (sets the GRU input width and the head output `pred_len * enc_in`).

Other hyperparameters: preset defaults in `configs/models/GRUForecasterTS.toml`; tune generically.

## Differences

- Clean-room implementation from the cited GRU equations and the repository tensor contract; no external implementation was copied.
- The paper (Chung et al., 2014, arXiv:1412.3555) compares gated recurrent units (GRU, LSTM) on polyphonic music and speech modeling and finds GRU comparable to LSTM; it does not define this final-state direct multi-horizon head or the optional RevIN, so no paper-result comparison is claimed.
