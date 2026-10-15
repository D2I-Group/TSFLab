---
name: "TCNForecasterTS"
description: "Generic TCN baseline: dilated causal residual convolutions over RevIN-normalized channels with a last-step direct multistep head. Use as a lightweight convolutional baseline on short-to-medium lookbacks with correlated channels; not for long lookbacks at default depth or probabilistic output."
---

# TCNForecasterTS

## Idea

- Stacks `TemporalResidualBlock`s of two kernel-3 `CausalConv1d` layers with exponentially growing dilation (1, 2, 4, ...) and residual connections.
- All channels enter the first convolution together, so channels are mixed from the start (no channel independence).
- A linear head on the final time step emits the whole horizon for every channel at once (direct multistep); `revin` wraps the model and `aux_loss` is zero.

## When to use

- A simple, fast convolutional reference point; Bai et al. argue convolutions are a natural starting point for sequence modelling, with longer effective memory than LSTMs.
- Few, correlated channels (channel mixing in the first layer); the head grows as `d_model * pred_len * enc_in`, so many channels with long horizons get expensive.
- Not for very long lookbacks unless `num_layers` is raised (the head only reads the last step's receptive field), or for probabilistic output.

## Configure

- `enc_in`: number of channels; input must be `[B, seq_len, enc_in]`.
- `num_layers`: the last step sees `1 + 4 * (2^num_layers - 1)` steps (13 with the default 2); raise it until this covers the relevant lookback (`seq_len` or the dominant period).

Other hyperparameters: preset defaults in `configs/models/TCNForecasterTS.toml`; tune generically.

## Differences

- Clean-room design from the causal, dilated residual architecture of Bai et al.; no external implementation copied.
- Omits the paper's weight normalization, uses a final-timestep direct horizon head, and optionally applies RevIN, so no paper-result comparison is claimed.
