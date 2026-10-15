---
name: "ReFocus"
description: "Frequency-domain MLP that attenuates the low-frequency trend band, maps spectra with dense complex linear layers, and shares key frequencies across channels by energy-based pooling. Use for multivariate forecasting where mid-frequency structure and cross-channel key frequencies matter; not for univariate series."
---

# ReFocus

## Idea

- AMEO: `x - beta * moving_average(x)` (`EdgePaddedMovingAverage` from `series_decomposition`) attenuates the low-frequency trend band before encoding, reinforcing mid frequencies.
- `FLinear` is a dense complex linear map from every input frequency bin to every output bin (no low-pass truncation), used for the embedding, every encoder sub-layer, and the output `projection`.
- `EncoderBlock` (EKPB) picks one channel's spectrum per frequency bin from a softmax energy distribution across channels (`EnergyBasedFrequencyPooling`), then fuses it with each channel's own features.
- `revin` wraps the model; the output projection starts near identity when `initial=True`.

## When to use

- Multivariate data where channels share informative frequencies: EKPB passes each bin's key-channel spectrum to all channels.
- Series whose low-frequency trend would otherwise dominate the spectrum, hiding mid-frequency structure.
- Not for univariate data (the cross-channel pick is trivial) or when the trend itself is the main signal.

## Configure

- `enc_in` follows the channel count: must equal the number of input channels.
- `kernel_size` follows `seq_len`: a positive odd moving-average window no longer than `seq_len`.

Other hyperparameters: preset defaults in `configs/models/ReFocus.toml`; tune generically.

## Differences

Independent rewrite from the paper after inspecting `models/LiNo.py`, `layers/FLinear.py`, `layers/Encoder.py`, and `layers/RevIN.py` at revision `5b883b29f364b52a73835f1465e993433f94a1ed`; the repository has no license file and no source was copied.

- `FLinear` stays model-local: its dense complex bin-to-bin map is not equivalent to any cataloged frequency-interpolation component.
- Key-Frequency Enhanced Training (KET, Sec. 3.4), a channel mix-up augmentation of inputs and targets across alternating epochs, is a training-loop strategy and is not implemented; only the AMEO and EKPB blocks are.
- `EnergyBasedFrequencyPooling` samples with `multinomial` in training as the official code does, but picks the arg-max in evaluation so inference is reproducible (the official code samples always; the paper specifies only training).
- The AMEO filter reuses `series_decomposition.EdgePaddedMovingAverage`, which for odd `kernel_size` equals the official replicate-padded fixed-weight depthwise `Conv1d`.

Cite: "ReFocus: Reinforcing Mid-Frequency and Key-Frequency Modeling for Multivariate Time Series Forecasting", arXiv:2502.16890 (2025).
