---
name: "LightTS"
description: "Light MLP forecaster over continuous and interval down-sampled views with bottleneck information-exchange blocks and a linear highway. Use for fast multivariate long-horizon forecasting under tight compute; not when seq_len cannot be split into equal chunks."
---

# LightTS

## Idea

- Each series is reshaped two ways: contiguous non-overlapping chunks (`sample_continuous`) and strided interval samples (`sample_interval`), since down-sampling preserves most of a series' information.
- `InformationExchangeBlock` projects over time through a bottleneck, adds an identity-initialized channel projection, and maps to output rows; the two sampling views use separate blocks and are summarized by linear layers.
- A final `forecast_block` exchanges information across the variables of the concatenated features, and a `highway` linear layer from the raw window is added to the output.

## When to use

- Designed for efficient multivariate forecasting on large datasets: the paper reports under 5% of the FLOPs of earlier state-of-the-art models.
- Continuous sampling captures short-term local patterns and interval sampling long-term ones, which suits long lookbacks and horizons.
- Mixes variables in the final block; there is no instance normalization, so strongly shifting levels need external scaling.

## Configure

- `enc_in`: must equal the channel count.
- `chunk_size`: must divide `seq_len` (`seq_len % chunk_size == 0`).

Other hyperparameters: preset defaults in `configs/models/LightTS.toml`; tune generically.

## Differences

- Clean-room implementation: continuous and interval sampling map Equations 1-2 for divisible lengths; the three IEBlocks map the bottleneck information-exchange procedure.
- The official repository has no license file (`NOASSERTION`) and was used as reference only; nothing was copied.
- Published-metric reference comparison is not claimed.
