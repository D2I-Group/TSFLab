---
name: "DSFormer"
description: "Double sampling (piecewise and interval views) with temporal and variable attention, gated fusion and an attention decoder. Use for long-term multivariate forecasting where both local/global temporal structure and cross-variable correlation matter; not for very many channels or short lookbacks."
---

# DSFormer

## Idea

- `dual_sampling` reshapes each series into a piecewise view (contiguous segments, local information) and an interval view (every `num_samp`-th step, global information).
- `TVABlock` runs temporal attention within each channel and variable attention across channels, then blends them with a sigmoid cross gate and a feed-forward layer.
- The two views are processed by separate block stacks, concatenated and mixed linearly (`node_mix`), refined by `decoder_attention`, and mapped to the horizon; `revin` wraps the model.

## When to use

- Designed for multivariate long-term prediction that needs three features at once: global information, local information, and variable correlation (paper motivation).
- Variable attention mixes all channels; cost grows with channel count, and it helps little when channels are nearly independent.
- Each view token has width `seq_len / num_samp`; the lookback must be long enough to split.

## Configure

- `enc_in`: number of channels.
- `num_samp`: `seq_len` must be divisible by `num_samp`.
- `muti_head`: both `seq_len` and `seq_len / num_samp` must be divisible by it.

Other hyperparameters: preset defaults in `configs/models/DSFormer.toml`; tune generically.

## Differences

Independent rewrite for TSFLab; `main_model.py`, `block/TVA_block.py` and `block/decoder_block.py` of `GestaltCogTeam/DSformer` at `ccdbc354` were inspected as reference only (no license file, recorded `NOASSERTION`); nothing copied.

- Structure: dual sampling, temporal attention, variable attention, gated cross-view fusion, channel decoder, RevIN restoration.
- `if_node = false` averages the two views instead of the learned `node_mix`.
- Calendar marks and decoder inputs are not used.

Citation: Yu, C., Wang, F., Shao, Z., Sun, T., Wu, L., Xu, Y. "DSformer: A Double Sampling Transformer for Multivariate Time Series Long-term Prediction." CIKM 2023, pp. 3062-3072. doi:10.1145/3583780.3614851.
