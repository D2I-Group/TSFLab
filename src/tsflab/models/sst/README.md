---
name: "SST"
description: "Hybrid experts: a Mamba encoder on coarse long-range patches and a local-window Transformer on fine recent patches, fused by a learned router under RevIN. Use for long lookbacks that mix slow global patterns with recent local variation; not for short windows or cross-channel dependencies."
---

# SST

## Idea

- `patchify` / `pts_resolution` (Definition 4.1, Eq. 6): the full window `L` is cut into long, widely strided patches (default `P=48`, `Str=16`, low resolution) and the latest `short_len = S` steps (default `L / 2`) into short, densely strided patches (`16`, `8`), each end-padded by one stride.
- `encode_long` (patterns expert, Sec. 4.2): linear patch embedding, then Mamba layers (cataloged `mamba`, expansion 2) each followed by a feed-forward network, no positional encoding.
- `encode_short` (variations expert, Fig. 6): linear patch embedding plus learnable positions, then post-norm BatchNorm Transformer layers with attention restricted to `|i - j| <= local_ws // 2`, carrying pre-softmax scores across layers.
- `LongShortRouter` (Sec. 4.3) projects the normalized input over variates to `d_model`, flattens over time, and outputs softmax weights `(p_L, p_S)` per sample.
- `forward`: per variate, the flattened long and short embeddings are scaled by `p_L`, `p_S`, concatenated, and mapped to the horizon by one linear head; non-affine RevIN wraps the model.

## When to use

- Long lookbacks (the paper uses `L = 672`) where coarse long-range patterns and fine recent variation both matter.
- Per-variate experts with a shared head: channel interaction only through the router weights, so suits weakly correlated channels.
- Point forecasts only.

## Configure

- `enc_in` follows the dataset channel count; it must equal it exactly.
- `m_patch_len` / `m_stride` follow `seq_len`: `m_patch_len <= seq_len`.
- `patch_len` / `stride` follow the short range (`seq_len // 2` by default): `patch_len <= short_len`.
- Other hyperparameters: preset defaults in `configs/models/SST.toml` (official ETTh1 script); tune generically (`d_model` divisible by `n_heads`).

## Differences

Independent rewrite of Sec. 4 after reading `XiongxiaoXu/SST@b39292b` (no license, `NOASSERTION`); nothing copied or imported.

- The local window mask follows Fig. 6; the official `get_local_mask` inverts it (attention only outside the window).
- The selective SSM is the cataloged pure-PyTorch scan, not `mamba_ssm` fused kernels; numbers may differ slightly.
- The short range is a `short_len` model parameter instead of the task's `label_len`.
- Flattened embeddings are patch-major (a fixed permutation of head weights).
- Not implemented: short-range decomposition, additive fusion, `fc_dropout`, per-variate heads (all off in reported scripts).

Full detail in `reference.md`.
