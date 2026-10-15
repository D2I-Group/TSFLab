---
name: "APN"
description: "Learns soft temporal window boundaries per channel to aggregate observations into patches, then decodes at query times. Use for a light channel-independent baseline built for irregular multivariate series; not for ragged or missing-value protocols (TSFLab feeds dense regular windows)."
---

# APN

## Idea

- Time-Aware Patch Aggregation: `patch_weights` builds differentiable windows from per-channel learned offsets, log-widths and temperatures (product of sigmoids), so patch boundaries adapt.
- Observations are augmented with a learned time embedding and aggregated by normalized weighted averaging into `num_patches` patches per channel.
- A learned query per channel pools the patches with attention into one context vector.
- A shallow MLP decodes the context concatenated with future-time embeddings, one step per horizon.

## When to use

- Designed for irregular multivariate series (healthcare, climate, astronomy) where fixed patches misalign with observation times, and for tight compute budgets (shallow model).
- In TSFLab the dense contract uses regular timestamps unless observation times are passed, and no missing-value mask is applied, so the irregular-sampling advantage is not exercised on standard benchmarks.
- Channel-independent; point output only.

## Configure

- `enc_in`: number of channels.
- No period- or length-specific constraint; `num_patches` is a generic capacity choice.

Other hyperparameters: preset defaults in `configs/models/APN.toml`; tune generically.

## Differences

Local implementation from paper Eqs. (2)-(10) after inspecting the official `models/APN.py` at the pinned revision; nothing copied (no license file). It implements learned soft temporal windows, normalized time-aware aggregation, channel queries, and a query-time MLP decoder. It does not reproduce APN's asynchronous ragged-data loader or missing-value benchmark protocol.
