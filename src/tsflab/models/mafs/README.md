---
name: "MAFS"
description: "Multi-agent forecaster: variate-token Transformer agents specialized on horizon prefixes exchange messages over a learnable topology graph, and a voter aggregates them. Use for multivariate data with shifting patterns and correlated variates; not for tight compute (cost scales with agent count)."
---

# MAFS

## Idea

- Each of `num_agents` agents embeds every variate's whole window into a token and runs its own `AgentEncoderLayer` stack (attention across variates).
- After every layer agents exchange messages through a symmetric, self-looped, degree-normalized adjacency built from learnable `edge_logits` masked by a star, ring, chain or fully-connected topology (`normalized_adjacency`).
- A confidence gate blends each agent's state with its neighbourhood context, and an input-conditioned softmax `voter` weights the agents.
- The output is a head on the voted representation plus the voted sum of per-agent heads; `specialization_loss` trains agent k on the first `pred_len * (k + 1) / num_agents` steps (multi-scale prefixes).

## When to use

- Designed for datasets with diverse, evolving temporal patterns where one monolithic model generalizes poorly; agents specialize on different horizon views and cooperate.
- Attention across variate tokens models cross-channel correlation.
- Cost grows linearly with `num_agents`; not a choice for tight compute budgets.
- No instance normalization; scale the data externally if levels drift.

## Configure

- `enc_in`: must equal the channel count (one variate token per channel per agent).

Other hyperparameters: preset defaults in `configs/models/MAFS.toml`; tune generically.

## Differences

- Local rewrite; Eqs. (4)-(7) were checked against the pinned `Agent_iTrans_Cooperation.py`, nothing copied.
- Training is single-stage: the objective sums the configured criterion and `specialization_loss`, a joint approximation of the paper's ten-epoch specialization stage followed by frozen-agent collaboration (no staged schedule or freezing).
- Four agents and a star topology are the compact defaults.
