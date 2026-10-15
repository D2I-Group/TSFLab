---
name: "deviation_memory"
description: "Learnable prototype bank with softmax retrieval and nearest/second-nearest lookup, plus an L1 deviation score between two representations. Use for an auxiliary deviation-from-normal signal with a triplet loss on any encoder state; not for normalized distances, non-gradient updates, or retrieval over real samples."
---

# deviation_memory

## What it does

`PrototypeMemory` holds `M` learnable prototypes `P: [M, d]` and a query projection
`Wq: [query_dim, d]`. For `h: [..., query_dim]`:
`q = h @ Wq`; `s = softmax(q @ P^T, -1)`; `value = s @ P`; the indices of the
top-2 scores give `nearest` and `second_nearest` (rows of `P`).
`deviation_score(current, reference) = sum_D |current - reference|`, one
non-negative scalar per leading position.

## When to use

Use for an auxiliary "how far from the learned normal patterns" signal on any
encoder state, with a triplet-style loss over nearest/second-nearest prototypes.
Do not use when a mean-normalized or other distance is needed (`deviation_score` is
an unnormalized L1 sum), when prototype updates should be non-gradient, or as a
general retrieval store over real training samples (see `periodic_query_bank`).

## Interface

`PrototypeMemory(query_dim, prototype_dim, num_prototypes)`: all >= 1 and
`num_prototypes >= 2` (needed for top-2), else `ValueError`. Parameters (the only
state-dict keys): `prototypes` `[num_prototypes, prototype_dim]` and `query_proj`
`[query_dim, prototype_dim]`, both Xavier-normal. No buffers. `forward(h)` requires
`h.shape[-1] == query_dim` (`ValueError` otherwise; any leading shape, including none, and
any floating dtype/device matching the parameters) and returns a
`PrototypeRetrieval` dataclass (`value`, `query`, `nearest`, `second_nearest`,
`indices`; trailing axis `prototype_dim`, or 2 for `indices`). `query` and `value`
carry gradients; `nearest`/`second_nearest` are gathered by index, so gradients reach
the selected prototype rows but not the selection itself. Stateless, no memory
update rule: prototypes are trained only by whatever loss the caller builds.

`deviation_score(current, reference) -> Tensor`: last axes must be equal (`ValueError`),
leading axes broadcast; differentiable, no parameters; returns `[...]`. The value is the sum of
absolute differences over the last axis (`.sum(dim=-1)`), not a mean, so it
scales with `D`.
