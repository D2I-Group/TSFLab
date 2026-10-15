---
name: "periodic_query_bank"
description: "Zero-initialized learnable [period, channels] table read as a phase-aligned window of rows (wrapping modulo period) from a per-sample start phase. Use for series with a known fixed cycle whose phase comes from timestamps (TQNet); not for unknown or varying periods."
---

# periodic_query_bank

## What it does

`PeriodicQueryBank(period, channels)` stores one learnable `channels`-vector per
phase of a fixed period. For start phases `s[b]` and a window `length`, it
returns `table[(s[b] + j) mod period]` for `j = 0..length-1`, shape
`[batch, length, channels]`. It is gather-only indexing: no value is computed
from the data.

## When to use

Use when the data has a known fixed cycle (daily or weekly) and the start phase
of each sample can be derived from its timestamps, so a learned per-phase vector
can act as a periodic query or offset. Do not use for unknown or varying periods
(see `dominant_periods`) or when the phase cannot be computed at the call site.
The bank starts at zero and carries no information until trained.

## Interface

`PeriodicQueryBank(period: int, channels: int)`

- `period` (int >= 1) and `channels` (int >= 1), else `ValueError`.
- Parameter / state-dict key: `table`, shape `[period, channels]`, initialized to
  zeros (so an untrained bank returns zeros).
- `forward(start_phase, length) -> Tensor`: `start_phase` integer tensor of shape
  `[batch]` (`ndim != 1` raises `ValueError`; must be an integer dtype for
  indexing), `length` int >= 1 (else `ValueError`; may exceed `period`, then the
  window wraps repeatedly). Phases may be any integers; `remainder` makes them
  periodic (negative values wrap into range). Output is on the table's device;
  `start_phase` should be on the same device.
- Gradients reach only the gathered rows of `table`. Stateless otherwise.
