---
name: AdaRDiff
description: "Adaptive causal differencing and reversible reconstruction around a shared Linear backbone; useful for seasonal per-channel forecasting."
---

# AdaRDiff

## Idea
Learn per-channel previous-lag differencing, forecast its residuals and reconstruct
the forecast through the finite causal inverse. Phase1 optimizes residual MSE;
phase2 optimizes forecast MSE. Uses the shared channel_wise_linear component.

## When to use
Seasonal univariate-channel structure with a declared lag window. This entry is
the paper's Linear-backbone variant, not a complex-model benchmark substitute.
Do not treat admission as numerical reproduction of reported paper scores.

## Configure
Set enc_in to all channels, period no greater than seq_len, and ordinary native
training.epochs to phase2 epochs. phase1_epochs defaults30, phase1_lr .001,
phase1_patience3; phase2 optimizer belongs in training.optimizer. Set enabled=false
for the separate plain Linear control; pretraining is then skipped.

## Differences
Independent inferred implementation, zero initialized coefficients, no RevIN or
l1 reparameterization. Residual validation selects phase1 checkpoint; phase2 uses
a fresh optimizer. Newton polynomial doubling replaces companion squaring, with
finite recurrence output and gradient checks. Unconstrained coefficients can
produce unstable long forecasts; nonfinite losses fail explicitly. reference.md
records the equations and protocol limits. Historical results remain separate.
