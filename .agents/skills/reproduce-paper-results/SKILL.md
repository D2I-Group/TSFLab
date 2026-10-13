---
name: reproduce-paper-results
description: "Reproduce and compare a forecasting paper's reported experiments in TSFLab by mapping its protocol to runnable configs and aligned metrics. Use for paper-result replication; not for implementing the model or designing an unrelated benchmark."
---

# Reproduce paper results

Experiments module: rerun a paper's reported experiments under a traceable protocol map and compare
cell by cell. Success means the attempt is rerunnable and traceable, not that the
numbers match.

## Inputs

- An identified paper and a runnable, audited catalog model.
- The primary paper, supplement, authoritative source revision, and reported tables,
  all verified before spending compute.

## Steps

1. Route implementation doubts to `audit`; when the model is absent, use the
   `intake` task or `add-model` instead of implementing it here.
2. Map the protocol: dataset name and version, split boundaries, scaling, feature
   mode, lookback and horizons, covariates, loss, optimizer and schedule, batch
   size, epochs and stopping, seeds, checkpoint selection, metric formula and
   aggregation, baselines, and hardware-sensitive settings. Label every field
   `aligned`, `adapted`, `unknown`, or `blocked`; never silently fill gaps with defaults.
   Document justified assumptions as adaptations, including their effect on the claim.
3. Encode aligned settings in dedicated inherited run configs, keep faithful
   replication separate from controlled adaptations, and inspect the matrix:

   ```bash
   uv run tsf run <paper-run.toml> --dry-run
   ```

   Bind each dataset to its own explicit path and channel/target settings. Before
   training, check the resolved cells against that binding; changing a sweep's
   dataset name must not retain another dataset's path. Resolve optimizer, loss,
   schedule and model dimensions before launching the full matrix.

4. When authorized, execute through `run-experiment` on a GPU machine or CI (no local
   training). Reuse a supplied round; create one with `tsf research start --goal ...
   --max-runs N` only when persistent memory/budgets are needed, then pass its ID
   with `tsf run --round`. Batch cells and wait for metrics as in `run-experiment`.
   Preserve raw outputs, resolved
   configs, environment facts, seeds, and failed runs returned from that machine.
   Use blocking completion returns, not healthy-log polling. Keep overlapping
   windows lazy or batch-gathered rather than materializing the full matrix.
   Reuse a frozen encoder across horizons only when the paper's training stages
   permit it; never reuse checkpoints from another measured attempt. Preserve
   required epochs and seeds when improving throughput.
5. Aggregate compatible cells with `analyze-results`, then report per cell: paper
   value, local value, absolute and relative difference, run count, uncertainty,
   and every protocol deviation. Missing or failed cells stay visible.

## Chain

- Module: Experiments.
- Reads: paper protocol, model card (L2), dataset card, audited evidence.
- Produces: protocol map, paper-run configs, per-cell paper-versus-local comparison.
- Hands off to: `run-experiment`, `analyze-results`, `submit-results` (when records are shareable).

## Success

- A rerunnable, traceable comparison; call it reproduced only when the recorded
  protocol and results support that, otherwise partial or blocked.

## Stop and hand off

- Stop for a decision when missing data, licensing, ambiguous metrics, or
  infeasible compute would materially change the claim.
