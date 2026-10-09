---
name: reproduce-paper-results
description: "Reproduce and compare a forecasting paper's reported experiments in TSFLab by mapping its protocol to runnable configs and aligned metrics. Use for paper-result replication; not for implementing the model or designing an unrelated benchmark."
---

# Reproduce paper results

Experiments module: rerun a paper's reported experiments under a traceable protocol map and compare
cell by cell. Success means the attempt is rerunnable and traceable, not that the
numbers match. Every local value comes from a run executed in this attempt; never
copy paper values into results.

## Inputs

- An identified paper; a catalog model, or the paper's method to implement.
- The primary paper, supplement, authoritative source revision, and reported tables,
  all verified before spending compute.

## Steps

1. Read the dataset protocol before anything else. It is a loader fact, not a ratio
   to re-derive; ETT, for example, splits only its first rows:

   ```bash
   uv run tsf data splits <dataset> --seq-len <L>    # rows used, borders, scaling
   uv run tsf data splits <dataset> --path <raw.csv> # same rule on a given raw copy
   ```

2. Get the model. A catalog model with doubts goes to `audit`. A method not in the
   catalog becomes a model module, not a private script: `tsf model scaffold`,
   implement it per `add-model` steps 1-6, register it with `tsf model add --name X`
   (no `--verify`), and run it through `tsf run` on the dataset presets (set
   `dataset.path` for a raw copy kept elsewhere). It stays unadmitted until
   `add-model` admits it; a reproduction attempt does not need admission.
3. Map the protocol: dataset name and version, split boundaries, scaling, feature
   mode, lookback and horizons, covariates, loss, optimizer and schedule, batch
   size, epochs and stopping, seeds, checkpoint selection, metric formula and
   aggregation, baselines, and hardware-sensitive settings. Label every field
   `aligned`, `adapted`, `unknown`, or `blocked`; never fill gaps with defaults: a
   setting the paper leaves open comes from the official script for that dataset and
   horizon (record the file), not from a generic preset in `configs/models/`. A paper
   split that differs from `tsf data splits` is `adapted` and run separately.
4. Encode aligned settings in dedicated inherited run configs, keep faithful
   replication separate from controlled adaptations, and inspect the matrix:

   ```bash
   uv run tsf run <paper-run.toml> --dry-run
   ```

5. Pilot one cell per dataset (shortest horizon) against the paper before the full
   matrix; diagnose a gap above 5% (settings, then implementation) and record the
   outcome. A remaining gap is reported, never tuned away on the test split.
6. When authorized, execute through `run-experiment` on the available GPU (this
   machine or CI); for multi-run work open a round (`tsf agent task start experiment
   --set question=...`) and pass it with `tsf run --round`. Block on a background matrix with
   `tsf run wait <dir> --timeout 1800` (it lists every finished run), never by reading logs.
   Preserve raw outputs, resolved configs, environment facts, seeds, and failed runs.
7. A custom loop, only if `tsf run` cannot express the procedure, takes splits and scaling
   from `tsflab.data.protocol.load_splits(<dataset>, seq_len, path=...)`; record why.
8. Aggregate compatible cells with `analyze-results`, then report per cell: paper
   value, local value, absolute and relative difference, run count, uncertainty,
   and every protocol deviation. Missing or failed cells stay visible.

## Chain

- Module: Experiments.
- Reads: paper protocol, `tsf data splits`, model card (L2), dataset card, audited evidence.
- Produces: protocol map, paper-run configs, per-cell paper-versus-local comparison.
- Hands off to: `add-model`, `run-experiment`, `analyze-results`, `submit-results`.

## Success

- A rerunnable, traceable comparison from executed runs; call it reproduced only
  when the recorded protocol and results support that, otherwise partial or blocked.

## Stop and hand off

- Stop for a decision when missing data, licensing, ambiguous metrics, or
  infeasible compute would materially change the claim.
- Stop rather than report a cell without an executed run behind it.
