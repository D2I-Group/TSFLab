---
name: diagnose-experiment
description: "Diagnose a failed, unstable, or invalid TSFLab experiment. Use for crashes, NaNs, OOMs, suspicious metrics, missing outputs, leakage, or irreproducible runs; not for ordinary result ranking."
---

# Diagnose an experiment

Experiments module: find the earliest supported root cause of a bad run. Scheduler labels such as OOM
are signals, not a diagnosis; use library state, audit, and recovery APIs where
their guarantees matter.

## Inputs

- The failing config, command, environment, logs, and artifacts; preserve them
  all before changing anything. Failures from GPU or CI runs arrive as `work_dirs/`
  copies and logs; diagnose from those, and reproduce only at smoke scale locally.

## Steps

1. For managed runs, read `uv run tsf run status <directory> --json` and its attempt
   log first; for environment failures run `uv run tsf env audit --config <run.toml> --json`.
2. Reproduce with the smallest equivalent config and classify the failure:
   environment, data, shape/contract, model, optimization, resource, evaluation,
   or output bookkeeping.
3. Inspect the resolved matrix (`uv run tsf run <run.toml> --dry-run`), the
   model's smoke case (`tsf run --smoke --model <Name>`, tiny CPU data), dataset
   splits, tensor shapes, loss/output pairing, metric direction, seeds, device
   placement, checkpoints, and finite values.
4. For OOM or instability, change one resource or optimization variable at a time;
   never lower the scientific workload and call it equivalent.
5. When the run belongs to a research round, append the supported classification
   or decision to it; do not paste the full log or keep a separate ledger.

## Chain

- Module: Experiments.
- Reads: run status, attempt logs, resolved config, model smoke case.
- Produces: a root-cause note (classification, affected runs) in the round when one exists.
- Hands off to: `run-experiment` (repaired runs), `analyze-results` (valid outputs), `handle-contribution`.

## Success

- Root cause, minimal reproduction, evidence, affected runs, and whether existing
  results are invalid.

## Stop and hand off

- Apply a fix only when requested, then rerun the minimal reproduction and the
  affected contract or smoke check.
- A supported continuation uses `uv run tsf run resume <directory>` after checking
  config, code, and data still match; scientific changes need a new run.
- Never overwrite costly artifacts or restart a broad sweep without authorization.
  Return repaired runs to `run-experiment`; valid outputs to `analyze-results`.
