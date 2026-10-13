---
name: run-experiment
description: "Design and execute TSFLab runs or sweeps, including standalone projects. Use for training, evaluation and ablations; paper replication uses reproduce-paper-results, autonomous search uses run-autoresearch, contract checks use audit."
---

# Run experiments

Agent owns decisions; APIs own validation, execution, budgets and recovery.

## Inputs

- A question or supplied configs, datasets, metric, seeds and resource budget.
- An authorized GPU/CI run machine; the maintainer's local machine is for design,
  previews and returned records. A standalone project starts with `tsf init <dir>`.

## Steps

1. With supplied configs, validate them directly. Otherwise use
   [design](references/design.md) for controls, selection rule and fair baselines.
   Read the named preset/card only. For parameter names/constraints, query
   `tsf catalog show MODEL --params` once; fixed sizes need not be overridden.
   For an admitted model, inspect source only for a failure or fidelity audit.
   For a fully specified task, use [direct requests](references/request.md) to
   avoid config authoring. Use [minimal TOML](references/config.md) for a saved study.
2. Preserve the task's dataset identity and native row/window rules even when
   supplied as a CSV; use `dataset.name=custom` for generic CSV tasks only. Check
   data, split, horizons, seeds, output paths, budget, profiling and GPU
   intent. Batch independent cells in one sweep; models with different schemas
   use separate configs passed together (`tsf run A.toml B.toml ...`). For batching
   comparisons, verify cells per request; single-cell full requests do not test batching. Search uses validation;
   frozen finalists alone receive a separate explicit test evaluation.
3. Execute once on the authorized run machine; run validates the whole matrix
   before starting compute. Repair rejected fields using the named schema; a
   prelaunch validation error does not finish the task. Preview only for handoff:

   ```bash
   uv run tsf run <config.toml> [--round ID] [--jobs N] [--gpus 0,1] --json --compact --output results/completion.json
   ```

   A separate preview adds `--dry-run` to that same command.
   On a multi-GPU run host, use `--gpus auto --jobs auto` for visible idle GPUs.
   Busy visible cards are awaited within policy; do not retry with physical `0,1`.
   Explicit GPU indices/UUIDs must remain inside `CUDA_VISIBLE_DEVICES`.
   Set `--eval-split val` for search and `--eval-split test` for frozen evaluation.
4. Run blocks and returns metrics, timings and artifact paths. Set the shell
   tool's initial `yield_time_ms=30000`; for code-mode orchestration set the
   outer exec yield to 30000 too. If it yields, wait on the same process with
   a long wait; avoid repeated short waits. A detached run uses
   `tsf run wait DIRECTORY --timeout 1800 --json`; timeout does not cancel it.
   If the shell handle is lost, recover the original sweep directory from the
   launch output or saved `work_dirs/_sweeps/*/sweep.json`, then wait on that
   directory. Never repeat the launch command to recover its output. A missing
   completion file does not mean the original process stopped.
5. `--output` saves returned JSON directly, including checkpoint and record paths.
   Its path identifies one matrix request: repeating it reuses the original sweep.
   Use a new output path for a different matrix. Within a research round, an
   identical matrix also reuses its sweep even if the output path changes.
   Accept its native `runs` rows; do not re-read successful artifacts or reformat
   them into a second report. Inspect failed runs only; healthy logs, GPU
   status and process lists need no Agent polling. Use records for execution
   checks and reports.
6. Load [execution controls](references/execution.md) only for a policy setting
   not covered here, queues, tracking or recovery. Ordinary exclusive auto-GPU
   execution needs no extra reference. For GIFT-Eval, prepare only requested
   data with `tsf data prepare --from gift-eval`, preview its sweep, and record
   versions, compatibility and missing-series policy. Design-only machines
   hand off the previewed command and report pending results.

## Chain

- Experiments: reads cards/presets; saves records in `work_dirs/`; hands off to `analyze-results`, `submit-results`, `diagnose-experiment`, `run-autoresearch`.

## Success

- Every requested cell has a returned status, metrics or error, and artifact path.

## Stop and hand off

- Stop at the budget; preserve failures and never silently restart costly runs.
- Diagnose failed/suspect outputs; analyze completed protocol-compatible results.
