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

   A separate preview adds `--dry-run` to that same command. Attach a round only when
   supplied or when the task needs persistent research context. On a multi-GPU host,
   use `--gpus auto --jobs auto`; explicit indices must stay inside `CUDA_VISIBLE_DEVICES`.
   Set `--eval-split val` for search and `--eval-split test` for frozen evaluation.
4. Run blocks and returns metrics, timings and artifact paths. A detached sweep is
   awaited with `tsf run wait <dir> --timeout 1800 --json` (timeout does not cancel it);
   never tail logs, poll GPUs, or repeat a launch to recover its output: recover the
   sweep directory from the launch output or `work_dirs/_sweeps/*/sweep.json`.
   `--output` identifies one matrix request; repeating it reuses the original sweep.
   Inspect failed runs only; use records for execution checks and reports.
5. New data that is not a preset: register it in place, check its split, and compare
   methods with one sweep; catalog models always run through `tsf run`, never a
   private training loop (`tsf data add <data> --pattern custom --path <file.csv>`,
   then `tsf data splits <data> --seq-len <L>`; list methods under `[sweep.extend]`).
6. Load [execution controls](references/execution.md) only for policies, queues,
   tracking or recovery. For GIFT-Eval, prepare only requested data with
   `tsf data prepare --from gift-eval` and record versions and missing-series policy.

## Chain

- Experiments: reads cards/presets; saves records (`work_dirs/<dataset>/<model>/records/`,
  `performance.csv` with metrics, times, params, epochs, peak memory); hands off to
  `analyze-results`, `submit-results`, `diagnose-experiment`, `run-autoresearch`.

## Success

- Every requested cell has a returned status, metrics or error, and artifact path.

## Stop and hand off

- Stop at the budget; preserve failures and never silently restart costly runs.
- Diagnose failed/suspect outputs; analyze completed protocol-compatible results.
