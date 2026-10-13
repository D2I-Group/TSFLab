# Budgets, seeds, stopping, and the round ledger

## Budget plan (default task: 12 runs, 4 iterations, 360 minutes)

| Iteration | Operation id | Spend | Purpose |
| --- | --- | --- | --- |
| 1 | `panel` | up to a third of runs | Baseline panel, one seed, one horizon. |
| 2 | `screen-1` | about a quarter | Single-slot swaps on the incumbent (cheapest hypothesis first). |
| 3 | `screen-2` | about a sixth | Combine the two best compatible swaps, or the recombined spec. |
| 4 | `confirm` | the rest | Finalists plus incumbent, 3 seeds (`experiment.random_seed` sweep). |

Scale proportionally for other budgets. Each iteration ends when its runs have returned
as records from the run machine; ledger events cite those records. Reserve confirmation runs up front; a search that
leaves none cannot conclude. Use `tsf run <run.toml> --dry-run` to count resolved runs
before spending, and `--round <id>` so the budget is consumed atomically.

## Seeds, horizons, and comparisons

Screening: one seed, one representative horizon. Confirmation: at least 3 seeds. Report
mean and spread; call a gain real only when it exceeds the baseline seed spread. Compare
only cells with the same split, preprocessing, lookback, horizon, metric, and evaluation
strategy. Aggregate with `tsf result aggregate --dataset <name>` and `tsf result rank`.

## Leakage guards

- Fix the selection rule (metric, source, tie-break) in a `decision` note before runs.
- Run baseline, screening, and seed confirmation with `--eval-split val`; the trainer
  keeps the best-validation checkpoint. Only after freezing the candidate and selection
  rule, run `tsf run evaluate RUN_DIRECTORY... --eval-split test` on the saved
  successful confirmation runs, with no retraining; never use it to pick or stop.
- Profile statistics for decisions come from train and train-to-val shift. A claim that
  needed `shift.train_to_test` is invalid and must be rerun without it.
- Never change a candidate after seeing its test result; a changed candidate is a new
  candidate with a new hypothesis note and a new budget claim.

## Round events

Use `tsf research note <id> --kind <kind> --text "..."` with these kinds:

- `observation`: profile path and the facts used (metric, value, rule id).
- `hypothesis`: claim, profile fact, prediction, refuting result (for example "H1:
  strong-seasonality; CycleNet beats RLinear by 3 percent MSE; refuted if within baseline spread").
- `decision`: selection rule, exclusions with reasons, slot pruning, run matrix.
- `run`: config path, seeds, output directory, validation and test metric references.
- `failure`: failed or excluded cell with cause; failures are never deleted.
- `conclusion`: verdict per hypothesis, evidence, limitations, recommendation.

Check state with `tsf research show <id>`. Finish with
`tsf research status <id> completed|stopped|blocked --message "..."`.

## Stopping rules

Stop and report when any holds: budget exhausted; two consecutive iterations without
improvement beyond baseline seed spread; repeated infrastructure failure (hand to
`diagnose-experiment`); the comparison is invalid (different protocol or leaked selection);
or the acceptance criterion is met. Say plainly when the answer is that simple baselines
suffice.

## Report

One table per dataset: hypothesis, treatment, control, seeds, validation and test metric
with spread, verdict. Add profile path, run ledger, excluded and failed cells,
limitations, and a stop or continue recommendation.
## Automatic execution and stopping

Keep one round and one session-wide budget. Submit independent candidates as
one sweep and wait on the original process; `tsf run wait DIR --timeout 1800
--json` returns saved metrics if the launch tool yielded. Do not query healthy
training logs, GPU status, or process lists between epochs. Each completed
sweep supplies the evidence for one next-step decision. A budget stop or two
iterations without improvement ends the search, including when no method wins.
Set validation explicitly for baseline, screening and confirmation runs;
only a frozen finalist gets a separate, explicit test evaluation.

## Minimal policy on a two-GPU run host

Use only needed limits; exclusive allocation is already the default.

```toml
[budget]
max_runs = 10
max_parallel_jobs = 2
max_wall_minutes = 10
[resources]
threads_per_run = 4
```

Create one round with `tsf research start --goal QUESTION --max-runs 10
--max-iterations 3 --policy POLICY.toml`. Execute with the same policy,
`--round ID --gpus auto --jobs auto`. Do not enable GPU sharing with auto selection.
For arbitrary CSV paths, use `dataset.name=custom`; the file label is an alias,
not a dataset registry key.
