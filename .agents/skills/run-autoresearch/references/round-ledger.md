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
strategy. Aggregate with `tsf result aggregate <name>` and `tsf result rank`.

## Leakage guards

- Fix the selection rule (metric, source, tie-break) in a `decision` note before runs.
- Select among candidates by validation behavior (the trainer logs `val_loss` per epoch
  and keeps the best-validation checkpoint). Runs also write test metrics; read them only
  for frozen candidates at confirmation, never to pick, tune, or stop early.
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
