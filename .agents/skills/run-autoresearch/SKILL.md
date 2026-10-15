---
name: run-autoresearch
description: "Run a bounded, profile-driven forecasting research loop: analyze the data, state hypotheses, run a baseline panel, search and recombine cataloged components, and conclude from evidence. Use when autonomous experiment iteration is authorized; not for a single predefined run or a paper implementation."
---

# Run bounded autoresearch

Iterate profile, hypothesis, experiment, and evidence inside hard budgets, in the current
Agent; read each reference only at its step. A supplied candidate space follows only
[config search](references/config-search.md); open method discovery uses
[adaptive search](references/adaptive-search.md).

## Inputs
- A research question, approved datasets, primary metric, hard run, iteration, and
  time budgets, and authorization to execute experiments.
- Separate authorization before writing model code or registering a model; without
  it, search over configs and existing catalog methods only.
- Heavy runs happen on GPU machines or CI, not locally: preview each matrix with
  `--dry-run` and hand the exact `tsf run ... --round <id>` command to the run machine;
  continue from the records and round events in `work_dirs/`. On a design-only
  machine, report the pending handoff without inventing results.

Context, read progressively: dataset card and train-only profile (`tsf data analyze`);
card `fits`, compositions, `data_params` (`tsf catalog match|search|show`); validation
boards for selection (`tsf result board`); published/test leaderboards as context only;
the round ledger (`tsf research show <id>`).

## Steps

1. Profile each dataset first: `uv run tsf data analyze <preset>` (or `--path FILE`);
   read the fired rules (= card data terms) per [data-profile](references/data-profile.md).
2. Write 2-4 falsifiable hypotheses, each tied to a profile fact, with the metric
   change that would refute it. Design controls and seeds per `run-experiment`.
3. Open a round for durable budgets (`tsf research start --goal ... --max-runs N
   --max-iterations M --policy <policy.toml>`, or reuse a supplied one). The policy
   carries the hard time/resource limits. Read [round-ledger](references/round-ledger.md)
   before spending; reserve confirmation runs and note each hypothesis
   (`tsf research note <id> --kind hypothesis --text ...`). Claim one iteration per phase
   with `tsf research iteration <id> --operation <phase>`; reuse of an id is free.
4. Retrieve candidates: `uv run tsf catalog match <preset...> [--extra TERM...]` ranks
   models and slot-grouped components whose card `fits` overlap the dataset's
   characteristics; widen with `tsf catalog search`, open the shortlist's L1 in one call
   (`tsf catalog show <name...>`); see [retrieval-baselines](references/retrieval-baselines.md).
5. Evaluate a small baseline panel before any search (same split, horizon, metric,
   seeds, validation split) as one sweep; it sets the bar every later candidate must beat.
6. Search in batches: each iteration puts several candidates, each one factor away from
   the best so far, in one sweep (`[sweep.extend]` or `[sweep]`), runs it once with
   `--round`, waits once (`tsf run wait <dir> --json`), and reads every result returned.
   Start with the cheapest runs that can reject a hypothesis; set data-dependent
   parameters from card `[data_params]` and the profile; recombine per
   [recombination](references/recombination.md) (`tsf model compose <spec.toml>`).
7. Select the validation-best completed model across controls and candidates; keep a
   baseline fallback and report no gain if nothing improves it. Confirm finalists in one
   seed sweep with spread. If authorized and a recombined method wins, use `add-model`.
8. Record decisions, runs, and conclusions as round events per
   [round-ledger](references/round-ledger.md); route failures to `diagnose-experiment`
   and compatible results to `analyze-results`, whose board feeds the next iteration.
   Never overwrite a costly run.

## Leakage guards

- Select and stop on train and validation only, with the rule fixed first. Runs with
  `--round` score validation by default; only frozen finalists are scored on test, once
  (`--eval-split test` or `[evaluation] split = "test"`). Never act on `shift.train_to_test`.

## Chain

- AutoResearch: reads the context above; produces the round ledger, conclusions, and a
  winning composition spec; hands off to `add-model` (registers a winner), `submit-results`, `analyze-results`, `diagnose-experiment`.

## Success

- Profile paths, hypotheses with verdicts, baseline panel, run ledger with seeds, failed
  cells, limitations, and a stop/continue recommendation.

## Stop and hand off

- Stop on budget exhaustion, repeated infrastructure failure, an invalid comparison,
  no measurable progress after the declared search patience, or an acceptance criterion met.
- External publication and task dispatch stay out of scope unless separately authorized.
