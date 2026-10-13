---
name: run-autoresearch
description: "Run a bounded, profile-driven forecasting research loop: analyze the data, state hypotheses, run a baseline panel, search and recombine cataloged components, and conclude from evidence. Use when autonomous experiment iteration is authorized; not for a single predefined run or a paper implementation."
---

# Run bounded autoresearch

Iterate profile, hypothesis, experiment, and evidence inside hard budgets, in the
current Agent (no second Agent needed). Read each reference only at its step.
For a supplied model/candidate space, follow only [config search](references/config-search.md)
and stop there; skip the general Steps below. For open method discovery, use
[adaptive search](references/adaptive-search.md); inspect source when implementing
an authorized change or diagnosing failure.

## Inputs
- A research question, approved datasets, primary metric, hard run, iteration, and
  time budgets, and authorization to execute experiments.
- Separate authorization before writing model code or registering a model; without
  it, search over configs and existing catalog methods only.
- Heavy runs happen on GPU machines or CI, not locally: for a handoff preview with
  `--dry-run`, hand the exact `tsf run ... --round <id>` command to the run machine,
  and continue from the records and round events that come back in `work_dirs/`.
  On the authorized run machine, execute and wait for returned metrics; on a
  design-only machine, report the pending handoff without inventing results.

Read profiles/cards progressively; expand only for a decision. Use validation
boards for selection; published/test leaderboards are context only.

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
4. Retrieve candidates: `uv run tsf catalog match <preset> [--extra TERM...]` ranks
   models and slot-grouped components whose card `fits` overlap the dataset's
   characteristics; widen with `tsf catalog search`, open L1 with `tsf catalog show`;
   see [retrieval-baselines](references/retrieval-baselines.md). A match is a hypothesis.
5. Evaluate a small baseline panel before any search (same split, horizon, metric,
   seeds, `--eval-split val`); it sets the bar every later candidate must beat.
   Batch independent cells in one sweep; block on the run or use `tsf run wait
   DIRECTORY --json`. Read logs only for failures; decide after returned metrics.
6. Search by changing one factor per iteration from the best baseline, starting with
   the cheapest run that can reject a hypothesis. Set data-dependent parameters from
   card `[data_params]` and the profile; tune generic hyperparameters yourself within
   budget. Recombine via the slot grid per [recombination](references/recombination.md)
   (`tsf model compose <spec.toml>` dry-runs a composition).
7. Select the validation-best completed model across controls and derived candidates;
   retain a baseline fallback and report no discovery gain if no change improves it.
   Confirm only finalists with the declared seeds; compare with spread, not a single
   best run. If authorized and a recombined method wins, register it with `add-model`.
8. Record decisions, runs, and conclusions as round events per
   [round-ledger](references/round-ledger.md); route failures to `diagnose-experiment`
   and compatible results to `analyze-results`, whose board feeds the next iteration.
   Never overwrite a costly run.

## Leakage guards
- Select and stop on train and validation only, with the rule fixed first; test metrics
  confirm frozen candidates once (`--eval-split test`); never act on `shift.train_to_test`.

## Chain

- Module: AutoResearch; reads the context table above.
- Produces: the round ledger, conclusions, and a winning composition spec.
- Hands off to: `add-model` (registers a winner), `submit-results`, `analyze-results`, `diagnose-experiment`.

## Success

- Profile paths, hypotheses with verdicts, baseline panel, run ledger with seeds and
  uncertainty, excluded or failed cells, limitations, and a stop/continue recommendation.

## Stop and hand off

- Stop on budget exhaustion, repeated infrastructure failure, an invalid comparison,
  no measurable progress after the declared search patience, or an acceptance criterion met.
- External publication and task dispatch stay out of scope unless separately authorized.
