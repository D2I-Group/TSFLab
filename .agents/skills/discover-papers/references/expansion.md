# Authorized catalog expansion

Search, implementation, and admission end to end for a fixed number of papers. Needs
authorization to modify the repository and an explicit candidate and model budget;
without it, stop at the ranked queue.

1. Build the queue as in the main steps, deduplicated against
   `uv run tsf catalog list --kind model --json`.
2. For each retained candidate, verify the primary paper, forecasting task, required
   inputs, official source, revision, and license (recorded as missing when absent),
   then select at most the authorized number.
3. Per paper, run `add-model` (its references cover structure extraction and
   implementation) or `integrate-foundation-model` for a released pretrained runtime.
   Build the `reuse-existing` / `extract-new` / `model-local` map before code.
4. Papers are independent: run them in parallel workers only on disjoint model
   directories; the model and component catalogs are derived from the packages, so
   there is no shared registry file to merge (regenerate pages once at the end).
5. Each model is admitted with `tsf model add <Name> --verify` (`[admission]`
   passed) and passes the model audit, component audit, and repository audit.
   A declined paper goes into `catalog/declined.toml` with its reason and issues.

Outcome: every admitted model has a truthful card and a passed admission; skipped candidates
listed with reasons. No candidate clearing the gate is a valid no-change result.
Stop when the budget is exhausted or ambiguity would require inventing a claim. Never
use search relevance or a shape-only test as implementation evidence. Do not publish,
push, open issues, or dispatch more tasks unless separately authorized.
