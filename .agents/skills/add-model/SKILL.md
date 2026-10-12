---
name: add-model
description: "Turn one forecasting paper into an admitted flat-catalog model, from structure extraction and component decisions through local implementation, card, spec, preset, and admission. Use for adding or replacing one locally implemented model; not for paper discovery, foundation checkpoints, or placeholder entries."
---

# Add a model

Models module, step two: create one flat model package (the catalog discovers it from
`spec.py`; no registry to edit) and admit it once it is implemented. Models and methods
are peers; no family directories. Its card (description, fits, composition,
data_params, Idea) is the model context AutoResearch retrieves and `tsf catalog match` ranks.

## Inputs

- Public name, lowercase module slug, parameters, input needs, output type.
- Paper identity and, when official code exists, its URL, revision, and license
  together (a missing license is recorded, not guessed).

## Steps

1. Extract the paper structure and give every defining operation a component
   decision (`reuse-existing`, `extract-new`, `model-local`) per
   [references/paper-structure.md](references/paper-structure.md). Stop if an
   ambiguity would change the method.
2. Scaffold the workspace (its SCAFFOLD markers fail the audit until replaced); pass
   `--task-mode` explicitly when it is not ordinary `time_series`:

   ```bash
   uv run tsf model scaffold MyModel --paper-title "..." --paper-url <url> \
     --venue <venue> --year <year> --components revin \
     --params "enc_in:int,hidden:int=128" [--task-mode spatiotemporal]
   ```

3. Before implementing any block, `uv run tsf catalog search --kind component <terms>` (L0), open
   the candidates with one `tsf catalog show <name...>` (L1), and read `--depth 2|3` only to
   settle a doubt. Retrieval is a shortlist: verify shapes, axes, normalization,
   masking, residual order, initialization, state, and outputs before reusing.
4. Implement locally per [references/implementation.md](references/implementation.md);
   replace every scaffold placeholder and never import another named model package.
5. Write the card (STANDARDS "Cards"). `card.toml`: `tags` (one family), `fits`
   (vocabulary terms the method is designed for), `fidelity`, `[paper]`, `[code]`
   with `reference_sources`, the six-slot `[composition]`, `[data_params]` for every
   time-series-specific data-dependent parameter (`from` + `rule`; generic
   hyperparameters stay in the preset), and `[[issues]]` for every paper or
   official-code problem (bugs, mismatches, gaps, leakage, license), else
   `issues_checked`; these findings are a contribution. `README.md`: `description`
   (what; when to use and not), then Idea, When to use, Configure, Differences
   (<= 60 lines; overflow to `reference.md`). Leave `[admission]` to the tools.
6. Add a preset (and a smoke preset when training is cheap). Equation and reference
   checks are part of implementation review, not a test suite; record what was
   compared in Differences and the fidelity.
7. Admit; admission checks the package, runs the static audits and the strict
   contract, and writes `[admission]` (on failure, fix and rerun):

   ```bash
   uv run tsf model add MyModel --verify   # refreshes pages, audits, lists reuse candidates
   ```

## Chain

- Module: Models.
- Reads: component cards (L0 search, L1 interface) and dataset cards when inputs matter.
- Produces: card (`card.toml` facts with fits, composition, data_params, `[admission]`; `README.md`), spec, preset `configs/models/<Name>.toml`.
- Hands off to: `run-experiment` (preset), `run-autoresearch` (catalog search and compositions), `audit`.

## Success

- One audited card and runtime spec, a preset, `[admission]` status `passed`,
  truthful source and artifact facts, and every reused component declared in
  `spec.py` and the card composition.

## Stop and hand off

- Generated placeholder code is never admitted; stop if admission fails
  (fix and rerun `tsf model verify MyModel`).
  A declined paper goes into `catalog/declined.toml` (reason plus issues found).
- Released pretrained runtimes use `integrate-foundation-model`; cross-model
  consolidation uses `curate-components`; do not widen into an unsolicited refactor.
