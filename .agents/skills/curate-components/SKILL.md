---
name: curate-components
description: "Identify repeated or reusable operators across existing TSFLab models, prove semantic and runtime-contract equivalence, and extract them as cataloged components without changing behavior. Use for deliberate cross-model consolidation or preparing building blocks for automated research; not for reorganizing models into families or sharing code based on similar names."
---

# Curate shared components

Models module: turn duplicated or recombinable operators into cataloged components while keeping
every consumer's behavior bit-for-bit unchanged. Components are reusable
implementation units, never model categories; read the components section of
`.agents/STANDARDS.md` first.

## Inputs

- An explicit scope: named models, a family of candidates, or a repository pass.
- The current catalog, read progressively: `uv run tsf catalog search --kind component <terms>` (L0),
  `tsf catalog show <name>` (L1), `--depth 2` (full card), `--depth 3`
  (paths to open).

## Steps

1. Build the shortlist with `uv run tsf model similar [slug...] [--json]`: static,
   alpha-renamed AST clusters ranked by breadth x agreement, each marked
   `reuse-existing candidate: <component>` or `extract-new candidate`. It only
   nominates; triage per [references/similarity.md](references/similarity.md).
2. For each candidate compare equations, tensor axes and shapes, normalization and
   residual order, masking, initialization, state, dtype/device behavior, outputs,
   and error contracts. Decide `reuse-existing`, `extract-new`, or
   `keep-model-local`, and record why.
3. Prefer migrating to an existing component. Extract a new one when two or more
   consumers match exactly, or when the operator is a paper-neutral building block
   with a standalone contract worth recombining.
4. Before editing, capture reference fixtures from the unmodified models under a
   fixed seed and input: `state_dict()` keys and values, eval-mode outputs, and
   gradients. Keep parameter attribute names so checkpoints still load.
5. Define the smallest paper-neutral API (list it in `__all__`) and explicit imports;
   the catalog entry is derived from the package and card, with no list to edit
   (`role` is the contract, `tags` the search keywords). Write the card from the
   code, not from memory: `card.toml` (`role`,
   `slot`, `fits`, `category`, `tags`, `input`, `output`, `origin`,
   `origin_models`) and `README.md` (What it does, When to use, Interface; detail
   in `reference.md`). Migrate consumers in reviewable groups without flags that
   hide material variants; keep attribution and explanatory comments.
6. Compare every migrated consumer against the fixtures (keys, outputs, gradients),
   then re-admit every model that uses a touched component:

   ```bash
   uv run tsf repo check --scope changed --base <ref>   # re-admits every consumer, audits cards
   ```

## Chain

- Module: Models.
- Reads: component cards (L0-L2) and model-local code.
- Produces: component cards (role, slot, fits, interface); consumers derived from imports; re-admitted consumers.
- Hands off to: `add-model` and `run-autoresearch` (recombination reads component interfaces), `audit`.

## Success

- Each new component has a curated card that passes `tsf model audit --components`
  and declared consumers.
- Every migrated consumer matches its fixtures (keys, outputs, gradients) and has
  `[admission]` status `passed`.
- No peer-model imports.

## Stop and hand off

- If equivalence cannot be shown, leave the variant local and note the reason in
  its card; a no-change pass is valid.
- Report the scope inspected, decisions, consumers, and next candidates so a later
  pass can resume. This skill does not schedule background monitoring.
