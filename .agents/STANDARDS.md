# Repository standards
Read only the relevant section; `AGENTS.md` holds the always-on rules.

## Models
Every model or method is a peer under `src/tsflab/models/<lowercase_module_slug>/`
with no architecture categories. Each entry owns `model.py`, a card (`card.toml`,
`README.md`, optional `reference.md`), and a `spec.py` limited to its factory, parameter
schema, and runtime contract. Registration is the admission boundary;
configs are presets, not registrations. A releasable entry imports, validates
parameters, constructs, returns finite correctly shaped output, and has a passed admission.
Each check lives once: params schema, `TaskConfig`, or `tsflab.models._inputs` (tensor shapes).

## Components
Reusable building blocks live in `src/tsflab/models/_components/`; they never classify
models. Every new model maps its defining operations to `reuse-existing`, `extract-new`, or
`model-local` first. Reuse a component only when mathematics, shapes, normalization,
masking, residual order, initialization, and outputs match; similar names are not evidence.
Extract a new one when two or more consumers match exactly, or when it is a paper-neutral
building block with a standalone contract (transform, attention variant, router,
normalization) that automated research can recombine; paper-specific glue stays local.

Extraction from an existing model preserves state-dict keys, outputs, and gradients,
shown against fixtures captured before the change, and re-admits every consumer
(`tsf model verify --changed`). Material variants stay local and named. Models never import
peer-model code; shared code moves into a component with a card. Audit limits: function complexity <= 10, method package <= 600 lines.

## Cards (`tsflab.card/1`)
One format for models, components, and datasets; schema in `catalog/cards/schema.py`,
audit in `tsf repo check --audit`. Each card directory holds:
- `card.toml`: facts only. Model: `tags` (one architecture family), `fits`, `fidelity`
  (`reference-checked|paper-only|inferred|composed`), `[paper]`, `[code]` (url, revision,
  license, `reference_sources`), `[composition]` (six slots
  `normalization|decomposition|temporal|channel|head|loss` = `component:/local:/loss:/none`,
  naming only imported components), `[data_params]`, `[[issues]]` or `issues_checked`,
  `[admission]`. Component: `role`, `slot`, `fits`, `category`, `tags`, `input`, `output`,
  `origin`, `origin_models`. Dataset: `domain`, `topic`, `benchmarks`, `tags`, `characteristics` +
  `characteristics_basis`, `related`, `[source]` (`redistribution`, `conditions`, `license_url`),
  `[shape]`, `[protocol]`; enums `DOMAINS`, `BENCHMARKS`, `REDISTRIBUTION` in `cards/schema.py`.
- `README.md`: Skill-shaped. Front matter `name` + `description` (what it is; when to use
  and not) is L0. The body is L1 (<= 60 lines) with fixed sections: model Idea, When to
  use, Configure, Differences; component What it does, When to use, Interface; dataset
  Overview, Protocol and pitfalls.
- `reference.md`: optional L2 (derivations, full interface, provenance) only when L1
  would overflow.
Nothing generated lives in a card: config path, parameters, imports, consumers, loaders,
and task modes are derived at read time. Never invent facts; write what was checked.

`fits` and dataset `characteristics` share one vocabulary (`catalog/characteristics.py`):
data terms are rule ids of `src/tsflab/data/profile_rules.toml`, measured by
`tsf data analyze <preset> --write-card`; task terms are curated; `any` marks generic
components. `[data_params]` maps a time-series-specific, data-dependent parameter to
`from = period|frequency|channels|nodes|seq_len|pred_len|graph|train-split` and a `rule`;
generic hyperparameters are left to the agent. `[[issues]]` record upstream problems
(`kind`, `where` at the pinned revision, `what`, `resolution`; kinds in the schema).

## Sources and fidelity
Ordinary paper architectures are local code; inspect official code at a pinned revision
to resolve omissions without copying it, and record it in `[code]` (or
`inspected_sources` when no license allows a `[code]` record). A released pretrained
foundation model uses its official package and checkpoint behind
`src/tsflab/models/_foundation/`, loads offline from an explicit local path, and is
inference-only. `reference-checked` needs official code; a shape-only smoke run is not
a fidelity claim.

## Admission
`tsf model verify <Name...>|--all|--changed [--base REF]` runs the strict executable
contract once (construction, forward, backward or one training step, shapes, finite
outputs, active gradients, state-dict round trip, CPU) and writes `[admission]`
(status, date, commit, device, reference, note) into `card.toml`. Check a model when it
is added (`tsf model add --name X --verify`) or when its package, preset, or a used
component changes (`--changed`, also run in CI). There is no fingerprint, evidence file,
or global model test suite; `tests/` holds infrastructure tests. A release requires
every admission `passed` (`tsf model audit --release`, `tsf repo check --scope release`).

## Data, experiments, and model artifacts
Dataset bytes live only in `dataset/`, loaders in `src/tsflab/data/`, cards in
`catalog/datasets/<name>/`. Task modes are executable contracts checked at config loading.
Experiments are resolved TOML and immutable records under `work_dirs/`; execution policy never
changes scientific settings. Large weights are `ModelArtifact` facts in `spec.py`, pinned by
revision and SHA-256, never bundled or fetched implicitly (`tsf model artifacts`).

## Documentation ownership
Human docs (root, `docs/`, cards) explain public behavior without Agent paths; Agent
procedures live only under `.agents/`. Schemas, card facts, specs, configs, and tests are
executable truth; generated pages (the human model table, `.agents/README.md`) are
projections: update the truth first, then `tsf repo cards`.

## Modules, skills, and tasks
Chain: Data -> Models -> Experiments -> Release; each module yields context AutoResearch
reads (cards, profiles, `tsf catalog match`, compositions, result board, round ledger).
Maintenance (audit, handle-contribution; tasks `maintenance`, `contribution`) sits beside
the chain and is opt-in for `tsf init`. `src/tsflab/agent/modules.py` is the only source of
the skill/task-to-module map; `.agents/README.md` is generated from it by `tsf repo cards`
and checked by the asset audit, which also requires each skill and task in one module.

Skills live only at `.agents/skills/<skill-name>/SKILL.md` (80-line budget; detail in
`references/`), kebab-case `name`, discriminating quoted `description`, and Inputs, Steps,
Chain, Success, Stop sections using public commands only. Training and sweeps run on GPU
machines or CI; skills preview, hand off, and read the returned records. Tasks are
`.agents/tasks/<name>.toml` (`tsf agent task`). Check with `uv run python -m tsflab.agent.assets`.
