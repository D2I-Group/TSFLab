---
name: add-dataset
description: "Register a new dataset in TSFLab, from a standard CSV, a custom loader, or a traffic bundle, including fetching or converting its files into loader-ready data. Use for dataset integration, data preparation, and preset configuration; not for profiling an existing dataset."
---

# Add a dataset

Data module, step one: make one dataset runnable through a registered loader, a
strict parameter schema, a preset, and a card. The card and the profile from
`inspect-dataset` are the context AutoResearch reads about data.

## Inputs

- The source files and their layout, target and feature columns, frequency,
  intended task modes, split policy, and any adjacency.
- Which pattern fits, smallest first: `custom` (standard CSV), `single` (a distinct
  loader), the traffic converter (values plus adjacency), or an existing store
  loader such as `realtime_panel_st` / `realtime_panel_ts`.

## Steps

1. Keep bytes in `dataset/`, loader and schema code in `src/tsflab/data/`, the
   preset in `configs/datasets/`, and the card in `catalog/datasets/`. Loader, preset,
   and card changes need a TSFLab checkout; in a standalone project, only plan them.
2. Get the data loader-ready (download a published preset, window a CSV, convert a
   traffic bundle, or build the UltraTraffic history store of the traffic tracks) per
   [references/prepare.md](references/prepare.md); never alter the source files.
3. Scaffold a CSV-backed preset; for traffic bundles read
   `uv run tsf data prepare --from traffic --help` and pass explicit inputs and splits:

   ```bash
   uv run tsf data add my_data --pattern custom \
     --path ./dataset/my_data/my_data.csv --target OT   # also measures characteristics into the card
   ```

4. Keep `[dataset]` to `name`, optional display/track fields, `path`, optional `id`,
   and `[dataset.params]`. Loader options live in a strict registered schema that
   rejects misspellings.
5. Write the card `catalog/datasets/<name>/` (`card.toml` facts and a `README.md`
   with Overview, Protocol and pitfalls) from measured or cited facts per
   [references/card.md](references/card.md) (`data add` already recorded the measured
   characteristics; re-run `tsf data analyze my_data --write-card` if the data changes), then:

   ```bash
   uv run tsf data analyze my_data --plot   # look at the series (profile, TFB table, samples)
   uv run tsf repo check --audit            # card facts, no TODO
   ```

## Chain

- Module: Data.
- Reads: preset `configs/datasets/<name>.toml` and the card at L1/L2 for conventions of sibling presets.
- Produces: preset, card `catalog/datasets/<name>/` (characteristics, source and license, shape, protocol, pitfalls), loader-ready bytes in `dataset/`.
- Hands off to: `inspect-dataset` (profile), `run-experiment` (preset), `run-autoresearch` (`tsf catalog match <name>`).

## Success

- Train-only scaling, correct feature/target selection, stable split boundaries,
  and adjacency injection where declared.
- `uv run tsf repo check --audit` passes: card facts
  curated, no `TODO`, `characteristics` recorded with their basis.

## Stop and hand off

- Stop before overwriting an existing dataset or output directory unless asked.
  Publishing files needs explicit authorization and redistributable terms.
- Profile the result with `inspect-dataset`.
