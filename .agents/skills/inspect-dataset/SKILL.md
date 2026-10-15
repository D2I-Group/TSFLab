---
name: inspect-dataset
description: "Inspect, profile, or visualize an existing TSFLab dataset. Use for resolved shapes, trend and seasonality characteristics, split checks, leakage checks, model-selection profiles, or raw sample plots; not for result plots or model predictions."
---

# Inspect a dataset

Data module, step two: report what a dataset preset actually loads, separating
observations from inferred characteristics. The `analyze` profile is the data
context AutoResearch reads before it picks models.

## Inputs

- A dataset preset (`configs/datasets/<name>.toml`) and the split of interest.

## Steps

```bash
uv run tsf catalog show <name>
uv run tsf data analyze <name> --per-channel --plot   # profile, TFB table, sample plot
uv run tsf data analyze <name> --write-card   # only when asked: record characteristics
```

Start from the card facts (`domain`, `characteristics`, `[shape]` frequency, length,
channels, `[protocol]`) and its Protocol and pitfalls section, then confirm them
against the files. `analyze` writes `work_dirs/profiles/<name>/` (`profile.{json,md}` from
train statistics plus labelled train/val/test shift, the TFB characteristics CSV of
`--split`, and with `--plot` `samples_<split>.png`); use it before choosing models.
Its fired rule ids are the card's data `characteristics`; `--write-card` records them
with their basis, which `tsf catalog match <name>` then reads.

Check split boundaries, tensor dimensions, missing values, target-channel
behavior, leakage across splits, inferred seasonal period, and adjacency or
covariate metadata.

## Chain

- Module: Data.
- Reads: dataset card (L1, then L2) and the loaded splits.
- Produces: profile `work_dirs/profiles/<name>/profile.{json,md}`, a measured-fact report, and (when asked) card `characteristics`.
- Hands off to: `run-autoresearch` (re-runs `tsf data analyze` per approved dataset), `run-experiment` (design), `add-dataset` (card fixes).

## Success

- A short report with artifact paths, the facts observed, and which
  characteristics are inferred.
- Any card fact the data contradicts is reported with the measured value; fix the
  card (`stats_basis = "measured"`, `--write-card`) only when asked to update it.

## Stop and hand off

- Inspection never modifies data; the card changes only with `--write-card` when asked. Fixes belong to `add-dataset`; model output
  plots belong to `analyze-results`; profile-driven search to `run-autoresearch`.
