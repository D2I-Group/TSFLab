# Read the data profile

```bash
uv run tsf data analyze <preset>                 # configs/datasets/<preset>.toml
uv run tsf data analyze --path <file|store-dir> [--split-ratio 0.7 0.1 0.2] [--freq 10min]
uv run tsf data analyze <preset> --json          # print JSON instead of markdown
uv run tsf data analyze <preset> --write-card    # record fired data terms in the dataset card
uv run tsf data analyze <preset> --plot --split val --per-channel   # sample windows, per-channel TFB
```

Outputs `work_dirs/profiles/<name>/profile.json`, `profile.md`, and
`characteristics_<split>.csv` (`--out DIR` to move); `--config TOML` profiles the dataset
of any preset or run config.
Profiles read the preset's own chronological split with unscaled values. Presets whose
loader has no date column need `--freq` for period tags (for example `solar` is `10min`).
The TFB characteristics table (formerly `tsf data inspect`) describes one split
(`--split`, default train); the decision-safe sections above it are what to select on.

## Sections and how to use them

| Section | Read it for |
| --- | --- |
| `split`, `dataset.frequency` | Capacity: short train favors light models; steps per day names periods. |
| `train.quality` | Missing, zero, repeated, negative fractions; decides loss and imputation checks. |
| `train.scale` | Channel scale spread, heteroscedasticity, level-variance coupling, tails. |
| `train.periods` | Dominant periods with `peak_ratio`, `tag`, `harmonic_of`, ACF at the period; forecastability = 1 - spectral entropy. |
| `train.seasonal`, `train.trend` | STL-style strengths in 0..1 at the top period. |
| `train.stationarity` | Block mean and variance drift (`mean_drift` >= 0.5 is clearly non-stationary). |
| `train.cross_channel` | Mean absolute correlation (levels and differences), PC1 share, effective-rank ratio. |
| `train.anomalies` | Outlier fraction (5 MAD) and block-level jumps. |
| `shift.within_train`, `shift.train_to_val` | Decision inputs: level and scale drift visible without the test split. |
| `shift.train_to_test` | Diagnostic only. Never select, tune, or stop on it. |
| `lookback_candidates` | Train-only period multiples, ACF decay lag, benchmark defaults, capped by capacity. |
| `recommendations.fired` | Rules that fired: finding, hypothesis, components, models, slot options. |
| `recommendations.slots` | Per slot: static `defaults` plus options `promoted` by fired rules. |

Large panels are sampled to `--max-channels` (default 128) for spectral statistics; cheap
moments use every channel. Statistics fill missing values with channel means, so missingness
is reported separately and changes how results should be read.

## Profile to catalog

The mapping is data in `src/tsflab/data/profile_rules.toml`: each rule has an id, a
threshold, and the facts, components, models, and slot options it names. Rule ids are
also the measurable terms of the shared characteristics vocabulary: a dataset card's
`characteristics` are the rules that fired (`--write-card` records them with their
basis), and model and component cards list the terms they are designed for in `fits`.
`uv run tsf catalog match <preset>` intersects the two (add task terms with `--extra`,
for example `probabilistic-output`), so the profile and the cards point at the same
candidates. Examples: strong seasonality points to `series_decomposition`,
`periodic_query_bank`, `dominant_periods`; train-to-validation level shift to `revin`
and `last_value_center`; heteroscedastic or heavy-tailed data to `loss:mae`,
`quantile_head`, or `gaussian_parameter_head`; low forecastability to naive and linear
baselines. Thresholds are priors. A fired rule or a match is a hypothesis to test
against the baseline panel, never a conclusion. Do not edit the rules or cards
mid-study; propose changes as a separate repository change.

## Using the numbers

- Lookback: start from `lookback_candidates`; test two (one near a period multiple, one
  longer) rather than sweeping all.
- Data-dependent parameters follow the model card's `[data_params]` (`from = period`,
  `frequency`, `channels`, `nodes`, `seq_len`, `pred_len`, `graph`, `train-split`, plus a
  `rule`): for example a `period` parameter (CycleNet `cycle`, DLinear `kernel_size`) takes
  the top `periods.dominant[0].period`, rounded. Parameters without a `[data_params]`
  entry are generic hyperparameters you tune within the budget.
- Repeat the profile for every approved dataset; compare profiles before pooling claims.
