# Retrieve candidates and run the baseline panel

## Progressive retrieval

Start from the data, then read the shortest card level and escalate only for survivors:

1. Match: `uv run tsf catalog match <preset...> [--extra TERM...] [--json]` ranks models
   and components (grouped by composition slot) whose card `fits` overlap the dataset
   card's `characteristics`, and lists generic (`any`) components. Unannotated cards
   do not match; widen with lexical shortlists (`uv run tsf catalog search --kind
   model <terms>`, `--kind component`) and the fired rules' `models` and `components`. A match
   is a hypothesis to test against the baseline panel, not a verdict.
2. L1 for the shortlist (about 8 candidates), in one call: `uv run tsf catalog show
   <name> <name>...` gives each
   model's card facts (`fits`, `fidelity`, composition, `data_params`, `[admission]`),
   its README (Idea, When to use, Configure, Differences), and runtime facts (`config`,
   `smoke_config`, `task_modes`, `output_type`), or a component's role, slot, fits,
   interface, and consumers.
3. L2 (`tsf catalog show <finalist...> --depth 2`, adds `reference.md`) only for at most 3 finalists. `--depth 3`
   lists the files to open.

Drop a candidate as soon as a shallow level excludes it: wrong task mode (time_series vs
spatiotemporal vs covariate), a required input the data lacks, an inference-only or
artifact-gated model, an admission that is not `passed`, or capacity far above the
train length. Note each exclusion and its
reason as a round `decision` event.

## Baseline panel

Run before any search, on the same split, horizon, lookback, metric, and seeds:

- `NLinear` or `RLinear` (normalized linear anchor) and `DLinear` (decomposition anchor).
- `PatchTST` (channel-independent deep anchor).
- `iTransformer` (channel-mixing anchor) when `train.cross_channel` is applicable.
- `ExpSmoothingTS` or `ARIMATS` when forecastability is low, as a classical floor.
- Plus the single best model the fired rules name, if it is not already in the panel.

Use presets in `configs/models/<Name>.toml`; a run file extends `configs/base.toml`, one
`configs/datasets/<preset>.toml`, and one model preset, and overrides only scientific
variables. Set each model's `[data_params]` from the profile before the panel runs, so
baselines are not handicapped by a wrong period or channel count; keep generic
hyperparameters at preset defaults for the panel. Preview the matrix with `uv run tsf run <run.toml> --dry-run` before any
execution; when authorized, the run machine (GPU host or CI, not the local machine)
executes `uv run tsf run <run.toml> --round <round-id>`. Prefer one horizon and one
seed for the panel; the panel costs at most a third of the run budget.

The panel result fixes the bar: the best baseline per dataset (by validation loss, then
confirmed on test) and the spread across baselines. If a simple baseline is already within
seed noise of the best, say so; deep models then need a large margin to be worth keeping.
