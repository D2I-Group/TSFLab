---
name: setup-environment
description: "Install, repair, or verify the TSFLab Python environment and PyTorch backend, including the optional hub and realtime extras. Use for first-time setup, dependency failures, CUDA detection problems, or hardware changes."
---

# Set up the environment

Experiments module, step zero: produce a working, reproducible environment and report what was selected.

## Inputs

- The machine's hardware and the features needed (core, `hub` for weights
  bundles, `realtime` for live data sources, trackers only when requested).

## Steps

1. Set up automatically: `python3 scripts/setup_env.py [--extras models,data,experiments]`
   (`tsf env setup` once installed) detects the driver, picks the torch 2.14.1 profile
   (driver CUDA >= 13.0 -> `cuda130`; 12.6-12.9 -> `cuda126`, there is no cu128 wheel;
   older or no NVIDIA -> `cpu`), installs it, and checks that torch sees the GPU;
   `--dry-run` only prints the plan.
2. The commands it runs, for a manual install (add `hub`/`realtime` extras as needed):

```bash
# cuda130 (driver >= 580) and macOS/Windows cpu: the locked build
uv sync --frozen --python 3.12 --extra models --extra data --extra experiments
# cuda126 (driver 560-579) or Linux cpu: uv pip, then never re-sync
uv venv --python 3.12
uv pip install --torch-backend cu126 -e ".[models,data,experiments]" pytest   # or cpu
export UV_NO_SYNC=1
```

   `UV_TORCH_BACKEND`/`--torch-backend` only affect `uv pip`; `uv sync` and a
   syncing `uv run` always install the locked cu130 build. Never edit `uv.lock`
   or pin `+cuXXX` for one machine.
3. Verify and pick the run profile:

```bash
uv run tsf env doctor --json   # facts, torch build/GPU use, install + run profile
uv run tsf env audit --json
uv run tsf repo check --audit  # in a TSFLab checkout; skip in a standalone project
```

Run profiles are `configs/execution/{laptop-cpu,single-gpu,multi-gpu}.toml` for
`tsf run --policy`; doctor prints the `--gpus/--jobs` flags. Device and
`num_workers` stay in the run config. For one experiment, check readiness with
`uv run tsf env audit --config <run.toml> --json`; audits never change the env.

## Chain

- Module: Experiments.
- Reads: `tsf env doctor` and `tsf env audit` output.
- Produces: a reported environment (versions, backend, extras) recorded with runs.
- Hands off to: `run-experiment`, `diagnose-experiment`.

## Success

- Python and torch versions, the install and run profiles, accelerator visibility,
  the installed extras, and doctor warnings are reported; `uv.lock` is unchanged.

## Stop and hand off

- Do not change dependency pins to mask a driver mismatch; report it.
- Environment-caused run failures go to `diagnose-experiment`.
- Training machines (GPU hosts, CI, Slurm) get the same sync; this skill never starts
  a run. Local audits only describe the machine they ran on.
