# Agent-cost experiment

Measures what research tasks cost a coding agent in TSFLab and in other
environments. Supports §4.1 (research tasks) and Appendix C (measurement
procedure) of the paper.

## Tasks

| Task | Input to the agent | Success (checked by `check_success.py`) |
|---|---|---|
| `reproduce` | Link to one recent forecasting paper without official code, not in TSFLab; datasets and prediction lengths of its main table | Finite MSE and MAE for every cell in `results/results.json`, and `results/report.md` |
| `autoresearch` | One dataset, the best TSFLab method on it, its validation MSE, a budget | A method with 3 validation seeds below the target at every prediction length, and one test read per seed |

Prompts are in `prompts/`; a task is a JSON file (see `tasks/example-reproduce.json`).

## Arms (`make_env.sh`)

| Arm | Working directory |
|---|---|
| `tsflab` | TSFLab at a fixed commit |
| `tsflab-noknow` | Same code and skills; every `README.md` and `reference.md` of a method, component, or dataset deleted; `tsf catalog` disabled. `card.toml` stays because data fetching and admission read it. |
| `tsflab-flat` | Same content as `tsflab`, but concatenated into one `KNOWLEDGE.md` (no layered access); `tsf catalog` disabled |
| `tsflab-notools` | Same knowledge and method code; `tsf run`, `tsf result`, `tsf data` disabled, runner entry points deleted, experiment skills removed |
| `tsflab` + `TSFLAB_COMMIT` | An earlier version of TSFLab (interface-version experiment) |
| `tslib` | Time-Series-Library at `TSLIB_COMMIT` |
| `tfb` | TFB at `TFB_COMMIT` |
| `empty` | Empty directory with a minimal PyTorch environment |

Every arm gets the same raw data directory (`DATA_DIR`), the same prompt, the
same model, and network access. The Python environment is installed before the
session, so installation is not measured.

## Agents

`AGENT=claude` (default) runs Claude Code; `AGENT=codex` runs Codex CLI
(`codex exec --json`, default model `gpt-6-luna`) with `CODEX_HOME` in the run
directory, linked to the login in `~/.codex`. `INSTALL=server` installs torch
for CUDA 12.6. `launch.sh <task> <out-root> <spec>...` starts one session per GPU,
detached; a spec is an arm or `arm@commit`.

## Running one session

```bash
DATA_DIR=/path/to/raw-data \
  experiments/agent-cost/run_session.sh tasks/<task>.json <arm> runs/<run-id> [model]
```

Options: `TIMEOUT_H` (default 8, whole session), `MAX_RESUMES` (default 0), `MAX_BUDGET_USD` (Claude only),
`GPU_SAMPLE_S` (default 5), `TSLIB_COMMIT`, `SKIP_INSTALL=1` (tests only).
Search requires one explicit `TRAIN_GPU`. Its wall deadline is capped by the
task's GPU-hours, a conservative bound on allocated time for that one GPU.
The queue defaults to one session per GPU. Search never resumes automatically.

The session runs Claude Code headless with an isolated configuration
(`CLAUDE_CONFIG_DIR` holding only the login credentials, `--strict-mcp-config`):
no user `CLAUDE.md`, memory, plugins, or MCP servers. The project's own
`CLAUDE.md`/`AGENTS.md` and skills load as in normal use. Incomplete tasks stop
normally. Only an explicitly configured `MAX_RESUMES` permits supervised
resumes for non-search tasks; these are recorded separately from the first attempt.

## Outputs per run

| File | Content |
|---|---|
| `stream.<n>.jsonl` | Claude Code stream-json, one event per line, with arrival time `t` |
| `stages.tsv` | Stage markers written by the agent (`stage <name>`) |
| `interventions.tsv` | Resumes |
| `gpu.csv` | GPU utilization and memory samples |
| `diff.numstat`, `diff.patch` | Changes against the baseline commit |
| `success.json` | Result of `check_success.py` |
| `metrics.json` | Result of `analyze.py` |
| `limits.json`, `terminal.json` | Enforced session limits and final stopping reason |
| `attempts/<n>/` | Results and success check before any optional resume |
| `usage-events.jsonl` | Codex usage-only metadata, without credentials or rollout text |

## Metrics (`analyze.py`)

- Success, wall-clock time, cost (USD, from Claude Code), turns, interventions.
- Every tool call is classified as `paper` (reading the target paper), `read`
  (reading or searching the environment: knowledge, code, documentation),
  `write`, `run`, or `other`.
- Discovery time and calls are measured; exact discovery/action/core token shares
  remain unknown. Content-share diagnostics live only in `estimated_*` fields.
- Total tokens come from native usage: Codex cumulative chains, Pi assistant
  completions, or deduplicated Claude result/message usage. Missing usage stays
  null; output text length is never a source for the formal token total.
- Stage time is measured. Call-time stage tokens require native usage deltas;
  stage breakdowns without them stay null. GPU-busy seconds use utilization >=10%.
- Code changes by kind (code, config, docs, results).
- Discovery errors (a re-implemented existing block, data prepared differently
  from the protocol, a private evaluation loop) are judged from `diff.patch` by
  review and recorded separately.

## Where to run

Core-process accounting: `python3 core_stats.py ARCHIVE OUTPUT.json --arm tsflab
--csv OUTPUT.csv`. Excludes old versions, pilot repetitions, supervision,
pure monitoring, and repeated healthy log probes. Every excluded tool event
has a file and line reference. Core tokens remain unknown without model-call
usage; text-size shares cannot be used to subtract monitoring costs. This
separate analysis does not convert invalid results into successful sessions.

Repeated launches are recoverable without duplicate compute: CLI `--output`
identifies one matrix request; a different matrix requires a new output path.
Within a research round, identical matrices reuse the saved sweep even if the
output path changes. Request receipts and creation use a crash-safe lock.
Concurrent callers wait on the original controller; completed cells are reused.
Use a new round or an unkeyed stateless request for an intentional repetition.

- Pipeline tests (`SKIP_INSTALL=1`, small budget): any machine.
- Reproduction sessions train models on one GPU: local GPU when it is free,
  otherwise one GPU on the server.
- Autoresearch sessions need more GPU time: server.

### Subscription and dollar-cost interpretation

CLI subscription login does not invalidate recorded token usage. Report input,
cached input, output tokens, completion, interventions and elapsed time separately.
`cost_usd_luna` is a normalized API list-price estimate, and
`cost_usd_reported` is the CLI-reported estimate; neither verifies a subscription
bill. `actual_billed_usd` remains null without billing evidence. Historical auth
mode is unknown unless recorded; current login status does not establish it.
Record quota/rate-limit failures separately from scientific failures, and match
model, reasoning settings and CLI versions for comparisons.
