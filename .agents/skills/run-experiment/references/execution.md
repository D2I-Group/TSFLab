# Optional execution controls

Load this reference only for budgets, queueing, tracking, cancellation, or recovery.
Discover APIs through `tsflab.experiments.infra.api.describe_modules()` or the optional
`tsf agent interface` adapter. The Agent owns decisions; services own atomic state and
execution constraints. CLI examples below do not mandate a subprocess hop.

Heavy runs execute on GPU machines or CI, never on the maintainer's local machine; the
Agent previews with `--dry-run`, hands off the exact command or the prepared directory,
and reads the records that come back under `work_dirs/` (a synced copy of the run
machine's `work_dirs/`). Keep scientific TOML independent of execution policy. Use `tsf run <config>
--policy <policy.toml>` for resource/budget/tracking settings. Preview the same
command with `--dry-run --json --compact` when a resource review or handoff is needed.
Otherwise submit once with `--json --compact --output results/completion.json`: run already validates the entire
matrix before compute starts. Avoid a separate preview/inspection model turn when
configs and resource scope are already settled.
Missing environment/data/weights are readiness failures, not model evidence.

Ordinary `tsf run CONFIG --json` blocks and returns metrics and timings when the
matrix finishes. Keep independent runs in one sweep; do not background each
cell and ask the Agent to monitor it. If a tool yields before the command ends,
wait on that same tool process instead of issuing new shell diagnostics. Set
the shell tool and any outer code-mode exec yield to 30000 ms initially; short
yields can cause additional model turns even though the library itself blocks.
For an already launched run/sweep, call `tsf run wait DIRECTORY --timeout 1800
--json` once: local code checks records, then returns metrics or errors. This
internal waiting uses no model calls. A wait timeout leaves training running;
continue waiting on the same directory. Read logs only after a failed result.
Automatic allocation queues within the configured resource wait timeout when
visible cards are busy. Explicit `--gpus` uses physical NVIDIA indices/full UUIDs,
never indices renumbered within a CUDA mask; invisible cards are rejected before
launch. Prefer `--gpus auto` in a restricted environment.
Use the printed run/sweep directory with `tsf run status`, `cancel`, or `resume`.
Resume preserves logical run IDs, verifies config/data/code fingerprints, skips
completed cells, and resumes the most recent committed checkpoint. Optional batch
recovery requires zero data-loader workers and replayable input. LatentTSF
pretraining and fixed-window evaluation have stage checkpoints; external state
requires paired runtime hooks. Rolling evaluation restarts. Do not reinterpret
weight-only initialization as full recovery.

For a CUDA sweep, `--gpus auto --jobs auto` selects visible GPUs with no compute
processes and sufficient policy `min_free_memory_mb`, then caps workers by the
matrix size and GPUs per run. `CUDA_VISIBLE_DEVICES` bounds discovery, including
an empty mask. Preview with the same flags. Occupancy and memory are rechecked
under crash-safe cooperative locks before launch; unavailable cards wait locally.
Unknown occupancy is excluded. Explicit `--jobs N --gpus 0,1` remains available.
Automatic selection is exclusive; CPU matrices need an explicit worker count.
External non-cooperating programs can still race after admission. Keep the same
GPU-hour budget for comparisons; parallelism reduces wall time, not necessarily
total compute or model tokens.

GPU allocation defaults to exclusive; optional sharing reserves declared memory
among cooperating local processes and is not physical isolation. Never infer that
free-memory admission guarantees fitting. Do not silently alter workload after
OOM. Automatic retry is opt-in and must remain within the declared budget.

Tracking is disabled by default. TensorBoard/W&B are optional mirrors of local
scalar evidence; select W&B online only when sending the configured metadata is
within the user's requested scope. Offline attempts remain grouped by logical run.
No tracking account, round, daemon, or policy file is required for an ordinary run.

Keyed API preparation uses `prepare_sweep(..., request_key=...)`; CLI `--output`
provides that key. Repeating a key reuses its matrix under a crash-safe lock; a
different matrix needs a different key. Identical matrices within a round reuse
their sweep regardless of key. Unkeyed stateless API requests remain independent.

Round run counts, parallel slots, wall time, and GPU-hours are executable limits.
The Agent defines research iterations and claims each stable operation ID once
when an iteration budget applies; matrix preparation and resume do not count. Reserve external tokens/USD with `tsf env usage`
before dispatch and settle the same operation ID afterward. Unreported external
reasoning and billing cannot be metered by this repository. Host permissions stay with the
host. Do not describe task permissions as an OS sandbox or launch another agent.

For persistent queue or cluster submission, prepare the matrix once with
`tsf run --prepare-only`. Queue controllers preserve identity on restart. Slurm
submission is external dispatch and requires authorization. Storage cleanup is a
preview unless `--apply` is supplied; preserve user data and never automate
cleanup outside the authorized scope. Use `tsf --format json <command>` when a
harness needs one machine-readable envelope for any command.
