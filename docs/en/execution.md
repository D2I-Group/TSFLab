# Running and recovering experiments

Start with the same commands as before:

```bash
uv run tsf run configs/runs/smoke_crib.toml --dry-run
uv run tsf run configs/runs/smoke_crib.toml
```

No research round, service, tracking account, or execution policy is required.
A **Run** is one resolved experiment. A **Sweep** is its experiment matrix.
A **Research Round** optionally associates runs with a question and budget.
Scientific choices stay in the experiment TOML. Operational settings belong to
an optional execution policy. All execution uses the same trainer and evidence.

## Check before execution

```bash
uv run tsf env audit
uv run tsf run configs/runs/smoke_crib.toml --dry-run
```

Environment audit checks Python and core package availability, requested device
availability, output access, free disk space, and requested optional tracking
packages. With a run config it also checks dataset paths, model artifacts, and
static task/loss compatibility. It does not install dependencies, download data,
change a driver, or prove that every model will fit in memory. CUDA execution
requires an available CUDA runtime and schedulable NVIDIA devices; it does not
silently fall back to CPU. Use a CPU configuration for CPU experiments.

`tsf run --dry-run` resolves scientific configuration without requiring local data.
`tsf run --dry-run` and `tsf env audit --config <toml>` also check execution
readiness. Normal runs perform the same preflight automatically.

## Inspect, cancel, or resume

Each launch prints its sweep directory. It contains references to run directories,
so configuration files do not have to be reconstructed after an interruption.

```bash
uv run tsf run status <run-or-sweep-directory>
uv run tsf run cancel <run-or-sweep-directory>
uv run tsf run resume <run-or-sweep-directory>
```

Status includes the current stage and a subprocess heartbeat. Cancel writes a
request that the managing process observes; it stops only its own process group.
The latest complete epoch remains recoverable. A forced interruption in the middle
of an epoch loses work since the last completed epoch.

Resume verifies the configuration, explicit local dataset content, and source /
lockfile fingerprints before starting. Completed sweep cells are skipped. Failed
and interrupted cells retain their logical run identity and create a new execution
attempt. Completed cells with missing result artifacts are reported as inconsistent.
Changing the model, data, or scientific protocol requires a new run.

Training writes two different artifacts:

- `checkpoints/best_checkpoint.pth`: weights selected by validation for evaluation.
- `checkpoints/latest.pth`: model, optimizer, learning-rate progress, AMP scaler,
  early stopping, callback state, completed epoch, timing, and random state for recovery.

Latest checkpoints are replaced atomically. The initial implementation resumes at
an epoch boundary. Completed training can proceed directly to evaluation without
repeating epochs. Model-specific pretraining currently restarts its pretraining
procedure before restoring the main training checkpoint. Mid-pretraining and
mid-batch optimizer recovery are not supported. Older weight-only checkpoints
cannot restore a complete training state.

Deterministic equivalence is tested on CPU with dropout and shuffled data.
Different hardware, nondeterministic kernels, or external-runtime state may prevent
bitwise equality. Keep the same environment for a formal continuation. Exact data
fingerprints cover explicit local files; external selectors need their data pinned
locally to provide the same guarantee.

## Optional tracking

Install only the integrations you use:

```bash
uv sync --frozen --python 3.12 --extra tensorboard --extra wandb   # or: uv pip install -e ".[tensorboard,wandb]"
```

Save an execution policy, for example `execution.toml`:

```toml
[tracking]
tensorboard = true
wandb = "offline"
project = "TSFLab"
tags = ["baseline"]
# Optional: sample forecast figures (may be mirrored to enabled backends)
# prediction_samples = 2
```

```bash
uv run tsf run configs/runs/smoke_crib.toml --policy execution.toml
tensorboard --logdir work_dirs/_runs
```

Local scalar events are always written. TensorBoard and W&B mirror training loss,
validation loss, learning rate, elapsed training time, and final test metrics.
They do not control training. Runtime backend failures warn and leave local evidence
intact. Missing requested integrations are caught before execution.

W&B defaults to disabled; `offline` keeps its files local. Set `wandb = "online"`
explicitly to send configuration and metrics to W&B using your existing login.
Prediction figures are disabled unless `prediction_samples` is positive; when
enabled they show selected target values in model-space units and are mirrored to
the selected backends. Raw datasets, checkpoints, and source files are not explicitly uploaded by this
integration. W&B itself collects standard runtime metadata. Each execution attempt
has a distinct W&B ID grouped under the stable local run ID, including offline
attempts. TensorBoard keeps one logical series and purges superseded steps when
resuming. Local events retain attempt labels for audit.

See the upstream [W&B initialization contract](https://docs.wandb.ai/models/ref/python/functions/init)
and [TensorBoard writer API](https://docs.pytorch.org/docs/stable/tensorboard).

## Optional budgets and GPU scheduling

```toml
[budget]
max_runs = 12
max_parallel_jobs = 2
run_timeout_minutes = 60
max_wall_minutes = 240
max_gpu_hours = 8
max_retries = 0

[resources]
gpus = ["0", "1"]
gpus_per_run = 1
min_free_memory_mb = 8000
wait_timeout_minutes = 30
min_free_disk_gb = 10
```

```bash
uv run tsf run <experiment.toml> --policy execution.toml
```

Ready-made machine presets live in `configs/execution/` (`laptop-cpu.toml`,
`single-gpu.toml`, `multi-gpu.toml`); `tsf env doctor` recommends one and prints the
`--gpus`/`--jobs` flags for this host. `[resources] threads_per_run` caps
OpenMP/MKL/OpenBLAS threads in each run process, which avoids CPU oversubscription
when many runs share a large host.

The budget applies to resolved runs, not just input files. Wall time includes
queue waits. GPU-hours count allocated devices multiplied by execution time, not
utilization. Live usage is persisted. Retries are disabled by default; when enabled,
they reuse the same run identity and checkpoint, retain attempts, and consume time
and GPU budget. No retry silently changes batch size or another scientific parameter.

The local scheduler leases GPUs exclusively, checks free memory, and maps assigned
physical devices to subprocess-local CUDA indices. Without an explicit GPU policy,
CUDA runs use the visible devices selected by their runtime configuration. GPU leases
coordinate local invocations sharing the same user resource directory. They cannot
reserve memory against unrelated programs or other users. Free-memory checks are
admission estimates, not a guarantee against OOM. Resource scarcity waits up to the
configured timeout. Multi-GPU requests require a matching `use_multi_gpu` experiment.

There is no daemon requirement. The controlling command must stay alive; on an
unexpected controller exit, inspect status before resuming. Locks prevent a live run
from being executed twice. Use the same working directory when resuming experiments
whose data/output paths are relative.

## Optional research rounds

```bash
uv run tsf research start --task experiment --goal "Does normalization help?" --max-runs 8
uv run tsf run <experiment.toml> --round <round-id> --policy execution.toml
```

Use `tsf research start --policy <policy.toml>` to apply an execution policy’s
budget across the whole round. `--max-iterations` optionally bounds explicit
iteration claims. Task-created rounds carry their declared run, parallel, and time budgets. These
limits are checked by execution in addition to the local policy. Resuming a logical
run does not count it as a new experiment. Recorded GPU time includes all attempts.
A round's wall-time window starts at round creation and does not reset on resume.

For iterative research, `tsf research iteration <round-id>` claims the next declared
iteration. This bounds recorded iterations; it cannot count an external agent's
unreported reasoning steps. Task permissions remain host/agent instructions, not an
OS sandbox. The library does not enforce an external LLM's token or billing limits.

## Comparable results

New CSV results include protocol and model-variant fingerprints. Reports rank within
a protocol and expose seed coverage, duplicate seeds, and standard deviation.
Legacy rows without these fingerprints remain visible but are not ranked. Protocols
include data content, task, evaluation, and training settings; a matching fingerprint
is a necessary mechanical check, not proof of scientific fairness. Expected seed
coverage includes planned run manifests, so a run that never produced metrics
remains visible as an incomplete model/seed cell.

## Interface discovery

```bash
uv run tsf agent interface
uv run tsf agent interface schema --json
```

`interface` groups basic experiment, execution, research, and asset workflows.
The schema command exports the strict optional execution-policy contract.
`run`, `env`, and `agent interface` support JSON for automation; existing
catalog, task, and admission commands keep their established public interfaces.
Use per-command `--help` instead of importing internal command modules.

## Scope of this version

The implemented resource backend is local POSIX execution (Linux/macOS), including
CPU and NVIDIA GPU scheduling. Slurm dispatch, shared-GPU packing, durable queue
priorities, batch-level recovery, automatic disk cleanup, and externally metered
LLM/cloud billing budgets are future extensions. Existing outputs are never
silently deleted to satisfy a quota.

## Optional operations

Start with `tsf run experiment.toml`. The following controls are only needed for
long jobs, shared machines, or automated research. `tsf agent interface` discovers the
public commands; `tsf agent interface schema --json` describes the policy. Existing
`--json` outputs remain compatible. To receive one versioned envelope, including
exit codes and errors from older commands, use
`tsf --format json <command> [arguments]`.

### Recovery granularity

Epoch checkpoints remain the default. Set `[recovery] checkpoint_every_batches = 20`
to also checkpoint after every 20 batches at a completed optimizer update. With
gradient accumulation, saving waits for a matching optimizer boundary. Batch
recovery currently requires `num_workers = 0`; it recreates the epoch iterator
and skips consumed batches before restoring the saved random state. Dataset
iteration must be replayable without external side effects. An interrupted
kernel or partially completed optimizer operation is replayed from the previous
committed checkpoint.

LatentTSF pretraining has its own optimizer and RNG checkpoint and resumes before
forecast training. Model-owned pretraining can accept an optional `stage_runner`
callable; the experiment runner injects checkpoint policy. Plain `model.pretrain`
keeps its local training loop and imports no infrastructure. The earlier model-level
checkpoint keywords are replaced by this explicit injection point. Stateful external runtimes can
implement paired `runtime_state_dict()` and `load_runtime_state_dict(state)`
hooks for training checkpoints. Opaque runtime state and in-progress remote API
requests cannot be reconstructed automatically. Fixed-window evaluation uses the
same optional batch interval to save predictions, RNG and runtime hooks, so
inference-only adapters can resume completed batches. The checkpoint stores
accumulated predictions and may be large; rolling evaluation still restarts its
window traversal. External adapters must provide the hooks for opaque state.

### Durable local queue

Prepare a matrix once, then use the returned sweep directory:

```bash
tsf run experiment.toml --policy execution.toml --prepare-only --json
tsf run --backend queue add /path/to/queue --run /path/to/sweep --priority 10
tsf run --backend queue work /path/to/queue --slots 2
tsf run --backend queue status /path/to/queue
tsf run --backend queue cancel /path/to/queue/JOB_ID
```

Higher priorities start first; ties use arrival order. Priority does not preempt
running work. Each slot owns an independent sweep, whose policy still controls
its run concurrency. Controllers run separately from the queue worker. Restarting
the worker detects their locks and adopts them without launching duplicates.
If a controller dies, its child stops and retains inherited GPU leases until it
exits; a running queue worker retries the abandoned job using saved checkpoints.
The worker itself must be restarted by the user or a host service after a machine
reboot. `--once` launches currently available jobs and returns; it does not install
a service. Logs stay in each queue job's `controller.log`.

For cooperative GPU sharing, set `[resources] sharing = true`,
`memory_per_run_mb = 4096`, and `max_processes_per_gpu = 2`. Shared and exclusive
leases coordinate through the same host locks. These are admission reservations,
not hardware memory isolation; other software and undeclared peak usage can
still cause OOM. Default allocation remains exclusive.

### Slurm

On a shared filesystem with the same Python environment, prepare a sweep and run:

```bash
tsf run --backend slurm submit /path/to/sweep --partition gpu --gpus 1 --minutes 120
tsf run --backend slurm status /path/to/sweep
tsf run --backend slurm cancel /path/to/sweep
```

Submission is explicit and persists the job receipt. Slurm allocates the job's
resources; the normal runner executes within the allocation. The script uses the
submitting Python path and working directory, so both must exist on compute nodes.
Preparation currently audits the submitting environment, including requested GPUs;
use a suitable allocation to prepare CUDA workloads. A lost submission receipt is
kept as uncertain and blocks automatic resubmission to avoid duplicate spending.
Reconcile that receipt with the cluster before retrying. This is a transport for
one prepared sweep, not a distributed training implementation.

The adapter follows the documented parsable interfaces of
[sbatch](https://slurm.schedmd.com/sbatch.html) and
[sacct](https://slurm.schedmd.com/sacct.html).

### Storage and external spending

`[storage] max_run_gb = 20` stops a managed run when its output directory exceeds
that limit. This is a periodically checked limit, not a filesystem quota: an
atomic checkpoint can briefly exceed it. `keep_epoch_checkpoints = 3` controls
explicit cleanup of obsolete `epoch_*.pth` files. Cleanup preserves best/latest,
pretraining state, top-k references, and unrelated artifacts:

```bash
tsf env storage status /path/to/run --policy execution.toml
tsf env storage cleanup /path/to/run --policy execution.toml
tsf env storage cleanup /path/to/run --policy execution.toml --apply
```

The first cleanup command only previews. Active runs cannot be cleaned.

For API/LLM/cloud operations outside the runner, set `[budget] max_tokens` and
`max_cost_usd`. The Harness must reserve an upper bound before making a call and
settle its actual usage using the same operation ID:

```bash
tsf env usage reserve /path/to/ledger --policy execution.toml --operation call-001 --tokens 4000 --cost-usd 0.05
# Execute the external call only after reservation succeeds.
tsf env usage settle /path/to/ledger --operation call-001 --tokens 1200 --cost-usd 0.02
tsf env usage status /path/to/ledger
```

Retries with the same amounts are idempotent; concurrent admission is locked.
Unsettled reservations continue to consume budget. Prices are supplied by the
caller in USD; the library does not guess provider pricing or observe calls made
outside this protocol. Stored ledger limits cannot be loosened by omitting a
policy on a later request. Use one ledger per intended spending scope.

The Agent defines a research iteration and claims it explicitly with
`claim_iteration(round_id, operation="hypothesis-a")` or
`tsf research iteration <round-id> --operation hypothesis-a`. Reusing that ID is
idempotent. Preparing or resuming matrices does not consume iterations; one
iteration may contain several matrices. This bounds declared workflow steps,
not private reasoning or the number of prepared artifacts.

### Capability filtering

`tsf catalog search --kind model --capability pretraining-stage --json` filters model
metadata. Repeat `--capability` to require every capability and optionally add
text terms. This does not introduce another model registry or architecture tree.

## Use individual Python modules

The supported facade is `tsflab.experiments.infra.api`. It loads only the requested
implementation; importing it does not initialize a run, query hardware, start a
worker, or connect to a tracking service. Discover the exported functions and their
operational requirements with `tsf agent interface api --json`. Inspect just one
configuration section with `tsf agent interface schema --module storage --json`.

| Responsibility | Focused module | Required context |
| --- | --- | --- |
| Local records and locks | `tsflab.experiments.infra.storage` | A local filesystem; locks use POSIX |
| Scientific identity | `tsflab.experiments.infra.fingerprints` | Config or repository/installed-code context |
| External spending | `tsflab.experiments.infra.accounting` | A ledger directory and optional `Budget` |
| Metrics | `tsflab.experiments.infra.tracking` | A directory and run label; mirrors are optional |
| Hardware inventory | `tsflab.experiments.infra.hardware` | NVIDIA CLI when querying real CUDA devices |
| GPU admission | `tsflab.experiments.infra.resources` | `Resources`, a lease directory, inventory provider |
| Storage inspection | `tsflab.experiments.infra.retention` | An existing directory and optional `Storage` |
| State recovery | `tsflab.experiments.infra.checkpoint` | PyTorch training state; external state hooks when needed |
| Result comparison | `tsflab.experiments.infra.comparison` | Result records and optional planned cells |
| Queue bookkeeping | `tsflab.experiments.infra.queue` | Local queue records; execution can be injected |
| Experiment orchestration | `tsflab.experiments.infra.execution` | Resolved experiment configs and manifests |
| Cluster transport | `tsflab.experiments.infra.slurm` | Prepared experiment, Slurm and shared filesystem |

The orchestrator composes services; low-level services do not require it. The
`ExecutionPolicy` groups module settings for the CLI, but independent services
accept their own settings. Existing calls that pass a full policy to storage
inspection/cleanup remain compatible. Scientific model/config catalogs remain the
source of scientific metadata; interface discovery only lists callable exports.

### Standalone metrics, spending and storage

```python
from tsflab.experiments.infra.api import Budget, Storage, Tracker, UsageLedger, storage_status

with Tracker("output/metrics", run_id="my-script") as metrics:
    metrics.log({"loss": 0.25}, step=1)

ledger = UsageLedger("output/spending", Budget(max_tokens=10000, max_cost_usd=1))
ledger.reserve("request-1", tokens=4000, cost_usd=0.10)
# Invoke the external service only after reservation succeeds.
ledger.settle("request-1", tokens=800, cost_usd=0.02)

capacity = storage_status("output/metrics", Storage(max_run_gb=2))
```

No experiment config, research round, GPU, daemon or tracking account is involved.
`Tracker` creates its directory and closes mirrors on context exit, including an
exception. Closing it repeatedly is safe; logging after close is an error.
Ledger limits persist across Python objects and processes. Cleanup still requires
a managed run and explicit application; generic storage inspection never deletes.

### Standalone resource admission

```python
from tsflab.experiments.infra.api import Resources, lease_gpus

with lease_gpus(Resources(gpus=["0"]), directory="output/gpu-leases") as devices:
    # Run work using the assigned device UUIDs.
    pass
```

All cooperating processes must use the same lease directory. The function yields
UUIDs; it does not rewrite `CUDA_VISIBLE_DEVICES` for independent callers. An
`inventory=` callable can supply device metadata for a different discovery
mechanism or tests. It must provide the same index/UUID/free-memory fields as
`gpu_inventory()` and total memory when using shared admission. For subprocess
ownership that survives parent failure, pass `devices.descriptors` through
`subprocess.Popen(pass_fds=...)` and stop the child before leaving the context;
the built-in execution adapter handles this lifetime contract.

### Explicit cancellation and execution adapters

`FileCancellation(path)` is a callable signal: querying it does not create files;
`request()` persists cancellation. `any_cancelled(...)` combines signals.
`execute(directory, cancelled=signal)` accepts caller cancellation independently
of run cancellation files. Queue execution passes a per-job signal directly and
does not modify process environment variables.

For in-process integration, `enqueue(..., validate=callable)` accepts an explicit
input validator and `run_job(job_directory, executor=callable)` accepts an executor
with signature `executor(directory, *, cancelled)`, returning a mapping with `ok`.
The validator must reject unsuitable inputs; the executor must honor cancellation
and own its cleanup. The structural interfaces are in `tsflab.experiments.infra.contracts`.
They require no base class or plugin registration. To use the same executor in
both in-process and detached modes, persist its importable reference with
`enqueue(..., executor="my_package.jobs:execute")`; the CLI accepts
`tsf run --backend queue add ... --executor my_package.jobs:execute`. Both modes resolve and
validate that callable against the same contract. The module must be installed
or importable in the worker environment. Closures remain in-process only.
The default remains the standard experiment adapter.

Training accepts a metrics sink with `start`, `log` and `close` behavior; runtime
checkpoint extensions use the paired state hooks documented above. These
contracts provide extension points without another model registry or a required
service container. Queue locks and GPU leases are local coordination, not a
multi-machine scheduler; Slurm remains a separate explicit transport.


## Agent-owned work and library guarantees

TSFLab augments the current Agent. The Agent uses its native reasoning,
editing, browsing and presentation tools to choose hypotheses, design experiments,
diagnose failures, interpret comparisons and communicate conclusions. It does
not need to render a task or run a CLI command for those activities.

Library calls provide validated computation and durable guarantees: resolved
configuration contracts, protocol-aware aggregation, atomic evidence, budget
admission, GPU leases, cancellation and state restoration. Native Agent ability
does not replace these guarantees with estimates or manually edited records.
The Python API and CLI are two entry points to these services. CLI examples are
convenient adapters, not a mandatory cognitive workflow. Statistical calculations
must still be reproducible and tied to source artifacts, whether the Agent uses
the supplied comparison function or an appropriate analysis tool.

Persistent research context is optional. Existing conversation and native plans
can hold the Agent's working context. When limits must span several experiment
matrices, use a research round for machine-enforced budget state and references
to important evidence; do not duplicate the whole conversation there.

```python
from tsflab.experiments.infra.api import create_round, add_event, prepare_task

# Optional: inspect a reusable task without creating files or launching work.
suggestion = prepare_task("autoresearch", {"question": "Does this ablation help?"})
assert suggestion["round"] is None

# When a cross-experiment budget is needed, persist just its durable scope.
round_state = create_round(task="ablation", goal="Compare the agreed ablation",
                           max_runs=4, budget={"max_runs": 4})
add_event(round_state["id"], "hypothesis", "Record the Agent's actual hypothesis")
# Pass round_id=round_state["id"] to prepared/managed experiment execution.
```

`prepare_task(..., persist=True)` explicitly prepares a template-backed round and
prompt. The compatible `tsf agent task start` adapter calls this same service.
Neither executes a reasoning loop, chooses the next experiment, nor starts another
Agent. Queue workers schedule computation only. External-runtime state hooks and
spending ledgers augment the host Agent without claiming to observe its private
reasoning or unreported external usage.


## Consistent invocation and validation

`preflight(configs, policy)` never changes the supplied policy. Its report contains
`resolved_policy`; `resolve_policy(configs, policy)` returns a fresh validated-policy
object with device discovery applied. Managed execution uses that resolved copy.
Round clamping also operates on a copy.

`create_round(..., budget={"max_runs": 4})` enforces the same limit as the legacy
`max_runs=4` argument. Supplying conflicting values fails before creating files.
Round records retain the top-level field as a compatibility projection; budget
values are normalized and validated on creation and loading.

For a common result/error envelope in Python, use `invoke(function, *args, **kwargs)`
and its `OperationResult.to_dict()`. `tsf --format json ...` uses that same schema
without launching another CLI process. Enhanced command adapters publish their
structured domain data directly. Older commands retain their original payload or
explicit textual data for compatibility; legacy subprocess tools may still run
as part of their actual work. CLI stream capture is serialized; use direct APIs
for concurrent integrations. Queue executors must return a mapping containing a
boolean `ok`; contract failures are recorded as structured errors.
