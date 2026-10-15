"""Foreground durable sweeps: one subprocess per resolved run, no daemon required."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import partial
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid

from tsflab.experiments.infra.contracts import Cancellation
from tsflab.experiments.infra.environment import audit_environment, validate_experiment
from tsflab.experiments.infra.policy import ExecutionPolicy
from tsflab.experiments.infra.processes import terminate as _terminate, launch_run, monitor
from tsflab.experiments.infra import round_execution
from tsflab.experiments.infra.resources import lease_gpus
from tsflab.experiments.infra.runs import TERMINAL, prepare_run, read_run, verify_resume
from tsflab.experiments.infra.storage import canonical_hash, file_lock, write_json


class RequestConflictError(ValueError):
    """A persisted request cannot be rebound to a different or missing matrix."""


def validated_config(snapshot):
    from tsflab.experiments.config.schema.root import RootConfig
    from tsflab.catalog.registry.datasets import DATASET_REGISTRY, register_dataset_by_name
    from tsflab.catalog.registry.models import MODEL_CATALOG
    from tsflab.experiments.config.loader import validate_task_compatibility

    config = RootConfig.model_validate(snapshot)
    register_dataset_by_name(config.dataset.name)
    data = DATASET_REGISTRY.get(config.dataset.name)
    if data.params_schema:
        config.dataset.params = data.params_schema.model_validate(config.dataset.params)
    model = MODEL_CATALOG.get(config.model.name)
    config.model.params = model.validate_params(config.model.params)
    validate_task_compatibility(config.task.mode, data, model)
    return config


def _visible_devices(inventory):
    """GPU identifiers visible to this process: ``CUDA_VISIBLE_DEVICES`` entries, else every inventoried UUID."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    return [item.strip() for item in visible.split(",") if item.strip()] if visible is not None else [gpu["uuid"] for gpu in inventory]


def _auto_selected_gpus(inventory, devices, min_free_memory_mb):
    """Visible GPUs whose memory meets ``min_free_memory_mb``."""
    return [
        gpu["uuid"] for gpu in inventory
        if any(item.strip() in (gpu["index"], gpu["uuid"]) for item in devices)
        and float(gpu.get("total_mb", gpu["free_mb"])) >= min_free_memory_mb
    ]


def _assign_gpus(policy, cuda_configs) -> None:
    """Resolve the policy's GPU set and per-run GPU count for a matrix with CUDA runs."""
    from tsflab.experiments.infra.environment import gpu_inventory

    inventory = gpu_inventory()
    devices = _visible_devices(inventory)
    indices = sorted(
        {i for c in cuda_configs for i in c.experiment.runtime.device_ids}
    )
    if policy.resources.auto_select:
        policy.resources.gpus = _auto_selected_gpus(inventory, devices, policy.resources.min_free_memory_mb)
    elif not policy.resources.gpus:
        policy.resources.gpus = [devices[i] for i in indices if 0 <= i < len(devices)]
    from tsflab.experiments.infra.hardware import validate_gpu_visibility
    validate_gpu_visibility(policy.resources.gpus, inventory)
    if any(c.experiment.runtime.use_multi_gpu for c in cuda_configs):
        policy.resources.gpus_per_run = max(
            len(c.experiment.runtime.device_ids) for c in cuda_configs
        )


def _resolve_parallel(policy, configs, cuda_configs) -> None:
    """Replace ``--jobs auto`` with the number of runs the GPU set can hold at once."""
    if configs and len(cuda_configs) != len(configs):
        raise ValueError("--jobs auto requires a CUDA-only matrix; set an explicit CPU worker limit")
    policy.budget.max_parallel_jobs = max(1, min(
        len(configs), len(policy.resources.gpus) // policy.resources.gpus_per_run
    ))
    # Persist the resolved limit; later preflights must not undo round caps.
    policy.budget.auto_parallel = False


def resolve_policy(configs, policy):
    """Return a resolved copy; never mutate caller-owned policy objects."""
    policy = policy.model_copy(deep=True)
    cuda_configs = [c for c in configs if c.experiment.runtime.device == "cuda"]
    if policy.resources.auto_select and policy.resources.sharing:
        raise ValueError("automatic GPU selection requires exclusive allocation")
    if cuda_configs:
        _assign_gpus(policy, cuda_configs)

    if policy.budget.auto_parallel:
        _resolve_parallel(policy, configs, cuda_configs)

    return policy


def _check_cuda_run(config, policy) -> None:
    """Raise when the policy cannot schedule a CUDA run."""
    if not policy.resources.gpus:
        raise ValueError("no schedulable NVIDIA GPUs visible")
    if len(policy.resources.gpus) < policy.resources.gpus_per_run:
        raise ValueError("fewer eligible GPUs than gpus_per_run")
    if (
        policy.resources.gpus_per_run > 1
        and not config.experiment.runtime.use_multi_gpu
    ):
        raise ValueError("multiple GPUs per run require use_multi_gpu=true")


def _check_run(config, policy) -> None:
    """Raise on the first problem that keeps one run from executing under the policy."""
    validate_experiment(config)
    if (
        policy.recovery.checkpoint_every_batches
        and config.experiment.runtime.num_workers
    ):
        raise ValueError("batch recovery requires runtime.num_workers=0")
    if policy.resources.sharing and policy.resources.memory_per_run_mb <= 0:
        raise ValueError("shared GPUs require memory_per_run_mb > 0")
    if config.experiment.runtime.device == "cuda":
        _check_cuda_run(config, policy)


def preflight(configs, policy):
    configs = list(configs)
    policy = resolve_policy(configs, policy)
    checks = []
    environments = {}
    for config in configs:
        key = (config.experiment.runtime.device, config.experiment.work_dir)
        if key not in environments:
            environments[key] = audit_environment(
                policy, device=key[0], work_dir=key[1]
            )
        errors = []
        try:
            _check_run(config, policy)
        except Exception as exc:
            errors.append(str(exc))
        errors.extend(
            c["detail"] for c in environments[key]["checks"] if c["status"] == "failed"
        )
        checks.append(
            {
                "model": config.model.name,
                "dataset": config.dataset.name,
                "seed": config.experiment.random_seed,
                "pred_len": config.task.pred_len,
                "ok": not errors,
                "errors": errors,
            }
        )
    limit = policy.budget.max_runs
    budget_ok = limit is None or len(configs) <= limit
    return {
        "schema_version": 1,
        "ok": all(c["ok"] for c in checks) and budget_ok,
        "total_runs": len(configs),
        "budget_ok": budget_ok,
        "resolved_policy": policy.model_dump(mode="json"),
        "runs": checks,
        "environments": list(environments.values()),
    }


def _require_running_round(round_id) -> None:
    from tsflab.research.rounds import load_round
    if load_round(round_id)["status"] != "running":
        raise ValueError("research round is not running")


def _saved_request(receipt: Path, matrix: str) -> Path:
    """The sweep directory a request receipt already points to, if it still matches ``matrix``."""
    saved = json.loads(receipt.read_text())
    if saved["matrix_sha256"] != matrix:
        raise RequestConflictError("request key already belongs to a different matrix; use a new output path/request key")
    directory = Path(saved["directory"])
    if not (directory / "sweep.json").is_file():
        raise RequestConflictError("saved request sweep is missing; preserve its receipt and use a new request key")
    return directory


def prepare_sweep(loaded, policy, round_id=None, *, request_key=None):
    """Persist one matrix per keyed request or identical matrix in a research round.

    Stateless unkeyed requests remain independent. A different keyed matrix must
    use a new key; changing a seed creates a distinct round matrix automatically.
    """
    loaded = list(loaded)
    if not loaded:
        raise ValueError("experiment matrix is empty")
    if round_id:
        _require_running_round(round_id)
    if request_key is None and round_id is None:
        return _prepare_sweep(loaded, policy, round_id)
    matrix = canonical_hash(sorted(canonical_hash(item.config.model_dump(mode="json")) for item in loaded))
    identity = canonical_hash({"round": round_id, "request": matrix if round_id else str(request_key)})
    root = Path(loaded[0].config.experiment.work_dir).expanduser().resolve()
    receipt = root / "_requests" / (identity + ".json")
    with file_lock(receipt.with_suffix(".lock")):
        if receipt.exists():
            return _saved_request(receipt, matrix)
        directory = _prepare_sweep(loaded, policy, round_id)
        write_json(receipt, {"schema_version": 1, "matrix_sha256": matrix,
                             "directory": str(directory), "round": round_id})
        return directory


def _prepare_sweep(loaded, policy, round_id=None):
    loaded = list(loaded)
    if round_id:
        from tsflab.research.rounds import load_round

        state = load_round(round_id)
        if state["status"] != "running":
            raise ValueError("research round is not running")
        remaining = None if state["max_runs"] is None else state["max_runs"] - state["runs_used"]
        if remaining is not None and len(loaded) > remaining:
            raise ValueError(f"matrix needs {len(loaded)} runs; round has {remaining} remaining")
    if not loaded:
        raise ValueError("experiment matrix is empty")
    policy = resolve_policy([item.config for item in loaded], policy)
    root = Path(loaded[0].config.experiment.work_dir).expanduser().resolve()
    directory = root / "_sweeps" / uuid.uuid4().hex
    runs = [str(prepare_run(item.config, item.raw, item.sweep_keys)) for item in loaded]
    for path in runs:
        state = read_run(path)
        state.update(sweep=str(directory), round=round_id)
        write_json(Path(path) / "manifest.json", state)
    write_json(
        directory / "sweep.json",
        {
            "schema_version": 1,
            "runs": runs,
            "policy": policy.model_dump(mode="json"),
            "round": round_id,
            "created_at": time.time(),
            "elapsed_sec": 0,
            "gpu_hours": 0,
        },
    )
    return directory


def status(directory):
    directory = Path(directory).resolve()
    if (directory / "sweep.json").exists():
        state = json.loads((directory / "sweep.json").read_text())
        return {
            "directory": str(directory),
            **state,
            "runs": [status(path) for path in state["runs"]],
        }
    state = read_run(directory)
    runtime_path = directory / "runtime.json"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    # A lock is a stronger liveness signal than a PID, which may have been reused.
    if state["status"] == "running":
        try:
            with (
                file_lock(directory / ".dispatch.lock", blocking=False),
                file_lock(directory / ".run.lock", blocking=False),
            ):
                state["status"] = "interrupted"
        except BlockingIOError:
            pass
    return {"directory": str(directory), **state, "runtime": runtime}


def _config_fields(row: dict, config: dict) -> None:
    """Copy the run's identifying config values (model, dataset, horizon, seed, ...) into ``row``."""
    for section, key, label in (("model", "name", "model"), ("dataset", "name", "dataset"),
                                ("task", "pred_len", "pred_len"), ("evaluation", "split", "split"),
                                ("experiment", "random_seed", "seed"), ("training", "batch_size", "batch_size"),
                                ("training", "epochs", "epochs")):
        if key in config.get(section, {}):
            row[label] = config[section][key]
    optimizer = config.get("training", {}).get("optimizer", {})
    if "lr" in optimizer:
        row["learning_rate"] = optimizer["lr"]


def _sweep_parameters(config: dict, sweep_keys) -> dict:
    """Values of the dot-delimited sweep keys present in ``config``."""
    parameters = {}
    for key in sweep_keys:
        value = config
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                break
            value = value[part]
        else:
            parameters[key] = value
    return parameters


def _result_fields(row: dict, path: Path) -> None:
    """Copy the scientific results of a succeeded run's ``result.json`` into ``row``."""
    result = json.loads(path.read_text())
    for key in ("metrics", "train_time_sec", "test_time_sec", "n_windows", "checkpoint_path", "cost"):
        if key in result:
            row[key] = result[key]
    row["result_file"] = str(path)
    pretraining = path.parent / "checkpoints" / "pretraining.pth"
    if pretraining.is_file():
        row["pretraining_checkpoint"] = str(pretraining)


def completion_report(state):
    """Return scientific results and diagnostics without dumping full manifests."""
    config = state.get("config", {})
    row = {key: state[key] for key in ("directory", "run_id", "status", "error") if key in state}
    _config_fields(row, config)
    parameters = _sweep_parameters(config, state.get("sweep_keys", []))
    if parameters:
        row["parameters"] = parameters
    evaluation = config.get("evaluation", {})
    if evaluation.get("checkpoint_path"):
        row["source_checkpoint"] = evaluation["checkpoint_path"]
        row["source_checkpoint_sha256"] = evaluation.get("checkpoint_sha256")
    path = Path(state.get("directory", "")) / "result.json"
    if state.get("status") == "succeeded" and path.is_file():
        _result_fields(row, path)
    elif state.get("status") == "succeeded" and state.get("directory"):
        row["error"] = "completed run has no result artifact"
    return row


def controller_active(directory):
    """A failed cell may still be awaiting a configured automatic retry."""
    directory = Path(directory)
    for name in (".controller.lock", ".dispatch.lock"):
        if not (directory / name).exists():
            continue
        try:
            with file_lock(directory / name, blocking=False):
                pass
        except BlockingIOError:
            return True
    return False


def _section(config: dict, name: str) -> dict:
    return config.get(name) or {}


def _result_row(run: dict) -> dict | None:
    """One line of a finished run: model, horizon, seed, scored split, metrics, and cost."""
    path = Path(run.get("directory") or "") / "result.json"
    if not path.is_file():
        return None
    result = json.loads(path.read_text())
    config = run.get("config") or {}
    metrics = result.get("metrics") or {}
    return {
        "model": _section(config, "model").get("name"),
        "pred_len": _section(config, "task").get("pred_len"),
        "seed": _section(config, "experiment").get("random_seed"),
        "split": _section(config, "evaluation").get("split", "test"),
        **{k: metrics[k] for k in ("mse", "mae") if k in metrics},
        "train_s": round(result.get("train_time_sec") or 0.0, 1),
        **(result.get("cost") or {}),
    }


def _poll_until_done(directory, deadline: float, interval: float, clock, sleep):
    """Poll ``directory`` until no run is busy or ``deadline`` passes; return (runs, busy)."""
    while True:
        result = status(directory)
        runs = result.get("runs") or [result]
        # A failed cell may still be awaiting an automatic retry while the controller holds its lock.
        busy = any(run.get("status") not in TERMINAL for run in runs) or controller_active(directory)
        if not busy or clock() >= deadline:
            return runs, busy
        sleep(min(interval, max(0.0, deadline - clock())))


def _summarize_runs(runs):
    """Counts by status, result rows of succeeded runs, and the runs that did not succeed."""
    counts: dict[str, int] = {}
    results, unfinished = [], []
    for run in runs:
        state = run.get("status", "unknown")
        counts[state] = counts.get(state, 0) + 1
        error = run.get("error")
        if state == "succeeded":
            row = _result_row(run)
            if row or not run.get("directory"):  # a status without a run directory has no artifact to read
                results.extend([row] if row else [])
                continue
            error = error or "completed run has no result artifact"
        unfinished.append({"status": state, "directory": run.get("directory"), "error": error})
    return counts, results, unfinished


def wait(directory, timeout: float = 1800.0, interval: float = 15.0, *, clock=time.monotonic, sleep=time.sleep):
    """Block until every run under ``directory`` has ended or ``timeout`` seconds pass.

    Returns one compact result (counts by status and the runs that did not succeed), so an agent that
    started a sweep in the background waits with one call instead of polling logs. A timeout does not
    cancel training: call it again.
    """
    if not math.isfinite(timeout) or not math.isfinite(interval) or timeout < 0 or interval <= 0:
        raise ValueError("timeout must be nonnegative and interval positive")
    deadline = clock() + timeout
    runs, busy = _poll_until_done(directory, deadline, interval, clock, sleep)
    counts, results, unfinished = _summarize_runs(runs)
    return {
        "directory": str(Path(directory).resolve()),
        "ok": not busy and not unfinished,
        "done": not busy,
        "counts": counts,
        "results": results,
        "unfinished": unfinished,
    }


def cancel(directory):
    directory = Path(directory).resolve()
    if (
        not (directory / "manifest.json").is_file()
        and not (directory / "sweep.json").is_file()
    ):
        raise ValueError("expected a run or sweep directory")
    (directory / "cancel.request").touch()
    return {"directory": str(directory), "cancel_requested": True}


def _saved_matrix(directory: Path, sweep: bool, policy):
    """The saved sweep state, a private copy of the policy to run with, and the matrix's run directories."""
    saved = json.loads((directory / "sweep.json").read_text()) if sweep else {}
    if not sweep:
        attempts = read_run(directory).get("attempts", [])
        if attempts:
            saved["policy"] = attempts[-1].get("policy", {})
    policy = (
        policy or ExecutionPolicy.model_validate(saved.get("policy", {}))
    ).model_copy(deep=True)
    run_dirs = [Path(p) for p in saved["runs"]] if sweep else [directory]
    if policy.resources.auto_select or policy.budget.auto_parallel:
        policy = resolve_policy([validated_config(read_run(p)["config"]) for p in run_dirs], policy)
    return saved, policy, run_dirs


def _apply_round_budget(policy, round_id) -> None:
    """Cap the policy's parallelism, run timeout, and retries at the research round's budget."""
    from tsflab.research.rounds import load_round

    round_budget = load_round(round_id).get("budget", {})
    for field in ("max_parallel_jobs", "run_timeout_minutes", "max_retries"):
        bound = round_budget.get(field)
        if bound is not None:
            current = getattr(policy.budget, field)
            setattr(
                policy.budget,
                field,
                bound if current is None else min(current, bound),
            )


def _pending_runs(directory: Path, run_dirs, selected_run, policy):
    """Validate every selected cell and preflight the unfinished ones; return (pending, resolved policy)."""
    # Validate every cell before any compute, even when restoring a matrix.
    pending = []
    for path in run_dirs:
        if selected_run is not None and path != selected_run:
            continue
        config = validated_config(read_run(path)["config"])
        with file_lock(path / ".run.lock", blocking=False):
            state = verify_resume(path, config)
        if state["status"] == "succeeded":
            if not (path / "result.json").is_file():
                raise ValueError(f"completed run has no result artifact: {path}")
            continue
        pending.append(path)
    report = preflight(
        [validated_config(read_run(p)["config"]) for p in pending], policy
    )
    if not report["ok"]:
        raise ValueError(json.dumps(report))
    policy = ExecutionPolicy.model_validate(report["resolved_policy"])
    (directory / "cancel.request").unlink(missing_ok=True)
    for path in pending:
        (path / "cancel.request").unlink(missing_ok=True)
    return pending, policy


@dataclass
class _Attempt:
    """Mutable facts of one attempt, read by its cleanup even when launching it raised."""

    number: int
    reservation: int | None = None
    process: object = None
    outcome: str = "failed"
    launch: float | None = None
    gpus: list = field(default_factory=list)


class _Controller:
    """Runs the pending cells of one matrix in parallel and accounts their wall time and GPU hours."""

    def __init__(self, directory: Path, policy, saved: dict, round_id, cancelled: Cancellation):
        self.directory = directory
        self.policy = policy
        self.saved = saved
        self.round_id = round_id
        self.cancelled = cancelled
        self.stop = threading.Event()
        self.started = time.monotonic()
        self.usage_path = directory / "usage.json"
        usage = json.loads(self.usage_path.read_text()) if self.usage_path.exists() else saved
        self.prior_elapsed = usage.get("elapsed_sec", 0)
        self.hours = usage.get("gpu_hours", 0.0)
        self.active = {}
        self.mutex = threading.Lock()

    def exhausted(self):
        if self.cancelled():
            return "cancelled"
        if self.stop.is_set() or (self.directory / "cancel.request").exists():
            return "cancelled"
        if (
            self.policy.budget.max_wall_minutes
            and self.prior_elapsed + time.monotonic() - self.started
            >= self.policy.budget.max_wall_minutes * 60
        ):
            return "timed_out"
        with self.mutex:
            used = self.hours + sum(
                (time.monotonic() - value[0]) * value[1] / 3600
                for value in self.active.values()
            )
            write_json(
                self.usage_path,
                {
                    "elapsed_sec": self.prior_elapsed + time.monotonic() - self.started,
                    "gpu_hours": used,
                },
            )
        if self.policy.budget.max_gpu_hours and used >= self.policy.budget.max_gpu_hours:
            return "timed_out"
        return None

    def reason(self, path: Path):
        """Why the run at ``path`` must stop now, or None."""
        if self.policy.storage.max_run_gb is not None:
            from tsflab.experiments.infra.retention import storage_status

            if not storage_status(path, self.policy)["ok"]:
                return "storage_exhausted"
        return self.exhausted() or (
            "cancelled" if (path / "cancel.request").exists() else None
        )

    def worker(self, path: Path):
        with file_lock(path / ".dispatch.lock", blocking=False):
            reason = partial(self.reason, path)
            for retry in range(self.policy.budget.max_retries + 1):
                outcome = self._attempt(path, reason)
                if outcome != "failed":
                    break
            return {"directory": str(path), "status": outcome}

    def _attempt(self, path: Path, reason) -> str:
        """Run one attempt of the cell at ``path`` and record it; return its outcome."""
        state = read_run(path)
        attempt = _Attempt(number=len(state["attempts"]) + 1)
        state["attempts"].append(
            {
                "number": attempt.number,
                "started_at": time.time(),
                "status": "queued",
                "stage": "resources",
                "policy": self.policy.model_dump(mode="json"),
            }
        )
        state["status"], state["stage"] = "queued", "resources"
        for key in ("error", "diagnosis"):
            state.pop(key, None)
        write_json(path / "manifest.json", state)
        try:
            self._launch(path, attempt, reason)
        except Exception as exc:
            attempt.outcome = reason() or (
                "timed_out" if isinstance(exc, TimeoutError) else "failed"
            )
            state = read_run(path)
            state["error"] = f"{type(exc).__name__}: {exc}"
            write_json(path / "manifest.json", state)
        finally:
            self._settle(path, attempt)
        return attempt.outcome

    def _launch(self, path: Path, attempt: _Attempt, reason) -> None:
        """Lease GPUs, reserve round capacity, and run the cell's subprocess to its outcome."""
        resources = self.policy.resources
        if (
            read_run(path)["config"]["experiment"]["runtime"]["device"]
            != "cuda"
        ):
            resources = resources.model_copy(update={"gpus": []})
        round_id = self.round_id
        with ExitStack() as owned:
            attempt.gpus = owned.enter_context(
                lease_gpus(resources, cancelled=lambda: bool(reason()))
            )
            owned.callback(
                lambda: (
                    _terminate(attempt.process) if attempt.process is not None else None
                )
            )
            if reason():
                raise InterruptedError(reason())
            if round_id:
                attempt.reservation = round_execution.reserve(
                    round_id,
                    read_run(path)["run_id"],
                    path,
                    len(attempt.gpus),
                    self.policy.resources.wait_timeout_minutes,
                    reason,
                )
            attempt.launch = time.monotonic()
            with self.mutex:
                self.active[str(path)] = (attempt.launch, len(attempt.gpus))
            with (path / f"attempt-{attempt.number}.log").open("a") as log:
                attempt.process = launch_run(path, self.policy, attempt.gpus, attempt.number, log)
                attempt.outcome = monitor(
                    attempt.process,
                    path,
                    attempt.gpus,
                    attempt.launch,
                    self.policy.budget.run_timeout_minutes,
                    lambda: (
                        reason()
                        or (
                            round_execution.stop_reason(round_id)
                            if round_id
                            else None
                        )
                    ),
                )
            if attempt.outcome == "failed":
                _diagnose_failure(path, attempt.number)
            if (
                attempt.outcome == "succeeded"
                and read_run(path)["status"] != "succeeded"
            ):
                attempt.outcome = "failed"

    def _settle(self, path: Path, attempt: _Attempt) -> None:
        """Stop the attempt's process, account its GPU hours, and persist its final status."""
        if attempt.process and attempt.process.poll() is None:
            _terminate(attempt.process)
        with self.mutex:
            self.active.pop(str(path), None)
            if attempt.launch:
                self.hours += (time.monotonic() - attempt.launch) * len(attempt.gpus) / 3600
        if self.round_id and attempt.reservation is not None:
            round_execution.settle(
                self.round_id, attempt.reservation, read_run(path)["run_id"], attempt.outcome
            )
        state = read_run(path)
        state["status"] = attempt.outcome
        state["attempts"][-1].update(
            status=attempt.outcome,
            finished_at=time.time(),
            stage=state["stage"],
        )
        if state.get("error"):
            state["attempts"][-1]["error"] = state["error"]
        write_json(
            path / "attempts" / f"{attempt.number}.json", state["attempts"][-1]
        )
        write_json(path / "manifest.json", state)

    def run(self, pending, sweep: bool) -> list:
        """Run every pending cell, then persist the matrix's elapsed time and GPU hours."""
        results = []
        try:
            with ThreadPoolExecutor(
                max_workers=self.policy.budget.max_parallel_jobs
            ) as executor:
                futures = [executor.submit(self.worker, path) for path in pending]
                try:
                    for future in as_completed(futures):
                        results.append(future.result())
                except BaseException:
                    self.stop.set()
                    raise
        finally:
            write_json(
                self.usage_path,
                {
                    "elapsed_sec": self.prior_elapsed + time.monotonic() - self.started,
                    "gpu_hours": self.hours,
                },
            )
            if sweep:
                self.saved.update(
                    elapsed_sec=self.prior_elapsed + time.monotonic() - self.started,
                    gpu_hours=self.hours,
                )
                write_json(self.directory / "sweep.json", self.saved)
        return results


def _diagnose_failure(path: Path, attempt: int) -> None:
    """Record whether a failed attempt ran out of memory, and where its log is."""
    tail = (path / f"attempt-{attempt}.log").read_text()[
        -8000:
    ]
    state = read_run(path)
    state["diagnosis"] = (
        "out-of-memory"
        if "out of memory" in tail.lower()
        else "execution-failure"
    )
    state["log"] = str(path / f"attempt-{attempt}.log")
    write_json(path / "manifest.json", state)


def _matrix_report(directory: Path, results, run_dirs, pending, selected_run) -> dict:
    return {
        "schema_version": 1,
        "directory": str(directory),
        "ok": all(r["status"] == "succeeded" for r in results),
        "skipped": len(run_dirs) - len(pending),
        # Include already completed cells too: resume should return the
        # complete matrix without another result-discovery model turn.
        "runs": [completion_report({"directory": str(path), **read_run(path)})
                 for path in run_dirs if selected_run is None or path == selected_run],
    }


def execute(
    directory,
    policy=None,
    *,
    round_id=None,
    selected_run=None,
    cancelled: Cancellation = lambda: False,
):
    """Run/resume a persistent matrix; preserve completed cells and all attempts."""
    directory = Path(directory).resolve()
    sweep = (directory / "sweep.json").exists()
    if not sweep:
        parent = read_run(directory).get("sweep")
        if parent:
            return execute(
                parent,
                policy,
                round_id=round_id,
                selected_run=directory,
                cancelled=cancelled,
            )
    saved, policy, run_dirs = _saved_matrix(directory, sweep, policy)
    if round_id and saved.get("round") and round_id != saved["round"]:
        raise ValueError("cannot change the research round of an existing sweep")
    round_id = round_id or saved.get("round")
    if round_id:
        _apply_round_budget(policy, round_id)
    with file_lock(directory / ".controller.lock", blocking=False):
        pending, policy = _pending_runs(directory, run_dirs, selected_run, policy)
        saved["policy"] = policy.model_dump(mode="json")
        results = _Controller(directory, policy, saved, round_id, cancelled).run(pending, sweep)
        return _matrix_report(directory, results, run_dirs, pending, selected_run)


def saved_policy(directory):
    """Recover the last policy without accidentally resetting unspecified controls."""
    directory = Path(directory).resolve()
    if (directory / "sweep.json").is_file():
        return ExecutionPolicy.model_validate(
            json.loads((directory / "sweep.json").read_text())["policy"]
        )
    state = read_run(directory)
    if state.get("sweep"):
        return saved_policy(state["sweep"])
    return ExecutionPolicy.model_validate(
        state["attempts"][-1]["policy"] if state["attempts"] else {}
    )
