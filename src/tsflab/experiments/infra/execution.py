"""Foreground durable sweeps: one subprocess per resolved run, no daemon required."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import json
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
from tsflab.experiments.infra.storage import file_lock, write_json


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


def resolve_policy(configs, policy):
    """Return a resolved copy; never mutate caller-owned policy objects."""
    policy = policy.model_copy(deep=True)
    cuda_configs = [c for c in configs if c.experiment.runtime.device == "cuda"]
    if cuda_configs and not policy.resources.gpus:
        from tsflab.experiments.infra.environment import gpu_inventory

        inventory = gpu_inventory()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        devices = visible.split(",") if visible else [gpu["uuid"] for gpu in inventory]
        indices = sorted(
            {i for c in cuda_configs for i in c.experiment.runtime.device_ids}
        )
        policy.resources.gpus = [devices[i] for i in indices if 0 <= i < len(devices)]
        if any(c.experiment.runtime.use_multi_gpu for c in cuda_configs):
            policy.resources.gpus_per_run = max(
                len(c.experiment.runtime.device_ids) for c in cuda_configs
            )

    return policy


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
            validate_experiment(config)
            if (
                policy.recovery.checkpoint_every_batches
                and config.experiment.runtime.num_workers
            ):
                raise ValueError("batch recovery requires runtime.num_workers=0")
            if policy.resources.sharing and policy.resources.memory_per_run_mb <= 0:
                raise ValueError("shared GPUs require memory_per_run_mb > 0")
            if config.experiment.runtime.device == "cuda" and not policy.resources.gpus:
                raise ValueError("no schedulable NVIDIA GPUs visible")
            if (
                config.experiment.runtime.device == "cuda"
                and policy.resources.gpus_per_run > 1
                and not config.experiment.runtime.use_multi_gpu
            ):
                raise ValueError("multiple GPUs per run require use_multi_gpu=true")
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


def prepare_sweep(loaded, policy, round_id=None):
    loaded = list(loaded)
    if round_id:
        from tsflab.research.rounds import load_round

        if load_round(round_id)["status"] != "running":
            raise ValueError("research round is not running")
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


def _result_row(run: dict) -> dict | None:
    """One line of a finished run: model, horizon, seed, scored split, metrics, and cost."""
    path = Path(run.get("directory") or "") / "result.json"
    if not path.is_file():
        return None
    result = json.loads(path.read_text())
    config = run.get("config") or {}
    metrics = result.get("metrics") or {}
    return {
        "model": (config.get("model") or {}).get("name"),
        "pred_len": (config.get("task") or {}).get("pred_len"),
        "seed": (config.get("experiment") or {}).get("random_seed"),
        "split": (config.get("evaluation") or {}).get("split", "test"),
        **{k: metrics[k] for k in ("mse", "mae") if k in metrics},
        "train_s": round(result.get("train_time_sec") or 0.0, 1),
        **(result.get("cost") or {}),
    }


def wait(directory, timeout: float = 1800.0, interval: float = 15.0, *, clock=time.monotonic, sleep=time.sleep):
    """Block until every run under ``directory`` has ended or ``timeout`` seconds pass.

    Returns one compact result (counts by status and the runs that did not succeed), so an agent that
    started a sweep in the background waits with one call instead of polling logs.
    """
    deadline = clock() + timeout
    while True:
        result = status(directory)
        runs = result.get("runs") or [result]
        pending = [run for run in runs if run.get("status") not in TERMINAL]
        if not pending or clock() >= deadline:
            break
        sleep(min(interval, max(0.0, deadline - clock())))
    counts: dict[str, int] = {}
    results = [row for run in runs if run.get("status") == "succeeded" and (row := _result_row(run))]
    for run in runs:
        counts[run.get("status", "unknown")] = counts.get(run.get("status", "unknown"), 0) + 1
    return {
        "directory": str(Path(directory).resolve()),
        "ok": not pending and set(counts) <= {"succeeded"},
        "done": not pending,
        "counts": counts,
        "results": results,
        "unfinished": [
            {"status": run.get("status"), "directory": run.get("directory"), "error": run.get("error")}
            for run in runs if run.get("status") != "succeeded"
        ],
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
    saved = json.loads((directory / "sweep.json").read_text()) if sweep else {}
    if not sweep:
        attempts = read_run(directory).get("attempts", [])
        if attempts:
            saved["policy"] = attempts[-1].get("policy", {})
    policy = (
        policy or ExecutionPolicy.model_validate(saved.get("policy", {}))
    ).model_copy(deep=True)
    if round_id and saved.get("round") and round_id != saved["round"]:
        raise ValueError("cannot change the research round of an existing sweep")
    round_id = round_id or saved.get("round")
    if round_id:
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
    run_dirs = [Path(p) for p in saved["runs"]] if sweep else [directory]
    with file_lock(directory / ".controller.lock", blocking=False):
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
        saved["policy"] = policy.model_dump(mode="json")
        stop = threading.Event()
        started = time.monotonic()
        usage_path = directory / "usage.json"
        usage = json.loads(usage_path.read_text()) if usage_path.exists() else saved
        prior_elapsed = usage.get("elapsed_sec", 0)
        hours = usage.get("gpu_hours", 0.0)
        active = {}
        mutex = threading.Lock()

        def exhausted():
            if cancelled():
                return "cancelled"
            if stop.is_set() or (directory / "cancel.request").exists():
                return "cancelled"
            if (
                policy.budget.max_wall_minutes
                and prior_elapsed + time.monotonic() - started
                >= policy.budget.max_wall_minutes * 60
            ):
                return "timed_out"
            with mutex:
                used = hours + sum(
                    (time.monotonic() - value[0]) * value[1] / 3600
                    for value in active.values()
                )
                write_json(
                    usage_path,
                    {
                        "elapsed_sec": prior_elapsed + time.monotonic() - started,
                        "gpu_hours": used,
                    },
                )
            if policy.budget.max_gpu_hours and used >= policy.budget.max_gpu_hours:
                return "timed_out"
            return None

        def worker(path):
            nonlocal hours
            with file_lock(path / ".dispatch.lock", blocking=False):

                def reason():
                    if policy.storage.max_run_gb is not None:
                        from tsflab.experiments.infra.retention import storage_status

                        if not storage_status(path, policy)["ok"]:
                            return "storage_exhausted"
                    return exhausted() or (
                        "cancelled" if (path / "cancel.request").exists() else None
                    )

                for retry in range(policy.budget.max_retries + 1):
                    number = None
                    process = None
                    outcome = "failed"
                    launch = None
                    gpus = []
                    state = read_run(path)
                    attempt = len(state["attempts"]) + 1
                    state["attempts"].append(
                        {
                            "number": attempt,
                            "started_at": time.time(),
                            "status": "queued",
                            "stage": "resources",
                            "policy": policy.model_dump(mode="json"),
                        }
                    )
                    state["status"], state["stage"] = "queued", "resources"
                    for key in ("error", "diagnosis"):
                        state.pop(key, None)
                    write_json(path / "manifest.json", state)
                    try:
                        resources = policy.resources
                        if (
                            read_run(path)["config"]["experiment"]["runtime"]["device"]
                            != "cuda"
                        ):
                            resources = resources.model_copy(update={"gpus": []})
                        with ExitStack() as owned:
                            gpus = owned.enter_context(
                                lease_gpus(resources, cancelled=lambda: bool(reason()))
                            )
                            owned.callback(
                                lambda: (
                                    _terminate(process) if process is not None else None
                                )
                            )
                            if reason():
                                raise InterruptedError(reason())
                            if round_id:
                                number = round_execution.reserve(
                                    round_id,
                                    read_run(path)["run_id"],
                                    path,
                                    len(gpus),
                                    policy.resources.wait_timeout_minutes,
                                    reason,
                                )
                            launch = time.monotonic()
                            with mutex:
                                active[str(path)] = (launch, len(gpus))
                            with (path / f"attempt-{attempt}.log").open("a") as log:
                                process = launch_run(path, policy, gpus, attempt, log)
                                outcome = monitor(
                                    process,
                                    path,
                                    gpus,
                                    launch,
                                    policy.budget.run_timeout_minutes,
                                    lambda: (
                                        reason()
                                        or (
                                            round_execution.stop_reason(round_id)
                                            if round_id
                                            else None
                                        )
                                    ),
                                )
                            if outcome == "failed":
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
                            if (
                                outcome == "succeeded"
                                and read_run(path)["status"] != "succeeded"
                            ):
                                outcome = "failed"
                    except Exception as exc:
                        outcome = reason() or (
                            "timed_out" if isinstance(exc, TimeoutError) else "failed"
                        )
                        state = read_run(path)
                        state["error"] = f"{type(exc).__name__}: {exc}"
                        write_json(path / "manifest.json", state)
                    finally:
                        if process and process.poll() is None:
                            _terminate(process)
                        with mutex:
                            active.pop(str(path), None)
                            if launch:
                                hours += (time.monotonic() - launch) * len(gpus) / 3600
                        if round_id and number is not None:
                            round_execution.settle(
                                round_id, number, read_run(path)["run_id"], outcome
                            )
                        state = read_run(path)
                        state["status"] = outcome
                        state["attempts"][-1].update(
                            status=outcome,
                            finished_at=time.time(),
                            stage=state["stage"],
                        )
                        if state.get("error"):
                            state["attempts"][-1]["error"] = state["error"]
                        write_json(
                            path / "attempts" / f"{attempt}.json", state["attempts"][-1]
                        )
                        write_json(path / "manifest.json", state)
                    if outcome != "failed":
                        break
                return {"directory": str(path), "status": outcome}

        results = []
        try:
            with ThreadPoolExecutor(
                max_workers=policy.budget.max_parallel_jobs
            ) as executor:
                futures = [executor.submit(worker, path) for path in pending]
                try:
                    for future in as_completed(futures):
                        results.append(future.result())
                except BaseException:
                    stop.set()
                    raise
        finally:
            write_json(
                usage_path,
                {
                    "elapsed_sec": prior_elapsed + time.monotonic() - started,
                    "gpu_hours": hours,
                },
            )
            if sweep:
                saved.update(
                    elapsed_sec=prior_elapsed + time.monotonic() - started,
                    gpu_hours=hours,
                )
                write_json(directory / "sweep.json", saved)
        return {
            "schema_version": 1,
            "directory": str(directory),
            "ok": all(r["status"] == "succeeded" for r in results),
            "skipped": len(run_dirs) - len(pending),
            "runs": results,
        }


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
