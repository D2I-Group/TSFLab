"""Concurrent smoke-test and experiment execution commands."""

from __future__ import annotations

from tsflab.cli.output import publish

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tsflab.cli.runtime import ROOT, RUN_CONFIG_DIR, module_for_model, run_config
from tsflab.research.rounds import load_round


def smoke_command(rest: list[str]) -> int:
    """Run selected end-to-end smoke configurations concurrently."""
    parser = argparse.ArgumentParser(
        prog="tsf run --smoke",
        description="Run smoke config(s) concurrently and report PASS/FAIL.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Run every smoke config")
    group.add_argument("--model", help="Run smoke configs belonging to one model")
    group.add_argument("--config", nargs="+", help="Explicit smoke config path(s)")
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(8, os.cpu_count() or 2),
        help="Concurrent workers (default: min(8, cpu))",
    )
    args = parser.parse_args(rest)

    if args.all:
        configs = sorted(str(path.relative_to(ROOT)) for path in RUN_CONFIG_DIR.glob("smoke_*.toml"))
    elif args.model:
        prefix = module_for_model(args.model)
        try:
            from tsflab.catalog.registry.models import MODEL_CATALOG

            other_modules = {
                path.split(".")[2]
                for name, path in MODEL_CATALOG.refs().items()
                if module_for_model(name) != prefix
            }
        except Exception:
            other_modules = set()
        configs = sorted(
            str(path.relative_to(ROOT))
            for path in RUN_CONFIG_DIR.glob(f"smoke_{prefix}*.toml")
            if path.stem[len("smoke_") :] not in other_modules
        )
        if not configs:
            configs = [f"configs/runs/smoke_{prefix}.toml"]
    elif args.config:
        configs = args.config
    else:
        parser.error("one of --all / --model / --config is required")

    missing = [config for config in configs if not (ROOT / config).exists()]
    if missing:
        print("Missing smoke config(s):", file=sys.stderr)
        for config in missing:
            print(f"  {config}", file=sys.stderr)
        return 1

    print(f"Running {len(configs)} smoke config(s) with {args.jobs} worker(s)...\n")
    results: list[tuple[str, int, str, float]] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {}
        starts = {}
        for config in configs:
            starts[config] = time.monotonic()
            futures[executor.submit(run_config, config)] = config
        for future in as_completed(futures):
            config, code, tail = future.result()
            duration = time.monotonic() - starts[config]
            status = "PASS" if code == 0 else "FAIL"
            extra = "" if code == 0 else f"  (exit {code}) {tail[:80]}"
            print(f"  {status}  {Path(config).stem:<28} {duration:5.1f}s{extra}")
            results.append((config, code, tail, duration))

    passed = sum(1 for _, code, _, _ in results if code == 0)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


def _extract_backend(args: list[str]) -> tuple[list[str], str]:
    """Remove ``--backend local|queue|slurm`` from ``args`` (default local)."""
    rest, backend, skip = [], "local", False
    for index, arg in enumerate(args):
        if skip:
            skip = False
        elif arg == "--backend" and index + 1 < len(args):
            backend, skip = args[index + 1], True
        elif arg.startswith("--backend="):
            backend = arg.split("=", 1)[1]
        else:
            rest.append(arg)
    if backend not in {"local", "queue", "slurm"}:
        raise SystemExit(f"tsf run: --backend must be local, queue, or slurm, not {backend!r}")
    return rest, backend


def _print_result(result: dict, args) -> None:
    """Print the result of one `tsf run` action (JSON with --json)."""
    import json

    if args.json:
        print(json.dumps(result, indent=2))
    elif "counts" in result:
        state = "done" if result["done"] else f"still running after {args.timeout:.0f}s"
        print(f"{state}: " + ", ".join(f"{n} {k}" for k, n in sorted(result["counts"].items())))
        for row in sorted(result.get("results", []), key=lambda r: (str(r["model"]), r["pred_len"] or 0, r["seed"] or 0)):
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        for run in result["unfinished"]:
            print(f"{run['status']}: {run['directory']}" + (f" ({run['error']})" if run.get("error") else ""))
    elif "runs" in result:
        for run in result["runs"]:
            print(f"{run.get('status', 'ready' if run.get('ok') else 'failed')}: {run.get('directory', run.get('model', ''))}")
            for error in run.get("errors", []):
                print(f"  {error}")
        if result.get("skipped"):
            print(f"Skipped {result['skipped']} completed runs")
    elif "run_id" in result:
        print(f"{result['run_id']}: {result['status']} ({result['stage']})")
        if result.get("error"):
            print(result["error"])
        print(f"Records: {result['directory']}")
    elif result.get("cancel_requested"):
        print(f"Cancellation requested: {result['directory']}")
    else:
        print(json.dumps(result, indent=2))


def _round_defaults(loaded: list, round_id: str | None) -> None:
    """Runs attached to a research round score the validation split unless the config sets the split:
    a search selects on validation, and its final evaluation states `evaluation.split = "test"`."""
    if not round_id:
        return
    for item in loaded:
        if "split" not in (item.raw.get("evaluation") or {}):
            item.config.evaluation.split = "val"
    if any(item.config.evaluation.split == "val" for item in loaded):
        print('Research round: scoring the validation split (set evaluation.split = "test" for a final run)')


def run_command(rest: list[str]) -> int:
    """Run, preview, smoke-test, cancel, or resume experiments through one public surface."""
    import json
    if "--smoke" in rest:
        return smoke_command([item for item in rest if item != "--smoke"])
    rest, backend = _extract_backend(rest)
    if backend in {"queue", "slurm"}:
        from tsflab.cli.commands.operations import operations_command

        return operations_command([backend, *rest])
    from tsflab.experiments.infra.execution import cancel, execute, preflight, prepare_sweep, status, wait
    from tsflab.experiments.infra.policy import load_policy
    parser = argparse.ArgumentParser(prog="tsf run", description="Run experiments. Optional: --policy execution.toml. Manage existing runs with status/cancel/resume <directory>; wait <directory> [--timeout S] blocks until a run or sweep started in the background ends and prints one summary. --smoke (--all | --model NAME | --config PATH...) runs smoke configs; --backend queue|slurm ACTION DIR submits prepared sweeps (see tsf run --backend queue --help).", epilog="Also: tsf run --smoke (--all | --model NAME | --config PATH...); tsf run --backend queue|slurm ACTION DIR.")
    parser.add_argument("configs", nargs="*")
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--round", dest="round_id", default=None)
    parser.add_argument("--policy", help="Optional execution policy TOML")
    parser.add_argument("--prepare-only", action="store_true", help="Persist a validated matrix for queue or Slurm submission")
    parser.add_argument("--dry-run", action="store_true", help="Preview the resolved matrix (coverage summary) and check it without executing")
    parser.add_argument("--json", action="store_true", help="Machine-readable result")
    parser.add_argument("--timeout", type=float, default=1800.0, help="wait: seconds before returning while runs continue")
    args = parser.parse_args(rest)
    try:
        policy = load_policy(args.policy)
        if not args.policy and len(args.configs) == 2 and args.configs[0] == "resume":
            from tsflab.experiments.infra.execution import saved_policy
            policy = saved_policy(args.configs[1])
        if args.jobs is not None:
            if args.jobs < 1:
                raise ValueError("--jobs must be positive")
            policy.budget.max_parallel_jobs = args.jobs
        if args.gpus is not None:
            policy.resources.gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
        if args.configs and args.configs[0] in {"status", "cancel", "resume", "wait"}:
            if len(args.configs) != 2:
                raise ValueError("usage: tsf run {status,cancel,resume,wait} <directory> [--json]")
            action, directory = args.configs
            if action == "wait":
                result = wait(directory, args.timeout)
            elif action == "status":
                result = status(directory)
            elif action == "cancel":
                result = cancel(directory)
            else:
                result = execute(directory, policy if args.policy or args.jobs is not None or args.gpus is not None else None, round_id=args.round_id)
        else:
            from tsflab.experiments.config.loader import load_config
            configs = args.configs or ["configs/runs/run_single_data.toml"]
            loaded = []
            for name in configs:
                path = Path(name).expanduser()
                if not path.exists():
                    path = ROOT / path
                loaded.extend(load_config(str(path.resolve())))
            if not loaded:
                raise ValueError("experiment matrix is empty")
            _round_defaults(loaded, args.round_id)
            if args.dry_run and not args.json:
                from tsflab.cli.commands.inspect_config import summarize

                summarize(loaded)
            result = preflight([item.config for item in loaded], policy)
            if args.round_id:
                state = load_round(args.round_id)
                if state["status"] != "running":
                    raise ValueError("research round is not running")
                remaining = None if state["max_runs"] is None else state["max_runs"] - state["runs_used"]
                if remaining is not None and len(loaded) > remaining:
                    raise ValueError(f"matrix needs {len(loaded)} runs; round has {remaining} remaining")
            if result["ok"] and not args.dry_run:
                directory = prepare_sweep(loaded, policy, args.round_id)
                if not args.json:
                    print(f"Experiment records: {directory}", flush=True)
                result = {"ok": True, "directory": str(directory), "prepared": True} if args.prepare_only else execute(directory, policy, round_id=args.round_id)
        publish(result)
        _print_result(result, args)
        return 0 if result.get("ok", True) else 1
    except Exception as exc:
        if args.json:
            print(json.dumps({"schema_version": 1, "ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}}))
        else:
            print(str(exc), file=sys.stderr)
        return 2
