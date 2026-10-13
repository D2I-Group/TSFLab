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



def _save_output(path: Path, result: dict) -> bool:
    """Keep a native execution receipt when a later request is rejected or previewed."""
    import json
    from tsflab.experiments.infra.storage import file_lock, write_json

    path = path.expanduser().resolve()
    with file_lock(path.with_name(path.name + ".output.lock")):
        try:
            previous = json.loads(path.read_text())
        except (OSError, ValueError):
            previous = {}
        if isinstance(previous, dict) and previous.get("directory") and not result.get("directory"):
            return False
        write_json(path, result)
    return True


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


def run_command(rest: list[str]) -> int:
    """Run, preview, smoke-test, cancel, or resume experiments through one public surface."""
    import json
    if "--smoke" in rest:
        return smoke_command([item for item in rest if item != "--smoke"])
    rest, backend = _extract_backend(rest)
    if backend in {"queue", "slurm"}:
        from tsflab.cli.commands.operations import operations_command

        return operations_command([backend, *rest])
    from tsflab.experiments.infra.execution import RequestConflictError, cancel, execute, preflight, prepare_sweep, status, wait
    from tsflab.experiments.infra.policy import load_policy
    parser = argparse.ArgumentParser(prog="tsf run", description="Run experiments and return saved metrics. Optional: --policy execution.toml. Manage existing runs with status/cancel/resume/wait <directory>; wait blocks until completion or --timeout. --smoke (--all | --model NAME | --config PATH...) runs smoke configs; --backend queue|slurm ACTION DIR submits prepared sweeps (see tsf run --backend queue --help).", epilog="Also: tsf run --smoke (--all | --model NAME | --config PATH...); tsf run --backend queue|slurm ACTION DIR.")
    parser.add_argument("configs", nargs="*", help="Config paths, or evaluate RUN_DIRECTORY... --eval-split val|test for frozen checkpoints")
    parser.add_argument("--model", help="Run a named model preset without authoring a TOML file")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON", help="Scientific setting for --model, e.g. task.seq_len=96; strings may be unquoted")
    parser.add_argument("--sweep", action="append", default=[], metavar="KEY=LIST", help="Sweep axis for --model, e.g. task.pred_len=[96,192]")
    parser.add_argument("--jobs", default=None, help="Worker count or auto (CUDA matrix only)")
    parser.add_argument("--gpus", default=None, help="GPU indices/UUIDs or auto (visible GPUs; waits for free exclusive leases)")
    parser.add_argument("--round", dest="round_id", default=None)
    parser.add_argument("--eval-split", choices=("val", "test"), default=os.environ.get("TSFLAB_EVAL_SPLIT"),
                        help="Explicit evaluation split for all new runs (research harness defaults to val)")
    parser.add_argument("--policy", help="Optional execution policy TOML")
    parser.add_argument("--prepare-only", action="store_true", help="Persist a validated matrix for queue or Slurm submission")
    parser.add_argument("--dry-run", action="store_true", help="Preview the resolved matrix (coverage summary) and check it without executing")
    parser.add_argument("--json", action="store_true", help="Machine-readable result")
    parser.add_argument("--compact", action="store_true",
                        help="Compact preflight JSON: retain failures and matrix coverage, omit successful package checks and policy dump")
    parser.add_argument("--output", help="Atomically save returned JSON; this path identifies one matrix request, so repeated launches reuse its sweep")
    parser.add_argument("--timeout", type=float, default=1800, help="Maximum seconds for run wait; leaves training running")
    args = parser.parse_args(rest)
    try:
        if args.model and args.configs:
            raise ValueError("use --model or positional configs, not both")
        if (args.set or args.sweep) and not args.model:
            raise ValueError("--set and --sweep require --model")
        policy = load_policy(args.policy)
        if not args.policy and len(args.configs) == 2 and args.configs[0] == "resume":
            from tsflab.experiments.infra.execution import saved_policy
            policy = saved_policy(args.configs[1])
        if args.jobs is not None:
            policy.budget.auto_parallel = args.jobs == "auto"
            jobs = 1 if policy.budget.auto_parallel else int(args.jobs)
            if jobs < 1:
                raise ValueError("--jobs must be positive")
            policy.budget.max_parallel_jobs = jobs
        if args.gpus is not None:
            policy.resources.auto_select = args.gpus == "auto"
            policy.resources.gpus = [] if policy.resources.auto_select else [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
        if args.configs and args.configs[0] in {"status", "cancel", "resume", "wait"}:
            if len(args.configs) != 2:
                raise ValueError("usage: tsf run {status,cancel,resume,wait} <directory> [--json]")
            action, directory = args.configs
            if action == "status":
                result = status(directory)
            elif action == "wait":
                result = wait(directory, timeout=args.timeout)
            elif action == "cancel":
                result = cancel(directory)
            else:
                if args.eval_split:
                    saved = status(directory)
                    cells = saved.get("runs", [saved])
                    if any(cell.get("config", {}).get("evaluation", {}).get("split") != args.eval_split for cell in cells):
                        raise ValueError("saved evaluation split differs from --eval-split or TSFLAB_EVAL_SPLIT; create a new sweep instead of changing a saved run")
                result = execute(directory, policy if args.policy or args.jobs is not None or args.gpus is not None else None, round_id=args.round_id)
        else:
            from tsflab.experiments.config.loader import load_config
            configs = [] if args.model or (args.configs and args.configs[0] == "evaluate") else args.configs or ["configs/runs/run_single_data.toml"]
            loaded = []
            for name in configs:
                path = Path(name).expanduser()
                if not path.exists():
                    path = ROOT / path
                loaded.extend(load_config(str(path.resolve())))
            if args.model:
                from tsflab.experiments.config.loader import load_model_request
                loaded = load_model_request(args.model, args.set, args.sweep)
            if args.configs and args.configs[0] == "evaluate":
                if len(args.configs) < 2 or args.eval_split not in {"val", "test"}:
                    raise ValueError("usage: tsf run evaluate RUN_DIRECTORY... --eval-split val|test")
                from tsflab.experiments.infra.frozen import frozen_evaluation
                loaded = [frozen_evaluation(name, args.eval_split) for name in args.configs[1:]]
            if not loaded:
                raise ValueError("experiment matrix is empty")
            if args.eval_split:
                for item in loaded:
                    item.config.evaluation.split = args.eval_split
                    item.raw.setdefault("evaluation", {})["split"] = args.eval_split
            if args.dry_run and not args.json:
                from tsflab.cli.commands.inspect_config import summarize

                summarize(loaded)
            result = preflight([item.config for item in loaded], policy)
            if args.round_id:
                state = load_round(args.round_id)
                if state["status"] != "running":
                    raise ValueError("research round is not running")
            if result["ok"] and not args.dry_run:
                directory = prepare_sweep(loaded, policy, args.round_id,
                                          request_key=str(Path(args.output).expanduser().resolve()) if args.output else None)
                if not args.json:
                    print(f"Experiment records: {directory}", flush=True)
                if args.prepare_only:
                    result = {"ok": True, "directory": str(directory), "prepared": True}
                else:
                    try:
                        result = execute(directory, policy, round_id=args.round_id)
                    except BlockingIOError:
                        # An identical keyed launch already owns this sweep.
                        result = wait(directory, timeout=args.timeout)
        if args.compact and "environments" in result:
            result = {key: result[key] for key in ("schema_version", "ok", "total_runs", "budget_ok", "runs") if key in result} | {
                "environments": [{"ok": env.get("ok"), "checks": [check for check in env.get("checks", [])
                                  if check.get("status") != "passed"]}
                                 for env in result["environments"]]
            }
        if args.output and not _save_output(Path(args.output), result):
            result["output_preserved"] = str(Path(args.output).expanduser().resolve())
        publish(result)
        if args.json:
            print(json.dumps(result, separators=(",", ":")) if args.compact else json.dumps(result, indent=2))
        elif "runs" in result:
            for run in result["runs"]:
                print(f"{run.get('status', 'ready' if run.get('ok') else 'failed')}: {run.get('directory', run.get('model', ''))}")
                for error in run.get("errors", []):
                    print(f"  {error}")
                if run.get("metrics"):
                    print("  " + json.dumps({k: run[k] for k in ("split", "metrics", "train_time_sec", "test_time_sec") if k in run}))
                if run.get("error"):
                    print(f"  {run['error']}")
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
        return 0 if result.get("ok", True) else 1
    except Exception as exc:
        failure = {"schema_version": 1, "ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}}
        if args.output and not isinstance(exc, RequestConflictError):
            try:
                if not _save_output(Path(args.output), failure):
                    failure["output_preserved"] = str(Path(args.output).expanduser().resolve())
            except OSError:
                pass  # Preserve the original error if output storage itself failed.
        if args.json:
            print(json.dumps(failure))
        else:
            print(str(exc), file=sys.stderr)
        return 2
