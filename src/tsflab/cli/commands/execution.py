"""Concurrent smoke-test and experiment execution commands."""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tsflab.cli.output import publish
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


def _smoke_parser() -> argparse.ArgumentParser:
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
    return parser


def _model_smoke_configs(model: str) -> list[str]:
    """Smoke configs of one model, excluding those of other models whose module shares its prefix."""
    prefix = module_for_model(model)
    try:
        from tsflab.catalog.registry.models import MODEL_CATALOG

        other_modules = {
            path.split(".")[2] for name, path in MODEL_CATALOG.refs().items() if module_for_model(name) != prefix
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
    return configs


def _smoke_configs(parser: argparse.ArgumentParser, args: argparse.Namespace) -> list[str]:
    if args.all:
        return sorted(str(path.relative_to(ROOT)) for path in RUN_CONFIG_DIR.glob("smoke_*.toml"))
    if args.model:
        return _model_smoke_configs(args.model)
    if args.config:
        return args.config
    parser.error("one of --all / --model / --config is required")


def _run_smokes(configs: list[str], jobs: int) -> list[tuple[str, int, str, float]]:
    """Run the smoke configs on ``jobs`` workers, printing one line per finished config."""
    results: list[tuple[str, int, str, float]] = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
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
    return results


def smoke_command(rest: list[str]) -> int:
    """Run selected end-to-end smoke configurations concurrently."""
    parser = _smoke_parser()
    args = parser.parse_args(rest)
    configs = _smoke_configs(parser, args)

    missing = [config for config in configs if not (ROOT / config).exists()]
    if missing:
        print("Missing smoke config(s):", file=sys.stderr)
        for config in missing:
            print(f"  {config}", file=sys.stderr)
        return 1

    print(f"Running {len(configs)} smoke config(s) with {args.jobs} worker(s)...\n")
    results = _run_smokes(configs, args.jobs)

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


def _print_runs(result: dict) -> None:
    """Text lines of a sweep result: one per run, with its errors and scored metrics."""
    import json

    for run in result["runs"]:
        print(
            f"{run.get('status', 'ready' if run.get('ok') else 'failed')}: {run.get('directory', run.get('model', ''))}"
        )
        for error in run.get("errors", []):
            print(f"  {error}")
        if run.get("metrics"):
            print(
                "  "
                + json.dumps({k: run[k] for k in ("split", "metrics", "train_time_sec", "test_time_sec") if k in run})
            )
        if run.get("error"):
            print(f"  {run['error']}")
    if result.get("skipped"):
        print(f"Skipped {result['skipped']} completed runs")


def _print_wait(result: dict, timeout: float) -> None:
    """Text lines of `tsf run wait`: counts, one line per finished run, then unfinished runs."""
    state = "done" if result["done"] else f"still running after {timeout:.0f}s"
    print(f"{state}: " + ", ".join(f"{n} {k}" for k, n in sorted(result["counts"].items())))
    for row in sorted(result.get("results", []), key=lambda r: (str(r["model"]), r["pred_len"] or 0, r["seed"] or 0)):
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    for run in result["unfinished"]:
        print(f"{run['status']}: {run['directory']}" + (f" ({run['error']})" if run.get("error") else ""))


def _print_result(result: dict, args) -> None:
    """Print the result of one `tsf run` action (JSON with --json; one line with --compact)."""
    import json

    if args.json:
        print(json.dumps(result, separators=(",", ":")) if args.compact else json.dumps(result, indent=2))
    elif "counts" in result:
        _print_wait(result, args.timeout)
    elif "runs" in result:
        _print_runs(result)
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


def _split_defaults(loaded: list, args) -> None:
    """``--eval-split`` (or ``TSFLAB_EVAL_SPLIT``) sets the split of every new run; otherwise round defaults apply."""
    if not args.eval_split:
        _round_defaults(loaded, args.round_id)
        return
    for item in loaded:
        item.config.evaluation.split = args.eval_split
        item.raw.setdefault("evaluation", {})["split"] = args.eval_split


def _run_parser() -> argparse.ArgumentParser:
    from tsflab.cli.inventory import group_usage

    parser = argparse.ArgumentParser(
        prog="tsf run", description=group_usage("run"), formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "configs",
        nargs="*",
        help="config paths; or status|wait|cancel|resume DIR; or evaluate RUN_DIR... --eval-split S",
    )
    parser.add_argument("--model", help="run a named model preset without authoring a TOML file")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="setting for --model, e.g. task.seq_len=96; strings may be unquoted",
    )
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        metavar="KEY=LIST",
        help="sweep axis for --model, e.g. task.pred_len=[96,192]",
    )
    parser.add_argument("--jobs", default=None, help="worker count, or auto (CUDA matrix only)")
    parser.add_argument("--gpus", default=None, help="GPU indices/UUIDs, or auto (visible GPUs; waits for free leases)")
    parser.add_argument("--round", dest="round_id", default=None)
    parser.add_argument(
        "--eval-split",
        choices=("val", "test"),
        default=os.environ.get("TSFLAB_EVAL_SPLIT"),
        help="evaluation split of every new run (default: the config; val inside a research round)",
    )
    parser.add_argument("--policy", help="Optional execution policy TOML")
    parser.add_argument(
        "--prepare-only", action="store_true", help="Persist a validated matrix for queue or Slurm submission"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the resolved matrix (coverage summary) and check it without executing",
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable result")
    parser.add_argument(
        "--compact", action="store_true", help="compact JSON: preflight keeps failures and coverage only"
    )
    parser.add_argument(
        "--output", help="save the returned JSON; the path keys one matrix request, so a relaunch reuses its sweep"
    )
    parser.add_argument(
        "--timeout", type=float, default=1800.0, help="wait: seconds before returning while runs continue"
    )
    return parser


def _apply_overrides(policy, args) -> None:
    """``--jobs N|auto`` and ``--gpus LIST|auto`` override the execution policy."""
    if args.jobs is not None:
        policy.budget.auto_parallel = args.jobs == "auto"
        jobs = 1 if policy.budget.auto_parallel else int(args.jobs)
        if jobs < 1:
            raise ValueError("--jobs must be positive")
        policy.budget.max_parallel_jobs = jobs
    if args.gpus is not None:
        policy.resources.auto_select = args.gpus == "auto"
        policy.resources.gpus = (
            [] if policy.resources.auto_select else [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
        )


def _check_saved_split(directory: str, eval_split: str) -> None:
    """A resumed sweep keeps its saved evaluation split."""
    from tsflab.experiments.infra.execution import status

    saved = status(directory)
    if any(
        cell.get("config", {}).get("evaluation", {}).get("split") != eval_split for cell in saved.get("runs", [saved])
    ):
        raise ValueError(
            "saved evaluation split differs from --eval-split or TSFLAB_EVAL_SPLIT; "
            "create a new sweep instead of changing a saved run"
        )


def _manage(policy, args) -> dict:
    """`tsf run status|wait|cancel|resume <directory>`."""
    from tsflab.experiments.infra.execution import cancel, execute, status, wait

    if len(args.configs) != 2:
        raise ValueError("usage: tsf run {status,cancel,resume,wait} <directory> [--json]")
    action, directory = args.configs
    if action == "wait":
        return wait(directory, args.timeout)
    if action == "status":
        return status(directory)
    if action == "cancel":
        return cancel(directory)
    if args.eval_split:
        _check_saved_split(directory, args.eval_split)
    overridden = args.policy or args.jobs is not None or args.gpus is not None
    return execute(directory, policy if overridden else None, round_id=args.round_id)


def _load_matrix(args) -> list:
    """The runs of a request: config files, a ``--model`` request, or frozen ``evaluate`` runs."""
    from tsflab.experiments.config.loader import load_config, load_model_request

    if args.model:
        return load_model_request(args.model, args.set, args.sweep)
    if args.configs and args.configs[0] == "evaluate":
        if len(args.configs) < 2 or args.eval_split not in {"val", "test"}:
            raise ValueError("usage: tsf run evaluate RUN_DIRECTORY... --eval-split val|test")
        from tsflab.experiments.infra.frozen import frozen_evaluation

        return [frozen_evaluation(name, args.eval_split) for name in args.configs[1:]]
    loaded = []
    for name in args.configs or ["configs/runs/run_single_data.toml"]:
        path = Path(name).expanduser()
        if not path.exists():
            path = ROOT / path
        loaded.extend(load_config(str(path.resolve())))
    return loaded


def _prepare_and_execute(loaded: list, policy, args) -> dict:
    """Persist a validated matrix, then execute it (unless ``--prepare-only``)."""
    from tsflab.experiments.infra.execution import execute, prepare_sweep, wait

    request_key = str(Path(args.output).expanduser().resolve()) if args.output else None
    directory = prepare_sweep(loaded, policy, args.round_id, request_key=request_key)
    if not args.json:
        print(f"Experiment records: {directory}", flush=True)
    if args.prepare_only:
        return {"ok": True, "directory": str(directory), "prepared": True}
    try:
        return execute(directory, policy, round_id=args.round_id)
    except BlockingIOError:  # an identical keyed launch already owns this sweep
        return wait(directory, args.timeout)


def _launch(policy, args) -> dict:
    """Validate the requested matrix, then prepare and (unless previewing) execute it."""
    from tsflab.experiments.infra.execution import preflight

    loaded = _load_matrix(args)
    if not loaded:
        raise ValueError("experiment matrix is empty")
    _split_defaults(loaded, args)
    if args.dry_run and not args.json:
        from tsflab.cli.commands.inspect_config import summarize

        summarize(loaded)
    result = preflight([item.config for item in loaded], policy)
    if args.round_id and load_round(args.round_id)["status"] != "running":
        raise ValueError("research round is not running")
    if not result["ok"] or args.dry_run:
        return result
    return _prepare_and_execute(loaded, policy, args)


def _compact(result: dict) -> dict:
    """Preflight without passed package checks and the policy dump."""
    if "environments" not in result:
        return result
    kept = {key: result[key] for key in ("schema_version", "ok", "total_runs", "budget_ok", "runs") if key in result}
    kept["environments"] = [
        {"ok": env.get("ok"), "checks": [c for c in env.get("checks", []) if c.get("status") != "passed"]}
        for env in result["environments"]
    ]
    return kept


def _save_failure(args, failure: dict, exc: Exception) -> None:
    from tsflab.experiments.infra.execution import RequestConflictError

    if not args.output or isinstance(exc, RequestConflictError):
        return
    try:
        if not _save_output(Path(args.output), failure):
            failure["output_preserved"] = str(Path(args.output).expanduser().resolve())
    except OSError:
        pass  # keep the original error if storing the output itself failed


def _check_request(args) -> None:
    if args.model and args.configs:
        raise ValueError("use --model or positional configs, not both")
    if (args.set or args.sweep) and not args.model:
        raise ValueError("--set and --sweep require --model")


def _request_policy(args, load_policy):
    """The ``--policy`` file's policy, or a resumed sweep's saved policy when none is given."""
    policy = load_policy(args.policy)
    if not args.policy and len(args.configs) == 2 and args.configs[0] == "resume":
        from tsflab.experiments.infra.execution import saved_policy

        policy = saved_policy(args.configs[1])
    return policy


def _request_result(policy, args) -> dict:
    """Manage or launch the request; compact and save its result as requested."""
    managing = args.configs and args.configs[0] in {"status", "cancel", "resume", "wait"}
    result = _manage(policy, args) if managing else _launch(policy, args)
    if args.compact:
        result = _compact(result)
    if args.output and not _save_output(Path(args.output), result):
        result["output_preserved"] = str(Path(args.output).expanduser().resolve())
    return result


def run_command(rest: list[str]) -> int:
    """Run, preview, smoke-test, cancel, or resume experiments through one public surface."""
    import json

    if "--smoke" in rest:
        return smoke_command([item for item in rest if item != "--smoke"])
    rest, backend = _extract_backend(rest)
    if backend in {"queue", "slurm"}:
        from tsflab.cli.commands.operations import operations_command

        return operations_command([backend, *rest])
    from tsflab.experiments.infra.policy import load_policy

    args = _run_parser().parse_args(rest)
    try:
        _check_request(args)
        policy = _request_policy(args, load_policy)
        _apply_overrides(policy, args)
        result = _request_result(policy, args)
        publish(result)
        _print_result(result, args)
        return 0 if result.get("ok", True) else 1
    except Exception as exc:
        failure = {"schema_version": 1, "ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}}
        _save_failure(args, failure, exc)
        if args.json:
            print(json.dumps(failure))
        else:
            print(str(exc), file=sys.stderr)
        return 2
