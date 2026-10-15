"""Optional operations, discovered through tsf agent interface."""

import argparse
import json

from tsflab.cli.output import publish


def _parser(kind):
    prog = {"queue": "tsf run --backend queue", "slurm": "tsf run --backend slurm"}.get(kind, f"tsf env {kind}")
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("action")
    parser.add_argument("directory")
    parser.add_argument("--policy")
    parser.add_argument("--json", action="store_true")
    if kind == "queue":
        parser.add_argument("--run")
        parser.add_argument("--executor", help="Importable module:callable for this job")
        parser.add_argument("--priority", type=int, default=0)
        parser.add_argument("--slots", type=int, default=1)
        parser.add_argument("--once", action="store_true")
    elif kind == "usage":
        parser.add_argument("--operation")
        parser.add_argument("--tokens", type=int, default=0)
        parser.add_argument("--cost-usd", type=float, default=0)
    elif kind == "storage":
        parser.add_argument("--apply", action="store_true")
    else:
        parser.add_argument("--partition")
        parser.add_argument("--account")
        parser.add_argument("--gpus", type=int, default=0)
        parser.add_argument("--minutes", type=int, default=60)
    return parser


def _queue(options):
    from tsflab.experiments.infra.queue import cancel_job, enqueue, jobs, work

    if options.action == "add":
        if not options.run:
            raise ValueError("queue add requires --run <prepared directory>")
        return enqueue(options.directory, options.run, priority=options.priority, executor=options.executor)
    if options.action == "cancel":
        return cancel_job(options.directory)
    if options.action == "status":
        return jobs(options.directory)
    if options.action == "work":
        return work(options.directory, once=options.once, slots=options.slots)
    raise ValueError("queue action must be add, status, work, or cancel")


def _usage(options, policy):
    from tsflab.experiments.infra.accounting import account

    return account(
        options.directory,
        options.action,
        options.operation,
        tokens=options.tokens,
        cost_usd=options.cost_usd,
        budget=policy.budget,
    )


def _storage(options, policy):
    from tsflab.experiments.infra.retention import cleanup, storage_status

    if options.action == "status":
        return storage_status(options.directory, policy)
    if options.action == "cleanup":
        return cleanup(options.directory, policy, apply=options.apply)
    raise ValueError("storage action must be status or cleanup")


def _slurm(options):
    from tsflab.experiments.infra.slurm import slurm

    return slurm(
        options.directory,
        options.action,
        partition=options.partition,
        account=options.account,
        gpus=options.gpus,
        minutes=options.minutes,
    )


def _run(kind, options):
    from tsflab.experiments.infra.policy import load_policy

    policy = load_policy(options.policy)
    if kind == "queue":
        return _queue(options)
    if kind == "usage":
        return _usage(options, policy)
    if kind == "storage":
        return _storage(options, policy)
    return _slurm(options)


def operations_command(args):
    options = _parser(args[0]).parse_args(args[1:])
    try:
        result = _run(args[0], options)
        publish(result)
        print(json.dumps(result, indent=2))
        return 0 if not isinstance(result, dict) or result.get("ok", True) else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
        )
        return 2
