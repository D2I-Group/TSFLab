"""CLI for small, explicit research round workspaces."""

from __future__ import annotations

import argparse
import json
import sys

from tsflab.research.rounds import (
    EVENT_KINDS,
    STATUSES,
    add_event,
    claim_iteration,
    create_round,
    list_rounds,
    load_round,
    read_events,
    set_status,
)


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _parser() -> argparse.ArgumentParser:
    from tsflab.cli.inventory import purpose

    parser = argparse.ArgumentParser(prog="tsf research")
    subparsers = parser.add_subparsers(dest="action", required=True)

    start = subparsers.add_parser("start", help=purpose("research", "start"))
    start.add_argument("--task", default="research", help="Short workflow name")
    start.add_argument("--goal", required=True, help="Falsifiable goal or question")
    start.add_argument("--max-runs", type=int, default=None)
    start.add_argument("--policy", help="Optional execution policy supplying round-wide budgets")
    start.add_argument("--max-iterations", type=int, default=None)
    start.add_argument("--id", dest="round_id", default=None)

    listing = subparsers.add_parser("list", help=purpose("research", "list"))
    listing.add_argument("--json", action="store_true")

    show = subparsers.add_parser("show", help=purpose("research", "show"))
    show.add_argument("round_id")

    note = subparsers.add_parser("note", help=purpose("research", "note"))
    note.add_argument("round_id")
    note.add_argument("--kind", choices=sorted(EVENT_KINDS), default="note")
    note.add_argument("--text", required=True)

    status = subparsers.add_parser("status", help=purpose("research", "status"))
    status.add_argument("round_id")
    status.add_argument("status", choices=sorted(STATUSES))
    status.add_argument("--message")

    iteration = subparsers.add_parser("iteration", help=purpose("research", "iteration"))
    iteration.add_argument("round_id")
    iteration.add_argument("--operation", help="Stable ID for idempotent Agent-defined iteration claims")

    reference = subparsers.add_parser("reference", help=purpose("research", "reference"))
    reference.add_argument("round_id")
    reference.add_argument("directories", nargs="+", metavar="RUN_DIR")

    screen = subparsers.add_parser("screen", help=purpose("research", "screen"))
    screen.add_argument("round_id")
    screen.add_argument("--baseline", nargs="+", required=True, metavar="RUN_DIR")
    screen.add_argument("--candidate", nargs="+", required=True, metavar="RUN_DIR")
    screen.add_argument("--metric", choices=("mse", "mae"), default="mse")
    screen.add_argument(
        "--min-improvement",
        type=float,
        default=0.0,
        help="Required relative reduction, strict at every matched seed/horizon",
    )
    screen.add_argument(
        "--stop-on-fail", action="store_true", help="Close the round when the task's stopping condition applies"
    )

    for action, sub in subparsers.choices.items():
        if action != "list":  # these always print JSON; --json is accepted for uniformity
            sub.add_argument("--json", action="store_true", help="structured output (the default here)")
    return parser


def _start(args: argparse.Namespace) -> None:
    from tsflab.experiments.infra.policy import load_policy

    policy = load_policy(args.policy) if args.policy else None
    budget = (
        policy.budget.model_dump(exclude_none=True, exclude_unset=True, exclude={"auto_parallel"}) if policy else {}
    )
    if policy and policy.budget.auto_parallel:
        budget.pop("max_parallel_jobs", None)
    if args.max_iterations is not None:
        if args.max_iterations < 1:
            raise ValueError("--max-iterations must be positive")
        budget["max_iterations"] = args.max_iterations
    _print(
        create_round(
            task=args.task,
            goal=args.goal,
            max_runs=args.max_runs if args.max_runs is not None else budget.get("max_runs"),
            round_id=args.round_id,
            budget=budget,
        )
    )


def _screen(args: argparse.Namespace) -> None:
    from tsflab.research.screening import screen_candidate

    result = screen_candidate(
        args.round_id,
        args.baseline,
        args.candidate,
        metric=args.metric,
        min_improvement=args.min_improvement,
        stop_on_fail=args.stop_on_fail,
    )
    _print(result)


def _list(args: argparse.Namespace) -> None:
    records = list_rounds()
    if args.json:
        _print(records)
        return
    for record in records:
        limit = record["max_runs"] if record["max_runs"] is not None else "unlimited"
        print(f"{record['id']}  {record['status']:<9} runs={record['runs_used']}/{limit}  {record['task']}")


def _dispatch(args: argparse.Namespace) -> None:
    if args.action == "start":
        _start(args)
    elif args.action == "reference":
        from tsflab.research.screening import register_references

        _print(register_references(args.round_id, args.directories))
    elif args.action == "screen":
        _screen(args)  # A valid no-winner screen is a scientific outcome, not a CLI error.
    elif args.action == "iteration":
        _print(claim_iteration(args.round_id, operation=args.operation))
    elif args.action == "list":
        _list(args)
    elif args.action == "show":
        _print({"round": load_round(args.round_id), "events": read_events(args.round_id)})
    elif args.action == "note":
        _print(add_event(args.round_id, args.kind, args.text))
    elif args.action == "status":
        _print(set_status(args.round_id, args.status, args.message))


def research_command(rest: list[str]) -> int:
    """Create, inspect, and update a lightweight research round."""
    args = _parser().parse_args(rest)
    try:
        _dispatch(args)
        return 0
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
