"""CLI for small, explicit research round workspaces."""

from __future__ import annotations

import argparse
import json
import sys

from tsflab.research.rounds import (
    EVENT_KINDS,
    STATUSES,
    add_event,
    create_round,
    claim_iteration,
    list_rounds,
    load_round,
    read_events,
    set_status,
)


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def research_command(rest: list[str]) -> int:
    """Create, inspect, and update a lightweight research round."""
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

    for action, sub in subparsers.choices.items():
        if action != "list":  # these always print JSON; --json is accepted for uniformity
            sub.add_argument("--json", action="store_true", help="structured output (the default here)")
    args = parser.parse_args(rest)
    try:
        if args.action == "start":
            from tsflab.experiments.infra.policy import load_policy
            budget = load_policy(args.policy).budget.model_dump(exclude_none=True) if args.policy else {}
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
        elif args.action == "iteration":
            _print(claim_iteration(args.round_id, operation=args.operation))
        elif args.action == "list":
            records = list_rounds()
            if args.json:
                _print(records)
            else:
                for record in records:
                    limit = record["max_runs"] if record["max_runs"] is not None else "unlimited"
                    print(
                        f"{record['id']}  {record['status']:<9} "
                        f"runs={record['runs_used']}/{limit}  {record['task']}"
                    )
        elif args.action == "show":
            _print({"round": load_round(args.round_id), "events": read_events(args.round_id)})
        elif args.action == "note":
            _print(add_event(args.round_id, args.kind, args.text))
        elif args.action == "status":
            _print(set_status(args.round_id, args.status, args.message))
        return 0
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
