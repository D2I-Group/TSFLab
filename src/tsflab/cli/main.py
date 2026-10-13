#!/usr/bin/env python3
"""Public Agent CLI for TSFLab: eleven commands, one per module.

The CLI is intentionally a thin router. Command behavior lives in focused
modules so the public surface stays stable while model, data, execution, and
repository concerns evolve independently.

Usage:
    tsf <command> [args...]

Read path:
    catalog          overview, search, list, show <name> for models, components, datasets

Modules:
    data             add, prepare (--from traffic|ultratraffic|gift-eval|tfb|dcrnn), analyze (profile,
                     TFB characteristics, --plot), splits (official split borders), download, publish, audit
    model            scaffold, add, artifacts, verify, compose, audit models
    run              run experiments; --smoke, --dry-run, --backend local|queue|slurm
    env              audit the environment; storage and usage subcommands
    result           aggregate, rank, plot, report, predictions, board, submit, leaderboard, hub
    realtime         rolling real-time tracks: update, open, forecast, score
    research         research rounds: start, list, show, note, status, iteration, reference, screen

Repository and Agents:
    repo             check (mergeable gate, audit/contract steps), cards, schema
    agent            task list/show/render/start/validate, interface, modules, sync
    init             scaffold a standalone project for chosen modules

Progressive disclosure: ``catalog search`` returns L0 lines; ``catalog show <name>
--depth {0,1,2,3}`` opens L0 line, L1 interface/constraints, L2 full card, or L3
paths to open.

Run ``tsf <command> --help`` for command-specific options.
"""

from __future__ import annotations

import sys

from tsflab.cli.inventory import overview

__doc__ = f"{__doc__}\n{overview()}"

def _lazy(module: str, name: str):
    def call(rest: list[str]) -> int:
        from importlib import import_module

        return getattr(import_module(module), name)(rest)

    return call


def _env(rest: list[str]) -> int:
    if rest and rest[0] in {"storage", "usage"}:
        from tsflab.cli.commands.operations import operations_command

        return operations_command(rest)
    from tsflab.cli.commands.infrastructure import infrastructure_command

    return infrastructure_command(["env", *rest])


def _result(rest: list[str]) -> int:
    from tsflab.cli.commands.data_results import result_command

    return result_command(rest)


COMMANDS = {
    "catalog": _lazy("tsflab.cli.commands.catalog_resources", "catalog_command"),
    "data": _lazy("tsflab.cli.commands.data_results", "data_command"),
    "model": _lazy("tsflab.cli.commands.catalog_resources", "model_command"),
    "run": _lazy("tsflab.cli.commands.execution", "run_command"),
    "env": _env,
    "result": _result,
    "realtime": _lazy("tsflab.realtime.cli", "main"),
    "research": _lazy("tsflab.cli.commands.research", "research_command"),
    "repo": _lazy("tsflab.cli.commands.repository", "repository_command"),
    "agent": _lazy("tsflab.cli.commands.agent_tasks", "agent_command"),
    "init": _lazy("tsflab.agent.scaffold", "main"),
}


def _without_format_json(argv: list[str]) -> list[str] | None:
    """``argv`` without the global ``--format json`` (any position), or None when it is absent."""
    for index, arg in enumerate(argv):
        if arg == "--format" and argv[index + 1:index + 2] == ["json"]:
            return argv[:index] + argv[index + 2:]
        if arg == "--format=json":
            return argv[:index] + argv[index + 1:]
    return None


def main(argv: list[str] | None = None) -> int:
    """Dispatch one public command without importing unrelated heavy modules."""
    argv = sys.argv[1:] if argv is None else argv
    stripped = _without_format_json(argv)
    if stripped is not None:
        from tsflab.cli.commands.envelope import envelope
        return envelope(stripped)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(overview(), end="")
        return 0
    handler = COMMANDS.get(argv[0])
    if handler is None:
        print(f"unknown command: {argv[0]!r}\n", file=sys.stderr)
        print(overview(), end="", file=sys.stderr)
        return 2
    return handler(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
