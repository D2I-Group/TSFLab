"""Public CLI for inspecting and rendering harness-neutral Agent tasks."""

from __future__ import annotations

import argparse
import json
import sys

from tsflab.agent.tasks import (
    AgentTaskError,
    audit_tasks,
    list_tasks,
    load_task,
    render_task,
    render_text,
)
from tsflab.research.rounds import ResearchRoundError


def _json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _render_parser(action: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"tsf agent task {action}")
    parser.add_argument("name")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--json", action="store_true")
    return parser


def _supplied(items: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in items:
        if "=" not in item or not item.split("=", 1)[0]:
            raise AgentTaskError("--set requires KEY=VALUE")
        key, value = item.split("=", 1)
        values[key] = value
    return values


AGENT_USAGE = (
    "usage: tsf agent task {list,show,render,start,validate} [args...]\n"
    "       tsf agent interface [show|schema|modules] [--module SECTION] [--json]\n"
    "       tsf agent modules [--json]     modules, skills, and tasks of this project\n"
    "       tsf agent sync                 refresh project skills/tasks from the installed TSFLab\n"
    "start prepares a research round and prompt; it does not dispatch an Agent"
)


def _modules_command(tail: list[str]) -> int:
    from tsflab.agent.modules import ALL_MODULES, MODULES, find_project

    found = find_project()
    chosen = found[1] if found else list(ALL_MODULES)
    records = [
        {"module": name, "purpose": MODULES[name]["purpose"], "skills": MODULES[name]["skills"],
         "tasks": MODULES[name]["tasks"], "commands": MODULES[name]["commands"],
         "context": MODULES[name]["context"]}
        for name in chosen
    ]
    if tail == ["--json"]:
        _json({"project": str(found[0]) if found else None, "modules": records})
        return 0
    print(f"Modules ({'project ' + str(found[0]) if found else 'all; not inside a tsf init project'}):")
    for record in records:
        print(f"  {record['module']}: {record['purpose']}")
        print(f"    skills: {', '.join(record['skills'])}")
        if record["tasks"]:
            print(f"    tasks: {', '.join(record['tasks'])}")
    return 0


def agent_command(args: list[str]) -> int:
    """Route task inspection, rendering, interface discovery, and project asset sync."""
    if not args or args[0] in {"-h", "--help", "help"}:
        print(AGENT_USAGE)
        return 0
    if args[0] == "interface":
        from tsflab.cli.commands.infrastructure import infrastructure_command

        return infrastructure_command(args)
    if args[0] == "modules":
        return _modules_command(args[1:])
    if args[0] == "sync":
        from tsflab.agent.scaffold import sync_command

        return sync_command(args[1:])
    if args[0] != "task":
        print(AGENT_USAGE, file=sys.stderr)
        return 2
    rest = args[1:]
    if not rest or rest[0] in {"-h", "--help", "help"}:
        print(AGENT_USAGE)
        return 0
    action, tail = rest[0], rest[1:]
    try:
        if action == "list":
            if tail not in ([], ["--json"]):
                raise AgentTaskError("usage: tsf agent task list [--json]")
            records = list_tasks()
            if tail == ["--json"]:
                _json(records)
            else:
                for record in records:
                    print(f"{record['name']}: {record['summary']}")
            return 0
        if action == "show":
            tail = [arg for arg in tail if arg != "--json"]  # always JSON
            if len(tail) != 1:
                raise AgentTaskError("usage: tsf agent task show <name> [--json]")
            _json(load_task(tail[0]))
            return 0
        if action == "validate":
            if tail:
                for name in tail:
                    load_task(name)
                print(f"Agent tasks OK: {len(tail)} selected")
                return 0
            errors = audit_tasks()
            if errors:
                for error in errors:
                    print(f"ERROR: {error}")
                return 1
            print(f"Agent tasks OK: {len(list_tasks())} templates")
            return 0
        if action in {"render", "start"}:
            parsed = _render_parser(action).parse_args(tail)
            payload = render_task(parsed.name, _supplied(parsed.set))
            if action == "start":
                from tsflab.experiments.infra.research import prepare_task

                payload = prepare_task(parsed.name, _supplied(parsed.set), persist=True)
                round_state = payload["round"]
                prompt_path = payload["prompt_path"]
                if not parsed.json:
                    print(
                        f"Prepared research round: {round_state['id']}\n"
                        f"Prompt: {prompt_path}\n\n",
                        end="",
                    )
                    print(render_text(payload["task"]), end="")
                    return 0
            _json(payload) if parsed.json else print(render_text(payload), end="")
            return 0
    except (AgentTaskError, ResearchRoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"unknown Agent task action: {action}", file=sys.stderr)
    return 2
