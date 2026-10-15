"""Progressively disclosed environment and interface discovery commands."""

import argparse
import json
import sys

from tsflab.cli.output import publish


def _parser(kind):
    from tsflab.cli.inventory import group_usage

    parser = argparse.ArgumentParser(
        prog="tsf env" if kind == "env" else "tsf agent interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=group_usage("env")
        if kind == "env"
        else "Execution recipes (show), the execution-policy schema (schema), and the Python API modules (api).",
        epilog="Before TSFLab is installed: python3 scripts/setup_env.py." if kind == "env" else None,
    )
    parser.add_argument("action", nargs="?", default="audit" if kind == "env" else "show")
    parser.add_argument(
        "--module",
        choices=["budget", "resources", "recovery", "storage", "tracking"],
        help="Show only this policy section with schema",
    )
    parser.add_argument("--config", help="Optional run config to audit")
    parser.add_argument("--policy", help="Optional execution policy")
    parser.add_argument("--json", action="store_true")
    return parser


def _print_api(as_json):
    from tsflab.experiments.infra.api import describe_modules

    payload = publish(describe_modules())
    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        for module in payload["modules"]:
            print(f"{module['name']}: {', '.join(module['exports'])}\n  {module['requirements']}")
    return 0


def _interface_schema(module):
    from tsflab.experiments.infra.policy import ExecutionPolicy

    if module:
        return ExecutionPolicy.model_fields[module].annotation.model_json_schema()
    return ExecutionPolicy.model_json_schema()


def _interface_recipes():
    from tsflab.cli.inventory import GROUPS, records

    return {
        "schema_version": 1,
        "basic": {
            "preview": "tsf run <toml> --dry-run [--json]",
            "run": "tsf run <toml> [--json]",
            "results": "tsf result --help",
        },
        "execution": {
            "preflight": "tsf run <toml> --dry-run --json",
            "prepare": "tsf run <toml> --prepare-only --policy <toml>",
            "queue": "tsf run --backend queue --help",
            "cluster": "tsf run --backend slurm --help",
            "storage": "tsf env storage --help",
            "usage": "tsf env usage --help",
            "json_envelope": "tsf --format json <command> [args...]",
            "environment": "tsf env audit [--config <toml>] [--policy <toml>] --json",
            "python_api": "tsf agent interface api --json",
            "module_schema": "tsf agent interface schema --module <section> --json",
            "policy_schema": "tsf agent interface schema --json",
            "policy": "tsf run <toml> --policy <toml>",
            "status": "tsf run status <directory> --json",
            "cancel": "tsf run cancel <directory> --json",
            "resume": "tsf run resume <directory> --json",
        },
        "research": {
            "rounds": "tsf research --help",
            "tasks": "tsf agent task --help",
            "iteration": "tsf research iteration <round-id>",
        },
        "assets": {group.name: f"tsf {group.name} --help" for group in GROUPS},
        "commands": records(),
    }


def _print_interface(payload, parsed):
    if parsed.json or parsed.action == "schema":
        print(json.dumps(payload, indent=2))
        return
    for group in ("basic", "execution", "research", "assets"):
        print(group.capitalize() + ":")
        for name, command in payload[group].items():
            print(f"  {name}: {command}")


def _interface(parsed):
    from tsflab.cli.inventory import retired_note

    if parsed.action == "modules":  # retired name of `api`
        print(retired_note("agent interface modules", forwarding=True), file=sys.stderr)
        parsed.action = "api"
    if parsed.action == "api":
        return _print_api(parsed.json)
    if parsed.module and parsed.action != "schema":
        raise ValueError("--module is only valid for tsf agent interface schema")
    if parsed.action not in {"show", "schema"}:
        raise ValueError("usage: tsf agent interface [show|schema|api] [--module SECTION] [--json]")
    payload = _interface_schema(parsed.module) if parsed.action == "schema" else _interface_recipes()
    publish(payload)
    _print_interface(payload, parsed)
    return 0 if payload.get("ok", True) else 1


def _env_setup(as_json):
    from tsflab.cli.runtime import ROOT
    from tsflab.experiments.infra.machine import format_setup, setup

    payload = setup(str(ROOT))
    print(json.dumps(payload, indent=2) if as_json else format_setup(payload))
    return 0 if payload["ok"] else 1


def _env_doctor(as_json):
    from tsflab.experiments.infra.machine import doctor, format_doctor

    payload = publish(doctor())
    print(json.dumps(payload, indent=2) if as_json else format_doctor(payload))
    return 0


def _audit_payload(parsed):
    from tsflab.experiments.infra.policy import load_policy

    policy = load_policy(parsed.policy)
    if parsed.config:
        from tsflab.experiments.config.loader import load_config
        from tsflab.experiments.infra.execution import preflight

        return preflight([item.config for item in load_config(parsed.config)], policy)
    from tsflab.experiments.infra.environment import audit_environment

    return audit_environment(policy)


def _print_environment(payload):
    print("Environment ready" if payload["ok"] else "Environment needs attention")
    print(f"  Python {payload['checks'][0]['detail']}; {len(payload['packages'])} runtime packages found")
    for check in payload["checks"]:
        if check["status"] == "failed" or check["name"] in {
            "torch",
            "cuda",
            "mps",
            "disk",
        }:
            print(f"  {check['status']}: {check['name']} — {check['detail']}")
    print("  Use --json for all checks and GPU inventory")


def _print_preflight(payload):
    print(f"{payload['total_runs']} resolved runs: {'ready' if payload['ok'] else 'not ready'}")
    for run in payload["runs"]:
        for error in run["errors"]:
            print(f"  {run['model']}: {error}")


def _env(parsed):
    if parsed.module:
        raise ValueError("--module is only valid for tsf agent interface schema")
    if parsed.action == "setup":
        return _env_setup(parsed.json)
    if parsed.action == "doctor":
        return _env_doctor(parsed.json)
    if parsed.action != "audit":
        raise ValueError("usage: tsf env [audit|doctor|setup] [--config <toml>] [--policy <toml>] [--json]")
    payload = _audit_payload(parsed)
    publish(payload)
    if parsed.json:
        print(json.dumps(payload, indent=2))
    elif "checks" in payload:
        _print_environment(payload)
    else:
        _print_preflight(payload)
    return 0 if payload.get("ok", True) else 1


def infrastructure_command(args):
    parsed = _parser(args[0]).parse_args(args[1:])
    try:
        return _interface(parsed) if args[0] == "interface" else _env(parsed)
    except (ValueError, OSError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "ok": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            ),
            file=sys.stdout if parsed.json else sys.stderr,
        )
        return 2
