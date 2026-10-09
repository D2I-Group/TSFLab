"""Progressively disclosed environment and interface discovery commands."""

import argparse

from tsflab.cli.output import publish
import json
import sys


def infrastructure_command(args):
    parser = argparse.ArgumentParser(
        prog="tsf env" if args[0] == "env" else "tsf agent interface",
        epilog="tsf env doctor [--json] detects this machine and prints the matching install and run "
        "profiles; tsf env setup installs that profile (before TSFLab is installed: python3 scripts/setup_env.py). tsf env storage|usage ... manage capacity and spending (see tsf env storage --help)."
        if args[0] == "env"
        else None,
    )
    parser.add_argument(
        "action", nargs="?", default="audit" if args[0] == "env" else "show"
    )
    parser.add_argument(
        "--module",
        choices=["budget", "resources", "recovery", "storage", "tracking"],
        help="Show only this policy section with schema",
    )
    parser.add_argument("--config", help="Optional run config to audit")
    parser.add_argument("--policy", help="Optional execution policy")
    parser.add_argument("--json", action="store_true")
    parsed = parser.parse_args(args[1:])
    try:
        from tsflab.experiments.infra.policy import ExecutionPolicy, load_policy

        if args[0] == "interface" and parsed.action == "modules":
            from tsflab.experiments.infra.api import describe_modules

            payload = publish(describe_modules())
            if parsed.json:
                print(json.dumps(payload, indent=2))
            else:
                for module in payload["modules"]:
                    print(
                        f"{module['name']}: {', '.join(module['exports'])}\n  {module['requirements']}"
                    )
            return 0
        if parsed.module and (args[0] != "interface" or parsed.action != "schema"):
            raise ValueError("--module is only valid for tsf agent interface schema")
        if args[0] == "env" and parsed.action == "setup":
            from tsflab.cli.runtime import ROOT
            from tsflab.experiments.infra.machine import format_setup, setup

            payload = setup(str(ROOT))
            print(json.dumps(payload, indent=2) if parsed.json else format_setup(payload))
            return 0 if payload["ok"] else 1
        if args[0] == "env" and parsed.action == "doctor":
            from tsflab.experiments.infra.machine import doctor, format_doctor

            payload = publish(doctor())
            print(json.dumps(payload, indent=2) if parsed.json else format_doctor(payload))
            return 0
        if args[0] == "interface":
            if parsed.action not in {"show", "schema"}:
                raise ValueError("usage: tsf agent interface [show|schema|modules] [--json]")
            payload = (
                (
                    ExecutionPolicy.model_fields[
                        parsed.module
                    ].annotation.model_json_schema()
                    if parsed.module
                    else ExecutionPolicy.model_json_schema()
                )
                if parsed.action == "schema"
                else {
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
                        "python_modules": "tsf agent interface modules --json",
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
                    "assets": {
                        name: f"tsf {name} --help"
                        for name in (
                            "catalog",
                            "data",
                            "model",
                            "result",
                            "repo",
                            "realtime",
                        )
                    },
                }
            )
        else:
            if parsed.action != "audit":
                raise ValueError(
                    "usage: tsf env [audit|doctor|setup] [--config <toml>] [--policy <toml>] [--json]"
                )
            policy = load_policy(parsed.policy)
            if parsed.config:
                from tsflab.experiments.config.loader import load_config
                from tsflab.experiments.infra.execution import preflight

                payload = preflight(
                    [item.config for item in load_config(parsed.config)], policy
                )
            else:
                from tsflab.experiments.infra.environment import audit_environment

                payload = audit_environment(policy)
        publish(payload)
        if parsed.json or parsed.action == "schema":
            print(json.dumps(payload, indent=2))
        elif args[0] == "interface":
            for group in ("basic", "execution", "research", "assets"):
                print(group.capitalize() + ":")
                for name, command in payload[group].items():
                    print(f"  {name}: {command}")
        elif "checks" in payload:
            print(
                "Environment ready" if payload["ok"] else "Environment needs attention"
            )
            print(
                f"  Python {payload['checks'][0]['detail']}; {len(payload['packages'])} runtime packages found"
            )
            for check in payload["checks"]:
                if check["status"] == "failed" or check["name"] in {
                    "torch",
                    "cuda",
                    "mps",
                    "disk",
                }:
                    print(f"  {check['status']}: {check['name']} — {check['detail']}")
            print("  Use --json for all checks and GPU inventory")
        else:
            print(
                f"{payload['total_runs']} resolved runs: {'ready' if payload['ok'] else 'not ready'}"
            )
            for run in payload["runs"]:
                for error in run["errors"]:
                    print(f"  {run['model']}: {error}")
        return 0 if payload.get("ok", True) else 1
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
