"""tsf repo check — the single definition of "mergeable".

    tsf repo check [--scope full|changed|release] [--base REF] [--only STEP...] [--fail-fast] [--json]
    tsf repo check --audit
    tsf repo check --contracts construct|forward|backward|strict [--models NAME...]

``full`` runs every check. ``changed`` runs the same checks plus, outside CI, the admission of the
changed models (``tsf model verify --changed``) and the targeted
smoke run for the models affected by the diff against ``--base`` (default:
``origin/$GITHUB_BASE_REF`` in CI, else the first of origin/dev, dev,
origin/main, main that exists). The full ``tsf run --smoke --all`` sweep stays a
nightly/manual job. ``release`` adds what a final release requires: a model audit in which every
model has a passed admission (``[admission]`` in its card). Admission runs once
per model, when it is added or changed (``tsf model verify --changed``). Fast steps run first; every step runs unless
``--fail-fast``; the exit code is 0 only when all steps pass.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from tsflab.core.paths import repository_root

PY = sys.executable
CLI = [PY, "-m", "tsflab.cli.main"]

# Diff paths whose change warrants a targeted smoke run (smoke-affected.yml).
SMOKE_TRIGGERS = (
    "src/tsflab/models/",
    "src/tsflab/experiments/",
    "src/tsflab/catalog/",
    "src/tsflab/data/",
    "src/tsflab/core/",
    "tests/",
    "configs/runs/smoke_",
)
# Changes that touch every model at once map to the representative set.
SMOKE_BROAD = (
    "src/tsflab/experiments/",
    "src/tsflab/catalog/registry/",
    "src/tsflab/models/_components/",
    "src/tsflab/data/",
    "src/tsflab/core/",
    "tests/",
)
SMOKE_FALLBACK = (
    "configs/runs/smoke_crib.toml",
    "configs/runs/smoke_quantile_dlinear.toml",
    "configs/runs/smoke_gaussian_mlp.toml",
)


@dataclass(frozen=True)
class Step:
    name: str
    argv: list[str]
    cwd: str = "."
    scopes: tuple[str, ...] = ("full", "changed", "release")
    setup: list[list[str]] = field(default_factory=list)


def _targeted_smoke_configs(changed: list[str], root: Path) -> tuple[set[str], bool]:
    """Smoke configs of the changed models and changed smoke configs; whether any model changed."""
    configs: set[str] = set()
    model_changed = False
    for path in changed:
        parts = path.split("/")
        if path.startswith("src/tsflab/models/") and len(parts) > 3:
            model_changed = True
            for found in sorted((root / "configs/runs").glob(f"smoke_{parts[3]}*.toml")):
                configs.add(found.relative_to(root).as_posix())
        if path.startswith("configs/runs/smoke_") and path.endswith(".toml") and (root / path).is_file():
            configs.add(path)
    return configs, model_changed


def affected_smoke_configs(changed: list[str], root: Path) -> list[str]:
    """Map changed files to smoke configs; empty when nothing needs smoking."""
    if not any(path.startswith(SMOKE_TRIGGERS) for path in changed):
        return []
    if any(path.startswith(SMOKE_BROAD) for path in changed):
        return list(SMOKE_FALLBACK)
    configs, model_changed = _targeted_smoke_configs(changed, root)
    if not configs and model_changed:
        return list(SMOKE_FALLBACK)
    return sorted(configs)


def changed_models(changed: list[str]) -> list[str]:
    """Model packages whose files appear in ``changed`` (shared code is covered by the smoke run)."""
    names = {path.split("/")[3] for path in changed if path.startswith("src/tsflab/models/") and path.count("/") >= 4}
    return sorted(name for name in names if not name.startswith(("_", ".")))


def select_steps(
    scope: str, changed: list[str], root: Path, *, base: str | None = None, local: bool = False
) -> list[Step]:
    """Return the ordered steps for ``scope`` (pure; runs nothing).

    ``repo-audit`` covers agent assets, component, dataset, and model cards, complexity,
    the catalog, and documentation in one process. With ``local`` (an author's checkout,
    not CI), ``changed`` also re-admits the changed models, which records ``[admission]``.
    """
    steps = [
        Step("schema-export", [PY, "-m", "tsflab.core", "--check", "--out", "src/tsflab/core/schema"]),
        Step("agent-tasks", CLI + ["agent", "task", "validate"]),
        Step("model-cards", CLI + ["model", "audit", "--summary"], scopes=("full", "changed")),
        Step("model-release", CLI + ["model", "audit", "--summary", "--release"], scopes=("release",)),
        Step("web-submissions", [PY, "pipeline/validate.py"], cwd="apps/web"),
        Step("repo-audit", CLI + ["repo", "check", "--audit"]),
    ]
    steps = [step for step in steps if scope in step.scopes]
    if scope == "changed":
        if local and changed_models(changed):
            steps.append(
                Step("admission-changed", CLI + ["model", "verify", "--changed", *(["--base", base] if base else [])])
            )
        configs = affected_smoke_configs(changed, root)
        if configs:
            steps.append(
                Step(
                    "smoke-affected",
                    CLI + ["run", "--smoke", "--config", *configs],
                    setup=[[PY, "scripts/make_smoke_data.py"]],
                )
            )
    steps.append(Step("pytest", [PY, "-m", "pytest", "-q", "-m", "not e2e", "tests"]))
    return steps


def changed_files(root: Path, base: str | None) -> list[str]:
    """Files changed on this branch relative to ``base`` (merge-base diff)."""
    candidates = [base] if base else []
    if not base:
        env = os.environ.get("GITHUB_BASE_REF")
        if env:
            candidates.append(f"origin/{env}")
        candidates += ["origin/dev", "dev", "origin/main", "main"]
    for ref in candidates:
        probe = subprocess.run(["git", "rev-parse", "--verify", "--quiet", ref], cwd=root, capture_output=True)
        if probe.returncode == 0:
            diff = subprocess.run(
                ["git", "diff", "--name-only", f"{ref}...HEAD"], cwd=root, capture_output=True, text=True
            )
            if diff.returncode == 0:
                tracked = diff.stdout.split()
                dirty = subprocess.run(
                    ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, capture_output=True, text=True
                ).stdout
                return sorted(set(tracked) | {line[3:] for line in dirty.splitlines()})
    raise RuntimeError("cannot resolve a base ref for --scope changed; pass --base REF")


def _run(step: Step, root: Path, quiet: bool) -> dict:
    start = time.monotonic()
    code = 0
    output = ""
    for command in [*step.setup, step.argv]:
        proc = subprocess.run(
            command,
            cwd=root / step.cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        output += proc.stdout
        code = proc.returncode
        if code:
            break
    if not quiet and code:
        print(output, file=sys.stderr)
    return {
        "step": step.name,
        "ok": code == 0,
        "code": code,
        "seconds": round(time.monotonic() - start, 1),
        "tail": "\n".join(output.strip().splitlines()[-15:]),
    }


def _check_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tsf repo check", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scope", choices=["full", "changed", "release"], default="full")
    parser.add_argument("--base", help="base ref for --scope changed")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--list", action="store_true", help="print the steps and exit")
    parser.add_argument("--only", nargs="+", metavar="STEP", help="run only these steps (see --list)")
    parser.add_argument(
        "--audit", action="store_true", help="run the static repository audits in-process (the repo-audit step)"
    )
    parser.add_argument(
        "--contracts",
        choices=["construct", "forward", "backward", "strict"],
        help="execute model tensor contracts at this depth",
    )
    parser.add_argument(
        "--models", nargs="+", metavar="NAME", help="with --contracts: only these models (default: the whole catalog)"
    )
    return parser


def _run_in_process(args: argparse.Namespace) -> int:
    from tsflab.cli.commands.repository import run_audit, run_contracts

    codes = []
    if args.audit:
        codes.append(run_audit())
    if args.contracts:
        codes.append(run_contracts(args.contracts, args.models))
    return 1 if any(codes) else 0


def _changed_for_scope(root: Path, args: argparse.Namespace) -> list[str] | None:
    """The changed files for ``--scope changed`` (else none); None, after printing why, when unresolvable."""
    try:
        return changed_files(root, args.base) if args.scope == "changed" else []
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return None


def _refresh_indexes(local: bool, list_only: bool) -> None:
    if local and not list_only:
        # Derived indexes (.agents/README.md, the model docs index) follow the cards; refresh
        # them first so a check never fails on a stale projection. CI only verifies them.
        from tsflab.cli.commands.repository import regenerate_cards

        regenerate_cards([])


def _only_steps(steps: list[Step], only: list[str]) -> list[Step] | None:
    """The steps named in ``only``; None, after printing them, when some are unknown."""
    unknown = sorted(set(only) - {step.name for step in steps})
    if unknown:
        print(f"unknown step(s): {', '.join(unknown)}; see `tsf repo check --list`", file=sys.stderr)
        return None
    return [step for step in steps if step.name in only]


def _run_steps(steps: list[Step], root: Path, args: argparse.Namespace) -> list[dict]:
    results = []
    for step in steps:
        if not args.json:
            print(f"[check] {step.name} ...", flush=True)
        result = _run(step, root, quiet=args.json)
        results.append(result)
        if not args.json:
            print(f"[check] {step.name}: {'PASS' if result['ok'] else 'FAIL'} ({result['seconds']}s)", flush=True)
        if args.fail_fast and not result["ok"]:
            break
    return results


def _report(args: argparse.Namespace, results: list[dict]) -> int:
    failed = [r["step"] for r in results if not r["ok"]]
    if args.json:
        print(json.dumps({"scope": args.scope, "passed": not failed, "failed": failed, "steps": results}, indent=2))
    else:
        print("\n== tsf repo check summary ==")
        for r in results:
            print(f"{'PASS' if r['ok'] else 'FAIL'}  {r['step']}  ({r['seconds']}s)")
        print("MERGEABLE" if not failed else f"NOT MERGEABLE: {', '.join(failed)}")
    return 1 if failed else 0


def check_command(argv: list[str]) -> int:
    parser = _check_parser()
    args = parser.parse_args(argv)
    if args.audit or args.contracts:
        return _run_in_process(args)
    if args.models:
        parser.error("--models requires --contracts")
    root = repository_root()
    changed = _changed_for_scope(root, args)
    if changed is None:
        return 2
    local = not os.environ.get("CI")
    _refresh_indexes(local, args.list)
    steps = select_steps(args.scope, changed, root, base=args.base, local=local)
    if args.only:
        steps = _only_steps(steps, args.only)
        if steps is None:
            return 2
    if args.list:
        for step in steps:
            print(f"{step.name}: {' '.join(step.argv[1:])}")
        return 0
    return _report(args, _run_steps(steps, root, args))
