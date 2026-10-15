"""tsf model verify — run admission checks and record them in each model card.

    tsf model verify <Name...> [--jobs N] [--json]
    tsf model verify --all [--jobs N] [--json]
    tsf model verify --changed [--base REF] [--jobs N] [--json]

Each model's strict executable contract runs once, in its own interpreter; the result is written to the
``[admission]`` table of its ``card.toml`` (see ``tsflab.catalog.admission``).
"""

from __future__ import annotations

import argparse
import json
import sys

from tsflab.core.paths import repository_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tsf model verify", description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="every catalog model")
    mode.add_argument(
        "--changed", action="store_true", help="models whose package, preset, or used component changed since --base"
    )
    parser.add_argument("--base", default="origin/dev", help="git ref for --changed (default origin/dev)")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    return parser


def _requested(args: argparse.Namespace, root) -> list[str]:
    from tsflab.catalog.admission import changed_models
    from tsflab.catalog.registry.models import MODEL_CATALOG

    if args.all:
        return list(MODEL_CATALOG.names())
    if args.changed:
        return changed_models(root, args.base)
    return list(args.names)


def _no_models(args: argparse.Namespace) -> int:
    print(
        "no models to verify" if args.changed else "give model names, --all, or --changed",
        file=sys.stderr if not args.changed else sys.stdout,
    )
    return 0 if args.changed else 2


def _admissible(parser: argparse.ArgumentParser, args: argparse.Namespace, root, names: list[str]) -> list[str]:
    from tsflab.catalog.admission import card_path, unfinished_files
    from tsflab.catalog.registry.models import MODEL_CATALOG

    unknown = sorted(set(names) - set(MODEL_CATALOG.names()))
    if unknown:
        parser.error(f"unknown model(s): {', '.join(unknown)}")
    # A scaffold is discoverable as soon as its spec exists; it must not record a passed admission.
    unfinished = sorted(name for name in names if unfinished_files(card_path(root, name).parent))
    if unfinished and not (args.all or args.changed):
        parser.error(f"unfinished scaffold (replace SCAFFOLD/PLACEHOLDER/TODO first): {', '.join(unfinished)}")
    if unfinished:
        print(f"skipping unfinished scaffold(s): {', '.join(unfinished)}", file=sys.stderr)
        names = [name for name in names if name not in unfinished]
    return names


def _print_summary(records: dict, failed: list[str]) -> None:
    for name in failed:
        print(f"FAILED {name}  {records[name]['note']}")
    print(f"{len(records) - len(failed)}/{len(records)} passed; results written to each card's [admission]")


def verification_command(argv: list[str]) -> int:
    from tsflab.catalog.admission import run_contracts, write_admission

    parser = _parser()
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    root = repository_root()
    names = _requested(args, root)
    if not names:
        return _no_models(args)
    names = _admissible(parser, args, root, names)

    def progress(name: str, failure: dict[str, str] | None) -> None:
        if not args.json:  # stream, so a long run shows where it is
            print(f"{'FAILED' if failure else 'PASSED':6} {name}", flush=True)

    results = run_contracts(names, args.jobs, isolated=True, progress=progress)
    records = {name: write_admission(root, name, failure) for name, failure in results.items()}
    failed = [name for name, record in records.items() if record["status"] != "passed"]
    if args.json:
        print(json.dumps(records, indent=2))
    else:
        _print_summary(records, failed)
    return 1 if failed else 0
