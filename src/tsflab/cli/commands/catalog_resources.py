"""Public inspection commands for the flat model and component catalogs."""

from __future__ import annotations

import json
import sys
from collections import Counter

from tsflab.cli.runtime import ROOT, passthrough


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _audit_codebase(fields: dict[str, object]) -> tuple[dict, list[str]]:
    """The model's ``[codebase]`` table and the source fields it is missing."""
    codebase_value = fields.get("codebase")
    codebase = dict(codebase_value) if isinstance(codebase_value, dict) else {}
    missing_source = [field for field in ("url", "revision", "license") if not codebase.get(field)]
    return codebase, missing_source


def _verification_status(fields: dict[str, object]) -> dict[str, object]:
    admission = dict(fields.get("admission") or {})
    status = str(admission.get("status") or "pending")
    return {
        "status": status,
        **{key: admission[key] for key in ("date", "commit", "reference", "note") if admission.get(key)},
    }


def _model_audit_record(
    fields: dict[str, object],
) -> dict[str, object]:
    """Return one machine-readable model-card and verification gate result."""
    paper = dict(fields.get("paper", {}))
    codebase, missing_source = _audit_codebase(fields)
    blockers = []
    if paper.get("url") and not paper.get("title"):  # baselines without a paper omit [paper]
        blockers.append("paper.title")
    if codebase:
        blockers.extend(f"codebase.{field}" for field in missing_source)
    verification_status = _verification_status(fields)
    status = verification_status["status"]
    # Admission is recorded once in card.toml; a release requires every model passed.
    release_blockers = list(blockers)
    if status != "passed":
        release_blockers.append(f"admission.{status}")
    return {
        "name": str(fields["name"]),
        "passed": not blockers,
        "blockers": blockers,
        "release_blockers": release_blockers,
        "paper": {
            "title": paper.get("title", ""),
            "venue": paper.get("venue", ""),
            "year": paper.get("year"),
            "url": paper.get("url", ""),
        },
        "codebase": {
            "url": codebase.get("url", ""),
            "revision": codebase.get("revision", ""),
            "license": codebase.get("license", ""),
            "missing": missing_source,
        },
        "smoke_config": fields.get("smoke_config"),
        "components": list(fields.get("components", ())),
        "verification": verification_status,
    }


_TASK_MODES = {
    "time-series": "time_series",
    "spatiotemporal": "spatiotemporal",
    "covariate": "covariate",
}


def _model_list_record(fields: dict[str, object]) -> dict[str, object]:
    from tsflab.catalog.cards.descriptions import read_model_card_description

    model_card = str(fields["model_card"])
    capabilities = set(fields.get("capabilities", ()))
    task_modes = {public for capability, public in _TASK_MODES.items() if capability in capabilities}
    return {
        "name": str(fields["name"]),
        "summary": read_model_card_description(ROOT / model_card).summary,
        "capabilities": sorted(capabilities),
        "task_modes": sorted(task_modes),
    }


def _model_list(rest: list[str]) -> int:
    if any(arg not in {"--details", "--json"} for arg in rest) or len(rest) > 1:
        print("usage: tsf catalog list --kind model [--details | --json]", file=sys.stderr)
        return 2
    from tsflab.catalog.cards.metadata import model_records

    fields_by_name = model_records(ROOT)
    if not rest:
        print("\n".join(sorted(str(fields["name"]) for fields in fields_by_name)))
        return 0
    records = [_model_list_record(fields) for fields in fields_by_name]
    if rest == ["--json"]:
        _print(records)
    else:
        for record in records:
            print(f"{record['name']}\n  {record['summary']}")
    return 0


def _model_lookup(name: str, names: list[str]) -> str:
    """``name``, or its only case-insensitive match among ``names``."""
    if name in names:
        return name
    matches = [candidate for candidate in names if candidate.casefold() == name.casefold()]
    return matches[0] if len(matches) == 1 else name


def _model_show_legacy(spec, fields: dict[str, object], audit: dict[str, object]) -> dict[str, object]:
    paper = dict(fields["paper"])
    return {
        "name": spec.name,
        "module": spec.module,
        "summary": fields["summary"],
        "parameters": spec.params_schema.model_json_schema(),
        "paper": {
            "title": paper["title"],
            "venue": paper["venue"],
            "year": paper["year"],
            "url": paper["url"],
        },
        "codebase": fields["codebase"],
        "config": spec.config_path,
        "model_card": spec.model_card,
        "smoke_config": spec.smoke_config,
        "capabilities": sorted(spec.capabilities),
        "components": list(spec.components),
        "output_type": spec.output_type,
        "task_modes": sorted(spec.task_modes),
        "artifacts": [
            {
                "name": artifact.name,
                "revision": artifact.revision,
                "filename": artifact.filename,
                "required": artifact.required,
            }
            for artifact in spec.artifacts
        ],
        "verification": audit["verification"],
        "blockers": audit["blockers"],
    }


def _model_show_facts(spec, audit: dict[str, object]) -> dict[str, object]:
    return {
        "config": spec.config_path,
        "smoke_config": spec.smoke_config or "(none)",
        "task_modes": sorted(spec.task_modes),
        "capabilities": sorted(spec.capabilities),
        "components": list(spec.components),
        "output_type": spec.output_type,
        "verification": audit["verification"]["status"],
        "blockers": audit["blockers"],
    }


def _model_show_paths(spec, fields: dict[str, object], card_path) -> list[str]:
    from tsflab.catalog.cards.show import existing

    package = card_path.parent.relative_to(ROOT).as_posix()
    return existing(
        ROOT,
        card_path.relative_to(ROOT).as_posix(),
        f"{package}/model.py",
        str(fields["spec_file"]),
        spec.config_path,
        spec.smoke_config or "",
        f"{package}/reference.md",
        *(f"src/tsflab/models/_components/{name}/README.md" for name in spec.components),
    )


def _model_show(rest: list[str]) -> int:
    from tsflab.catalog.cards.metadata import model_records
    from tsflab.catalog.cards.show import parse_show, show_card
    from tsflab.catalog.registry.models import MODEL_CATALOG

    parameters_only = "--params" in rest
    parsed = parse_show(
        "tsf catalog show --kind model", "public model name", [arg for arg in rest if arg != "--params"]
    )
    lookup = _model_lookup(parsed.name, MODEL_CATALOG.names())
    try:
        spec = MODEL_CATALOG.get(lookup)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if parameters_only:
        print(
            json.dumps(
                {
                    "name": spec.name,
                    "config": spec.config_path,
                    "parameters": spec.params_schema.model_json_schema(),
                },
                separators=(",", ":"),
            )
        )
        return 0
    fields = next(record for record in model_records(ROOT) if record["name"] == spec.name)
    card_path = ROOT / str(fields["model_card"])
    fields["card_text"] = card_path.read_text(encoding="utf-8")
    audit = _model_audit_record(fields)
    legacy = _model_show_legacy(spec, fields, audit)
    facts = _model_show_facts(spec, audit)
    paths = _model_show_paths(spec, fields, card_path)
    return show_card(ROOT, card_path, parsed, facts=facts, paths=paths, legacy=legacy)


def _print_artifact_records(spec, records: list[dict], as_json: bool) -> None:
    if as_json:
        _print(records)
    elif not records:
        print(f"{spec.name} declares no external runtime artifacts")
    else:
        for record in records:
            state = "verified" if record["verified"] else "missing-or-invalid"
            print(f"{record['name']}\t{state}\t{record['path']}")


def _model_artifacts(rest: list[str]) -> int:
    import argparse
    from pathlib import Path

    from tsflab.catalog.model_artifacts import artifact_status, fetch_artifact
    from tsflab.catalog.registry.models import MODEL_CATALOG

    parser = argparse.ArgumentParser(
        prog="tsf model artifacts",
        description="Inspect or explicitly fetch checksum-pinned model artifacts.",
    )
    parser.add_argument("name", help="public model name")
    parser.add_argument("--fetch", metavar="ARTIFACT", help="artifact name to download")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--json", action="store_true")
    parsed = parser.parse_args(rest)
    try:
        spec = MODEL_CATALOG.get(parsed.name)
    except KeyError as exc:
        parser.error(str(exc))
    if parsed.fetch:
        selected = next((item for item in spec.artifacts if item.name == parsed.fetch), None)
        if selected is None:
            parser.error(f"model {spec.name!r} has no artifact {parsed.fetch!r}")
        fetch_artifact(spec, selected, parsed.cache_dir)
    records = artifact_status(spec, parsed.cache_dir)
    _print_artifact_records(spec, records, parsed.json)
    return 0


def _model_search(rest: list[str]) -> int:
    from tsflab.catalog.cards.search import search_command

    return search_command(ROOT, rest, prog="tsf catalog search --kind model", kind="model")


def _model_audit_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="tsf model audit",
        description="Audit model cards and their admission records.",
    )
    parser.add_argument("names", nargs="*", help="model names; default: all")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="emit per-model JSON")
    output.add_argument("--summary", action="store_true", help="emit aggregate JSON")
    parser.add_argument("--release", action="store_true", help="also require a passed admission for every model")
    return parser


def _print_audit_summary(records: list[dict], failures: list[dict], key: str, pending: int) -> None:
    blockers = Counter(blocker for record in failures for blocker in record[key])
    _print(
        {
            "models": len(records),
            "passed": len(records) - len(failures),
            "failed": len(failures),
            "blockers": dict(sorted(blockers.items())),
            "verification": dict(sorted(Counter(str(r["verification"]["status"]) for r in records).items())),
            "complete_codebase": sum(bool(r["codebase"]["url"]) and not r["codebase"]["missing"] for r in records),
            "with_smoke_config": sum(bool(r["smoke_config"]) for r in records),
            "admission_pending": pending,
        }
    )


def _print_audit_text(records: list[dict], failures: list[dict], key: str, pending: int, release: bool) -> None:
    for record in failures:
        print(f"FAIL {record['name']}: {', '.join(record[key])}")
    print(f"{len(records) - len(failures)}/{len(records)} model audits passed")
    if pending and not release:
        print(f"{pending} model(s) lack a passed admission (advisory; required by `tsf model audit --release`)")


def _declared_models() -> dict[str, dict[str, object]]:
    from tsflab.catalog.cards.metadata import model_records

    return {str(fields["name"]): fields for fields in model_records(ROOT)}


def _report_unknown(names: list[str], declared: dict[str, dict[str, object]]) -> bool:
    """Print the names that are not declared models; return whether there were any."""
    unknown = [name for name in names if name not in declared]
    if unknown:
        print(f"Unknown model(s): {', '.join(unknown)}", file=sys.stderr)
    return bool(unknown)


def _audit_records(names: list[str], declared: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    for name in names:
        card = ROOT / str(declared[name]["model_card"])
        declared[name]["card_text"] = card.read_text(encoding="utf-8")
    return [_model_audit_record(declared[name]) for name in names]


def _admission_pending(records: list[dict]) -> int:
    return sum(any(b.startswith("admission.") for b in record["release_blockers"]) for record in records)


def _model_audit(rest: list[str]) -> int:
    parsed = _model_audit_parser().parse_args(rest)
    key = "release_blockers" if parsed.release else "blockers"
    declared = _declared_models()
    names = parsed.names or sorted(declared)
    if _report_unknown(names, declared):
        return 2
    records = _audit_records(names, declared)
    failures = [record for record in records if record[key]]
    pending = _admission_pending(records)
    if parsed.summary:
        _print_audit_summary(records, failures, key, pending)
    elif parsed.json:
        _print(records)
    else:
        _print_audit_text(records, failures, key, pending, parsed.release)
    return 1 if failures else 0


_MODEL_ACTIONS = {
    "list": _model_list,
    "show": _model_show,
    "artifacts": _model_artifacts,
    "search": _model_search,
    "audit": _model_audit,
}


def _model_impl(args: list[str]) -> int:
    """Implementation of the model actions (read actions are reached via `tsf catalog`)."""
    if not args or args[0] in {"-h", "--help", "help"}:
        print("usage: tsf catalog {list,show,search} --kind model")
        return 0
    action, rest = args[0], args[1:]
    if action == "scaffold":
        return passthrough("new_model.py", rest)
    if action == "add":
        return passthrough("finalize_model.py", rest)
    handler = _MODEL_ACTIONS.get(action)
    if handler is not None:
        return handler(rest)
    print(f"unknown model action: {action!r}", file=sys.stderr)
    return 2


def _issues_parser():
    import argparse

    from tsflab.catalog.cards.issues import KINDS

    parser = argparse.ArgumentParser(prog="tsf model issues", description=_issues_command.__doc__)
    parser.add_argument("names", nargs="*", help="model names; default: all")
    parser.add_argument("--kind", choices=sorted(KINDS))
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true")
    output.add_argument("--summary", action="store_true", help="counts by kind and coverage")
    return parser


def _issue_rows(records: dict[str, list], kind: str | None) -> list[dict[str, str]]:
    return [
        {"model": name, "kind": issue.kind, "where": issue.where, "what": issue.what, "resolution": issue.resolution}
        for name, issues in records.items()
        for issue in issues or []
        if not kind or issue.kind == kind
    ]


def _card_issues(names: list[str], declared: dict[str, dict[str, object]]) -> dict[str, list]:
    from tsflab.catalog.cards.issues import Issue

    records = {}
    for name in names:
        entries = declared[name].get("issues") or []
        records[name] = [Issue(e["kind"], e["where"], e["what"], e["resolution"]) for e in entries]
    return records


def _print_issue_rows(rows: list[dict[str, str]]) -> None:
    for row in rows:
        print(f"{row['model']}  [{row['kind']}]  {row['where']}: {row['what']}")
    print(f"{len(rows)} issue(s) in {len({row['model'] for row in rows})} model(s)")


def _issues_command(argv: list[str]) -> int:
    """List upstream issues recorded in model cards (paper and official-code problems)."""
    from tsflab.catalog.cards.issues import load_declined, summarize, summarize_declined

    args = _issues_parser().parse_args(argv)
    declared = _declared_models()
    if _report_unknown(args.names, declared):
        return 2
    records = _card_issues(args.names or sorted(declared), declared)
    malformed: list[str] = []
    declined, declined_problems = load_declined(ROOT)
    for problem in declined_problems:
        print(f"MALFORMED {problem}", file=sys.stderr)
    malformed.extend(declined_problems)
    if args.summary:
        _print({**summarize(records), "malformed": len(malformed), "declined": summarize_declined(declined)})
        return 1 if malformed else 0
    rows = _issue_rows(records, args.kind)
    if args.json:
        _print(rows)
        return 1 if malformed else 0
    _print_issue_rows(rows)
    return 1 if malformed else 0


def _component_audit(rest: list[str]) -> int:
    if rest:
        print("component audit takes no arguments", file=sys.stderr)
        return 2
    from tsflab.catalog.cards.resources import audit_resource_cards
    from tsflab.catalog.component_audit import audit_components
    from tsflab.catalog.components import COMPONENT_CATALOG
    from tsflab.core.paths import repository_root

    root = repository_root()
    failures = audit_components()
    failures.extend(error for error in audit_resource_cards(root) if "_components" in error)
    for failure in failures:
        print(f"ERROR: {failure}")
    total = len(COMPONENT_CATALOG.names())
    print(f"Component catalog/cards: {'PASS' if not failures else 'FAIL'} ({total} components)")
    return 1 if failures else 0


def _component_list(rest: list[str]) -> int:
    from tsflab.catalog.cards.components import component_card_path
    from tsflab.catalog.cards.depth import card_l0, l0_line, read_card
    from tsflab.catalog.components import COMPONENT_CATALOG

    records = []
    for spec in COMPONENT_CATALOG.specs():
        card = read_card(component_card_path(ROOT, spec.name))
        record = card_l0(card)
        record.update(
            slot=card.front.get("slot"),
            fits=list(card.front.get("fits", ())),
            module=spec.module,
            card=f"src/tsflab/models/_components/{spec.name}/README.md",
        )
        records.append(record)
    if rest == ["--json"]:
        _print(records)
    else:
        for record in records:
            print(l0_line(record))
    return 0


def _component_consumers(name: str) -> list[str]:
    """Model packages that use component ``name``."""
    from tsflab.catalog.component_audit import components_used_by

    consumers = []
    for package in sorted((ROOT / "src" / "tsflab" / "models").iterdir()):
        if package.is_dir() and not package.name.startswith("_") and name in components_used_by(package):
            consumers.append(package.name)
    return consumers


def _component_show_paths(spec, card: str, consumers: list[str]) -> list[str]:
    from tsflab.catalog.cards.show import existing

    tests = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests").glob("*.py")
        if spec.module in path.read_text(encoding="utf-8")
    )
    return existing(
        ROOT,
        card,
        f"src/tsflab/models/_components/{spec.name}/__init__.py",
        *tests[:6],
        *(f"src/tsflab/models/{name}/model.py" for name in consumers[:3]),
    )


def _component_show(rest: list[str]) -> int:
    from tsflab.catalog.cards.components import component_card_path
    from tsflab.catalog.cards.show import parse_show, show_card
    from tsflab.catalog.components import COMPONENT_CATALOG

    parsed = parse_show("tsf catalog show --kind component", "component name", rest)
    try:
        spec = COMPONENT_CATALOG.get(parsed.name)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    consumers = _component_consumers(spec.name)
    card = f"src/tsflab/models/_components/{spec.name}/README.md"
    legacy = {
        "name": spec.name,
        "module": spec.module,
        "contract": spec.contract,
        "public_symbols": list(spec.public_symbols),
        "keywords": list(spec.keywords),
        "consumers": consumers,
        "card": card,
    }
    shown = ", ".join(consumers[:8]) + (", ..." if len(consumers) > 8 else "")
    facts = {
        "public_symbols": list(spec.public_symbols),
        "consumers": f"{len(consumers)} models: {shown}" if consumers else "none",
    }
    paths = _component_show_paths(spec, card, consumers)
    return show_card(ROOT, component_card_path(ROOT, spec.name), parsed, facts=facts, paths=paths, legacy=legacy)


def _component_search(rest: list[str]) -> int:
    from tsflab.catalog.cards.search import search_command
    from tsflab.catalog.components import COMPONENT_CATALOG

    def augment(record: dict[str, object]) -> dict[str, object]:
        spec = COMPONENT_CATALOG.get(str(record["name"]))
        return {
            **record,
            "contract": spec.contract,
            "module": spec.module,
            "review_required": True,
        }

    code = search_command(ROOT, rest, prog="tsf catalog search --kind component", kind="component", augment=augment)
    if code == 0 and "--json" not in rest:
        print(
            "Candidate retrieval only; open one with `tsf catalog show <name> "
            "--kind component --depth 1` and review shapes and semantics before reuse.",
            file=sys.stderr,
        )
    return code


def _component_impl(args: list[str]) -> int:
    """List, match, or describe shared components and their consumers."""
    if not args or args[0] in {"-h", "--help", "help"}:
        print("usage: tsf catalog {list,show,search} --kind component")
        return 0
    action, rest = args[0], args[1:]
    if action == "audit":
        return _component_audit(rest)
    if action == "list" and (not rest or rest == ["--json"]):
        return _component_list(rest)
    if action == "show":
        return _component_show(rest)
    if action == "search":
        return _component_search(rest)
    print("usage: tsf catalog {list,show,search} --kind component", file=sys.stderr)
    return 2


def _catalog_counts() -> dict[str, object]:
    """Counts of every catalog, grouped, plus the drill-down commands."""
    from collections import Counter

    from tsflab.catalog.cards.components import component_card_path
    from tsflab.catalog.cards.datasets import dataset_facts, dataset_records
    from tsflab.catalog.cards.depth import read_card
    from tsflab.catalog.cards.metadata import model_records
    from tsflab.catalog.cards.models import FAMILY_TAGS
    from tsflab.catalog.components import COMPONENT_CATALOG

    models = model_records(ROOT)
    families = Counter(tag for record in models for tag in record.get("tags", ()) if tag in FAMILY_TAGS)
    modes = Counter(mode for record in models for mode in record.get("capabilities", ()))
    categories = Counter(
        str(read_card(component_card_path(ROOT, name)).front.get("category", "uncategorized"))
        for name in COMPONENT_CATALOG.names()
    )
    records = dataset_records(ROOT)
    facts = dataset_facts(ROOT)
    domains = Counter(str(facts.get(r.name, {}).get("domain", "unknown")).split(" / ")[0] for r in records)
    return {
        "models": {
            "count": len(models),
            "families": dict(families.most_common()),
            "task_modes": dict(modes.most_common()),
        },
        "components": {"count": len(COMPONENT_CATALOG.names()), "categories": dict(categories.most_common())},
        "datasets": {"count": len(records), "domains": dict(domains.most_common())},
        "next": [
            "tsf catalog search <terms> [--kind model|component|dataset]   # L0 lines",
            "tsf catalog show <name> [--kind K]                             # L1 contract",
            "tsf catalog show <name> [--kind K] --depth 2|3                 # full card | files",
        ],
    }


def _catalog_overview(as_json: bool) -> int:
    """Print the entry view: what exists, grouped, and how to drill down."""
    payload = _catalog_counts()
    from tsflab.agent.modules import project_modules

    modules = project_modules()
    if modules is not None:
        payload["project_modules"] = modules
    if as_json:
        print(json.dumps(payload, indent=2))
        return 0

    def fmt(counter: dict) -> str:
        return ", ".join(f"{key} {value}" for key, value in counter.items()) or "(untagged)"

    print(f"models      {payload['models']['count']:>4}  families: {fmt(payload['models']['families'])}")
    print(f"            capabilities: {fmt(payload['models']['task_modes'])}")
    print(f"components  {payload['components']['count']:>4}  categories: {fmt(payload['components']['categories'])}")
    print(f"datasets    {payload['datasets']['count']:>4}  domains: {fmt(payload['datasets']['domains'])}")
    if modules is not None:
        print("\nProject modules: " + ", ".join(modules))
    print("\nNext:\n" + "\n".join(f"  {line}" for line in payload["next"]))
    return 0


def _extract_kind(args: list[str]) -> tuple[str | None, list[str]]:
    """Remove ``--kind K`` from ``args``; return the kind and the remaining args."""
    rest, kind, skip = [], None, False
    for index, arg in enumerate(args):
        if skip:
            skip = False
        elif arg == "--kind" and index + 1 < len(args):
            kind, skip = args[index + 1], True
        elif arg.startswith("--kind="):
            kind = arg.split("=", 1)[1]
        else:
            rest.append(arg)
    return kind, rest


def _kinds_of(name: str) -> list[str]:
    """Return every catalog kind that has a resource called ``name``."""
    from tsflab.catalog.cards.resources import dataset_records
    from tsflab.catalog.components import COMPONENT_CATALOG
    from tsflab.catalog.registry.models import MODEL_CATALOG

    found = []
    if any(candidate.casefold() == name.casefold() for candidate in MODEL_CATALOG.names()):
        found.append("model")
    if name in COMPONENT_CATALOG.names():
        found.append("component")
    if any(record.name == name or record.loader == name for record in dataset_records(ROOT)):
        found.append("dataset")
    return found


def _dataset_read(action: str, rest: list[str]) -> int:
    from tsflab.cli.commands.data_results import dataset_read

    return dataset_read(action, rest)


def _read(kind: str, action: str, rest: list[str]) -> int:
    if kind == "model":
        return _model_impl([action, *rest])
    if kind == "component":
        return _component_impl([action, *rest])
    return _dataset_read(action, rest)


def _catalog_usage() -> str:
    from tsflab.cli.inventory import group_usage

    return group_usage("catalog") + (
        "\nEach search result is one L0 line: name, kind, summary, tags. `show` opens L1 by default;\n"
        "--kind is needed only when a name exists in more than one catalog. Several names are shown\n"
        "(or matched) in turn; with --json they print one JSON list."
    )


def _catalog_search(rest: list[str]) -> int:
    kind, rest = _extract_kind(rest)
    if kind in {"model", "component", "dataset"}:
        return _read(kind, "search", rest)
    from tsflab.catalog.cards.search import search_command

    return search_command(ROOT, rest if kind is None else [*rest, "--kind", kind], prog="tsf catalog search")


def _catalog_list(kind: str | None, rest: list[str]) -> int:
    if kind is not None:
        return _read(kind, "list", rest)
    for each in ("model", "component", "dataset"):
        print(f"# {each}")
        code = _read(each, "list", rest)
        if code:
            return code
    return 0


def _catalog_list_or_show(action: str, rest: list[str]) -> int:
    if {"-h", "--help"} & set(rest):
        print(_catalog_usage())
        return 0
    kind, rest = _extract_kind(rest)
    if kind not in (None, "model", "component", "dataset"):
        print(f"unknown kind {kind!r}; choose model, component, or dataset", file=sys.stderr)
        return 2
    if action == "list":
        return _catalog_list(kind, rest)
    return _show_many(kind, rest)


def catalog_command(args: list[str]) -> int:
    """Overview of all catalogs, ranked L0 search, list, or show one resource."""
    if not args or args[0] == "overview" or args == ["--json"]:
        return _catalog_overview("--json" in args)
    if args[0] in {"-h", "--help", "help"}:
        print(_catalog_usage())
        return 0
    action, rest = args[0], args[1:]
    if action == "search":
        return _catalog_search(rest)
    if action == "match":
        from tsflab.catalog.match import match_command

        return match_command(rest, ROOT)
    if action not in {"list", "show"}:
        print(_catalog_usage(), file=sys.stderr)
        return 2
    return _catalog_list_or_show(action, rest)


def _resolve_kind(name: str, kind: str | None) -> str | None:
    """The catalog kind of ``name`` (``kind`` when given); print why when it is unknown or ambiguous."""
    if kind is not None:
        return kind
    kinds = _kinds_of(name)
    if not kinds:
        print(f"Unknown resource {name!r}; try `tsf catalog search {name}`", file=sys.stderr)
        return None
    if len(kinds) > 1:
        print(f"{name!r} exists as {', '.join(kinds)}; pass --kind", file=sys.stderr)
        return None
    return kinds[0]


def _show_parser():
    import argparse

    from tsflab.catalog.cards.depth import DEPTHS

    parser = argparse.ArgumentParser(prog="tsf catalog show")
    parser.add_argument("names", nargs="+", metavar="name")
    parser.add_argument("--depth", type=int, choices=DEPTHS, default=1)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--params", action="store_true", help="model parameter schema only (compact JSON)")
    return parser


def _show_captured(kind: str, name: str, flags: list[str], payloads: list[object]) -> int:
    """Show one resource as JSON into ``payloads`` instead of stdout; return its exit code."""
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        status = _read(kind, "show", [name, *flags])
    if status == 0:
        payloads.append(json.loads(buffer.getvalue()))
    return status


def _show_flags(parsed, kind: str | None) -> tuple[str | None, list[str]]:
    """The kind and the per-resource ``show`` flags implied by the parsed options."""
    flags = ["--depth", str(parsed.depth), *(["--json"] if parsed.json else [])]
    if parsed.params:
        kind, flags = kind or "model", [*flags, "--params"]
    return kind, flags


def _show_many(kind: str | None, rest: list[str]) -> int:
    """Show one or more resources in turn; several names with --json print one JSON list."""
    parsed = _show_parser().parse_args(rest)
    kind, flags = _show_flags(parsed, kind)
    if len(parsed.names) == 1:  # one name keeps the single-object output
        resolved = _resolve_kind(parsed.names[0], kind)
        return 2 if resolved is None else _read(resolved, "show", [parsed.names[0], *flags])
    code, payloads = 0, []
    for index, name in enumerate(parsed.names):
        resolved = _resolve_kind(name, kind)
        if resolved is None:
            code = 2
            continue
        if parsed.json:
            status = _show_captured(resolved, name, flags, payloads)
        else:
            if index:
                print()
            status = _read(resolved, "show", [name, *flags])
        code = code or status
    if parsed.json:
        _print(payloads)
    return code


def _model_audit_components(rest: list[str]) -> int:
    rest = [item for item in rest if item != "--components"]
    if rest:
        print("model audit --components takes no other arguments", file=sys.stderr)
        return 2
    return _component_impl(["audit"])


def model_command(args: list[str]) -> int:
    """Scaffold, add, verify, compose, or audit models (reads live under `tsf catalog`)."""
    from tsflab.cli.inventory import group_usage

    usage = group_usage("model") + "\nFind and read models with `tsf catalog search|show --kind model`."
    if not args or args[0] in {"-h", "--help", "help"}:
        print(usage)
        return 0
    action, rest = args[0], args[1:]
    if action == "verify":
        from tsflab.cli.commands.verification import verification_command

        return verification_command(rest)
    if action == "compose":
        from tsflab.research.composition import compose_command

        return compose_command(rest, ROOT)
    if action == "similar":
        from tsflab.catalog.similarity import similarity_command

        return similarity_command(rest, ROOT)
    if action == "issues":
        return _issues_command(rest)
    if action == "audit" and "--components" in rest:
        return _model_audit_components(rest)
    if action in {"scaffold", "add", "artifacts", "audit"}:
        return _model_impl([action, *rest])
    print(usage, file=sys.stderr)
    return 2
