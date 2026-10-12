"""Match a dataset's characteristics to the models and components designed for them.

``tsf catalog match <dataset...> [--extra TERM ...] [--top N] [--json]`` reads the
dataset card's ``characteristics`` (measured data terms plus curated task terms),
adds any ``--extra`` task terms (for example ``probabilistic-output``), and ranks
every model and component whose card ``fits`` overlaps. Components are grouped by
the composition slot they fill, so the result reads as a recombination menu.
A match is a hypothesis to test against the baseline panel, not a verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _dataset_terms(root: Path, name: str) -> tuple[str, list[str], str]:
    from tsflab.catalog.cards.datasets import dataset_card_path, dataset_records
    from tsflab.catalog.cards.depth import read_card

    records = {r.name: r for r in dataset_records(root)}
    by_alias = {r.alias: r for r in records.values() if r.alias}
    record = records.get(name) or by_alias.get(name)
    if record is None:
        raise SystemExit(f"unknown dataset {name!r}; see `tsf catalog list --kind dataset`")
    card = read_card(dataset_card_path(root, record.name))
    terms = list(card.front.get("characteristics") or [])
    basis = str(card.front.get("characteristics_basis") or "not recorded")
    return record.name, terms, basis


def _rank(entries: list[dict], wanted: set[str]) -> list[dict]:
    ranked = []
    for entry in entries:
        fits = set(entry["fits"])
        hit = sorted(fits & wanted)
        if hit:
            ranked.append({**entry, "matched": hit, "score": len(hit)})
    ranked.sort(key=lambda e: (-e["score"], e["name"]))
    return ranked


def match(root: Path, dataset: str, extra: list[str]) -> dict[str, object]:
    from tsflab.catalog.cards.components import component_card_path
    from tsflab.catalog.cards.depth import read_card
    from tsflab.catalog.cards.metadata import model_records
    from tsflab.catalog.characteristics import vocabulary
    from tsflab.catalog.components import COMPONENT_CATALOG

    name, terms, basis = _dataset_terms(root, dataset)
    unknown = sorted(set(extra) - set(vocabulary()))
    if unknown:
        raise SystemExit(f"unknown characteristic(s) {unknown}; see tsflab.catalog.characteristics")
    wanted = set(terms) | set(extra)
    models = [
        {"name": str(r["name"]), "fits": list(r.get("fits") or []),
         "headline": str(r.get("tagline") or r["summary"])}
        for r in model_records(root)
    ]
    components = []
    for spec in COMPONENT_CATALOG.specs():
        card = read_card(component_card_path(root, spec.name))
        components.append({"name": spec.name, "fits": list(card.front.get("fits") or []),
                           "slot": card.front.get("slot"), "headline": card.headline})
    by_slot: dict[str, list[dict]] = {}
    for entry in _rank(components, wanted):
        by_slot.setdefault(str(entry["slot"] or "unassigned"), []).append(entry)
    generic = sorted(c["name"] for c in components if c["fits"] == ["any"])
    return {
        "dataset": name,
        "characteristics": sorted(wanted),
        "basis": basis,
        "models": _rank(models, wanted),
        "components": by_slot,
        "generic_components": generic,
        "unannotated": {
            "models": sum(1 for m in models if not m["fits"]),
            "components": sum(1 for c in components if not c["fits"]),
        },
    }


def match_command(argv: list[str], root: Path) -> int:
    parser = argparse.ArgumentParser(prog="tsf catalog match", description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="+", metavar="dataset",
                        help="dataset card name(s) or preset alias(es); several are matched in turn")
    parser.add_argument("--extra", nargs="*", default=[], help="additional task characteristics")
    parser.add_argument("--top", type=int, default=15, help="models to show (0 = all)")
    parser.add_argument("--json", action="store_true",
                        help="structured output (a list when several datasets are named)")
    args = parser.parse_args(argv)
    results = [match(root, name, args.extra) for name in args.datasets]
    if args.json:
        print(json.dumps(results[0] if len(results) == 1 else results, indent=2))
        return 0
    for index, result in enumerate(results):
        if index:
            print()
        _print_match(result, args.top)
    return 0


def _print_match(result: dict, top: int) -> None:
    print(f"{result['dataset']}: {', '.join(result['characteristics']) or '(no characteristics recorded)'}")
    print(f"  basis: {result['basis']}")
    models = result["models"][:top] if top else result["models"]
    print(f"\nModels ({len(result['models'])} match):")
    for entry in models:
        print(f"  {entry['score']}  {entry['name']:<22} [{', '.join(entry['matched'])}]  {entry['headline'][:80]}")
    print("\nComponents by slot:")
    for slot, entries in sorted(result["components"].items()):
        names = ", ".join(f"{e['name']}({e['score']})" for e in entries)
        print(f"  {slot:<13} {names}")
    gaps = result["unannotated"]
    if gaps["models"] or gaps["components"]:
        print(f"\nNot yet annotated with fits: {gaps['models']} models, {gaps['components']} components")
    return 0
