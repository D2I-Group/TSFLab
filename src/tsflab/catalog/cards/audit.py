"""One audit for every card (``tsflab.card/1``): format, coverage, and code agreement.

* Format: schema, README front matter and fixed sections, L1/L2 budgets
  (``tsflab.catalog.cards.store.problems``).
* Coverage: every catalog model, every component package, and every dataset
  preset (or family) has exactly one card; no orphan cards.
* Code agreement: a model card's composition names only components the model
  imports; a component card names existing origin models; declined papers are
  not catalog models.
"""

from __future__ import annotations

from pathlib import Path

from tsflab.catalog.cards.store import card_directories, load, problems


def _model_problems(root: Path) -> list[str]:
    from tsflab.catalog.cards.models import FAMILY_TAGS, _composition_problems
    from tsflab.catalog.component_audit import component_dependency_closure, components_used_by
    from tsflab.catalog.registry.models import MODEL_CATALOG

    out: list[str] = []
    registered = {}
    for name, module in MODEL_CATALOG.refs().items():
        directory = root / "src" / Path(*module.split(".")[:-1])
        registered[directory.resolve()] = name
        try:
            card = load(directory)
        except ValueError as exc:
            out.append(str(exc))
            continue
        out += problems(card)
        where = str(directory.relative_to(root))
        if card.facts.get("name") != name:
            out.append(f"{where}: card name {card.facts.get('name')!r} != spec name {name!r}")
        tags = set(card.facts.get("tags") or [])
        if not tags & FAMILY_TAGS:
            out.append(f"{where}: tags must include an architecture family: {', '.join(sorted(FAMILY_TAGS))}")
        composition = [f"{k}={v}" for k, v in (card.facts.get("composition") or {}).items()]
        out += _composition_problems(composition, where)
        used = set(component_dependency_closure(set(components_used_by(directory))))
        named = {part.split(":", 1)[1] for value in (card.facts.get("composition") or {}).values()
                 for part in str(value).split("+") if part.startswith("component:")}
        for component in sorted(named - used):
            out.append(f"{where}: composition names component {component!r} that the model does not import")
    for directory in card_directories(root)["model"]:
        if directory.resolve() not in registered:
            out.append(f"orphan model card (package without a discoverable spec): {directory.relative_to(root)}")
    return out


def _component_problems(root: Path) -> list[str]:
    from tsflab.catalog.components import COMPONENT_CATALOG

    out: list[str] = []
    base = root / "src" / "tsflab" / "models" / "_components"
    names = set(COMPONENT_CATALOG.names())
    for spec in COMPONENT_CATALOG.specs():
        directory = base / spec.name
        try:
            card = load(directory)
        except ValueError as exc:
            out.append(str(exc))
            continue
        out += problems(card)
        where = str(directory.relative_to(root))
        if card.facts.get("name") != spec.name:
            out.append(f"{where}: card name differs from its directory {spec.name!r}")
        for slug in card.facts.get("origin_models") or []:
            if not (root / "src" / "tsflab" / "models" / str(slug) / "model.py").is_file():
                out.append(f"{where}: origin model {slug!r} is not a model package")
    for directory in card_directories(root)["component"]:
        if directory.name not in names:
            out.append(f"orphan component card: {directory.relative_to(root)}")
    return out


def _dataset_problems(root: Path) -> list[str]:
    from tsflab.catalog.cards.datasets import dataset_card_path, dataset_records, family_names

    out: list[str] = []
    records = dataset_records(root)
    expected = {dataset_card_path(root, r.name).parent.resolve() for r in records}
    expected |= {dataset_card_path(root, loader).parent.resolve() for loader in family_names(records)}
    for directory in sorted(expected):
        try:
            out += problems(load(directory))
        except ValueError as exc:
            out.append(str(exc))
    for directory in card_directories(root)["dataset"]:
        if directory.resolve() not in expected:
            out.append(f"orphan dataset card: {directory.relative_to(root)}")
    return out


def audit_cards(root: Path) -> list[str]:
    """Every card problem in the repository (empty when clean)."""
    from tsflab.catalog.cards.issues import load_declined
    from tsflab.catalog.cards.metadata import model_records

    out = _model_problems(root) + _component_problems(root) + _dataset_problems(root)
    declined, declined_problems = load_declined(root)
    out += declined_problems
    admitted = {str(record["name"]) for record in model_records(root)}
    out += [f"catalog/declined.toml lists {p['name']!r}, which is an admitted model"
            for p in declined if p.get("name") in admitted]
    return out
