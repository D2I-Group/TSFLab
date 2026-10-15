"""Ranked, depth-0 retrieval across the model, component, and dataset catalogs.

Every search returns L0 records (``name``, ``kind``, ``summary``, ``tags``; see
``card_depth``) so an agent can scan many candidates cheaply and open only the
few that matter with ``show --depth``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

from tsflab.catalog.cards.depth import (
    card_l0,
    l0_line,
    l0_record,
    read_card,
)

KINDS = ("model", "component", "dataset")
# Weight of a term found in each surface; every distinct matched term also adds 100.
_WEIGHTS = {"name": 8, "tags": 5, "summary": 4, "paper": 3, "card": 1}


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def _model_resources(root: Path) -> list[tuple[dict[str, object], dict[str, str], dict]]:
    from tsflab.catalog.cards.metadata import model_records

    resources = []
    for fields in model_records(root):
        card = (root / str(fields["model_card"])).read_text(encoding="utf-8")
        tags = list(
            dict.fromkeys(
                [
                    *fields.get("tags", ()),
                    *fields.get("fits", ()),
                    *sorted(set(fields.get("capabilities", ()))),
                    *fields.get("components", ()),
                ]
            )
        )
        paper = dict(fields["paper"])
        surfaces = {
            "name": str(fields["name"]).casefold(),
            "tags": " ".join(tags).casefold(),
            "summary": f"{fields.get('tagline', '')} {fields['summary']}".casefold(),
            "paper": str(paper["title"]).casefold(),
            "card": card.casefold(),
        }
        resources.append(
            (
                l0_record(str(fields["name"]), "model", str(fields.get("tagline") or fields["summary"]), tags),
                surfaces,
                fields,
            )
        )
    return resources


def _card_resources(root: Path, kind: str) -> list[tuple[dict[str, object], dict[str, str], dict]]:
    if kind == "component":
        from tsflab.catalog.cards.components import component_card_path
        from tsflab.catalog.components import COMPONENT_CATALOG

        entries = [(spec.name, component_card_path(root, spec.name), spec) for spec in COMPONENT_CATALOG.specs()]
    else:
        from tsflab.catalog.cards.resources import dataset_card_path, dataset_records

        entries = [(record.name, dataset_card_path(root, record.name), record) for record in dataset_records(root)]
    resources = []
    for name, path, source in entries:
        try:
            card = read_card(path)
        except (OSError, ValueError):
            continue
        record = card_l0(card)
        extra = ""
        if kind == "component":
            extra = " ".join(
                (
                    source.contract,
                    *source.public_symbols,
                    str(card.front.get("category", "")),
                    str(card.front.get("slot", "")),
                    *card.front.get("fits", ()),
                    str(card.front.get("summary", "")),
                )
            )
        else:
            extra = " ".join(
                (
                    source.alias,
                    source.loader,
                    source.dataset_id or source.path,
                    source.track,
                    str(card.front.get("domain", "")),
                    str(card.front.get("topic", "")),
                    *card.front.get("benchmarks", ()),
                )
            )
        surfaces = {
            "name": f"{name} {getattr(source, 'alias', '')}".casefold(),
            "tags": " ".join(card.tags).casefold(),
            "summary": f"{card.summary} {extra}".casefold(),
            "paper": "",
            "card": card.text.casefold(),
        }
        resources.append((record, surfaces, {"source": source}))
    return resources


def search_catalog(
    root: Path,
    query: str,
    kinds: tuple[str, ...] = KINDS,
    *,
    capabilities: tuple[str, ...] = (),
    limit: int = 10,
) -> list[dict[str, object]]:
    """Return ranked L0 records with ``score`` and ``matched_terms`` attached."""
    terms = _terms(query)
    results: list[dict[str, object]] = []
    for kind in kinds:
        resources = _model_resources(root) if kind == "model" else _card_resources(root, kind)
        for record, surfaces, fields in resources:
            if capabilities and not set(capabilities).issubset(fields.get("capabilities", ())):
                continue
            matched = {t for t in terms if any(t in text for text in surfaces.values())}
            if terms and not matched:
                continue
            # Each matched term scores every surface it appears in, so a term carried by
            # the tags, the description, and the paper outranks one found only in prose.
            score = len(matched) * 100 + sum(
                w for t in matched for surface, w in _WEIGHTS.items() if t in surfaces[surface]
            )
            # An exact name (or alias) hit outranks partial matches on longer names.
            if set(surfaces["name"].split()) & terms:
                score += 50
            results.append({**record, "score": score, "matched_terms": sorted(matched)})
    results.sort(key=lambda item: (-int(item["score"]), str(item["kind"]), str(item["name"])))
    return results[:limit]


def search_command(
    root: Path,
    args: list[str],
    *,
    prog: str,
    kind: str | None = None,
    augment: Callable[[dict[str, object]], dict[str, object]] | None = None,
) -> int:
    """Parse and run one search; print L0 lines, or JSON with ``--json``."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Ranked search returning one L0 line per result: "
        "name, kind, summary, tags. Open a result with `show --depth 1|2|3`.",
    )
    parser.add_argument("query", nargs="*", help="terms describing the need")
    if kind is None:
        parser.add_argument("--kind", choices=KINDS, help="restrict to one catalog")
    if kind in (None, "model"):
        parser.add_argument(
            "--capability",
            action="append",
            default=[],
            help="require a model capability; repeat for intersection",
        )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    parsed = parser.parse_args(args)
    if parsed.limit < 1:
        parser.error("--limit must be positive")
    capabilities = tuple(getattr(parsed, "capability", ()))
    if not parsed.query and not capabilities:
        parser.error("give at least one query term")
    selected = kind or getattr(parsed, "kind", None)
    matches = search_catalog(
        root,
        " ".join(parsed.query),
        (selected,) if selected else KINDS,
        capabilities=capabilities,
        limit=parsed.limit,
    )
    if parsed.json:
        if augment is not None:
            matches = [augment(match) for match in matches]
        print(json.dumps(matches, ensure_ascii=False, indent=2))
    else:
        for match in matches:
            print(l0_line(match))
        if not matches:
            print("no matches", file=sys.stderr)
    return 0
