"""Read and check ``tsflab.card/1`` cards (see ``tsflab.catalog.cards.schema``).

A card directory holds ``card.toml`` (facts), ``README.md`` (front matter
``name`` + ``description`` = L0, fixed sections = L1), and an optional
``reference.md`` (L2). This module is the only reader of those files.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from tsflab.catalog.cards.schema import (
    CARD_TYPES,
    L1_MAX_LINES,
    L2_MIN_LINES,
    README_SECTIONS,
    SCHEMA,
)

TODO = "TODO(card-v1)"
#: Every card is curated: no TODO placeholders, model fits, component role/slot/fits.
REQUIRE_CURATED = True
DESCRIPTION_CHARS = 320


@dataclass(frozen=True)
class CardFiles:
    directory: Path
    facts: dict
    name: str
    description: str
    sections: tuple[tuple[str, str], ...]
    body: str  # README text after the front matter (L1)
    reference: str | None = None
    reference_sections: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def kind(self) -> str:
        return str(self.facts.get("kind", ""))


def _front(text: str, path: Path) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{path} must start with front matter")
    end = lines.index("---", 1)
    front: dict[str, str] = {}
    for line in lines[1:end]:
        if not line.strip():
            continue
        key, _, raw = line.partition(":")
        raw = raw.strip()
        front[key.strip()] = json.loads(raw) if raw.startswith('"') else raw
    return front, "\n".join(lines[end + 1 :]).strip()


def split_sections(text: str) -> tuple[tuple[str, str], ...]:
    """Level-two sections, ignoring headings inside fenced code."""
    out: list[tuple[str, list[str]]] = []
    fenced = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        if not fenced and line.startswith("## "):
            out.append((line[3:].strip(), []))
        elif out:
            out[-1][1].append(line)
    return tuple((title, "\n".join(body).strip()) for title, body in out)


def load(directory: Path) -> CardFiles:
    """Load one card directory; raises ``ValueError`` when files are missing or malformed."""
    toml_path, readme_path = directory / "card.toml", directory / "README.md"
    if not toml_path.is_file() or not readme_path.is_file():
        raise ValueError(f"{directory} needs card.toml and README.md")
    facts = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    front, body = _front(readme_path.read_text(encoding="utf-8"), readme_path)
    ref_path = directory / "reference.md"
    reference = ref_path.read_text(encoding="utf-8") if ref_path.is_file() else None
    return CardFiles(
        directory=directory,
        facts=facts,
        name=str(front.get("name", "")),
        description=" ".join(str(front.get("description", "")).split()),
        sections=split_sections(body),
        body=body,
        reference=reference,
        reference_sections=split_sections(reference or ""),
    )


def validated(card: CardFiles):
    """Return the typed card model, raising ``ValueError`` with readable messages."""
    kind = card.kind
    model = CARD_TYPES.get(kind)
    if model is None:
        raise ValueError(f"{card.directory}/card.toml: unknown kind {kind!r}")
    try:
        return model.model_validate(card.facts)
    except ValidationError as exc:
        details = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        raise ValueError(f"{card.directory}/card.toml: {details}") from None


def _schema_problems(card: CardFiles, where: str, curated: bool) -> list[str]:
    out: list[str] = []
    if card.facts.get("schema") != SCHEMA:
        out.append(f"{where}: card.toml schema must be {SCHEMA!r}")
    pending_curation = card.kind == "component" and not (
        card.facts.get("role") and card.facts.get("slot") and card.facts.get("fits")
    )
    if not (pending_curation and not curated):
        try:
            validated(card)
        except ValueError as exc:
            out.append(str(exc))
    return out


def _readme_problems(card: CardFiles, where: str) -> list[str]:
    out: list[str] = []
    kind = card.kind
    if card.name != card.facts.get("name"):
        out.append(f"{where}: README name {card.name!r} differs from card.toml name")
    if not card.description or len(card.description) > DESCRIPTION_CHARS:
        out.append(f"{where}: README description must be one line of 1-{DESCRIPTION_CHARS} characters")
    expected = README_SECTIONS["dataset" if kind == "dataset-family" else kind] if kind else ()
    titles = tuple(title for title, _ in card.sections)
    if titles != expected:
        out.append(f"{where}: README sections must be exactly {', '.join(expected)} (got {', '.join(titles)})")
    return out


def _length_problems(card: CardFiles, where: str) -> list[str]:
    out: list[str] = []
    l1 = len(card.body.splitlines())
    if l1 > L1_MAX_LINES:
        out.append(f"{where}: README body has {l1} lines; L1 budget is {L1_MAX_LINES}, move detail to reference.md")
    if card.reference is not None:
        if not card.reference.strip():
            out.append(f"{where}: empty reference.md")
        elif l1 + len(card.reference.splitlines()) < L1_MAX_LINES and len(card.reference.splitlines()) < L2_MIN_LINES:
            out.append(f"{where}: reference.md is short enough to fold into README")
    return out


def _curation_problems(card: CardFiles, where: str) -> list[str]:
    out: list[str] = []
    text = card.body + (card.reference or "")
    if TODO in text:
        out.append(f"{where}: {TODO} placeholders remain")
    if card.kind == "model" and not card.facts.get("fits"):
        out.append(f"{where}: card.toml needs fits")
    return out


def problems(card: CardFiles, *, curated: bool | None = None) -> list[str]:
    """Format problems of one card; content checks beyond the schema stay with each kind."""
    curated = REQUIRE_CURATED if curated is None else curated
    where = str(card.directory)
    out = _schema_problems(card, where, curated)
    out += _readme_problems(card, where)
    out += _length_problems(card, where)
    if curated:
        out += _curation_problems(card, where)
    return out


def card_directories(root: Path) -> dict[str, list[Path]]:
    """Every card directory by kind (models: registered packages are filtered by callers)."""
    models = sorted(
        p.parent for p in (root / "src/tsflab/models").glob("*/card.toml") if not p.parent.name.startswith("_")
    )
    components = sorted(p.parent for p in (root / "src/tsflab/models/_components").glob("*/card.toml"))
    datasets = sorted(p.parent for p in (root / "catalog/datasets").rglob("card.toml"))
    return {"model": models, "component": components, "dataset": datasets}


def _cli() -> int:  # pragma: no cover - used by curation agents and scripts
    import argparse

    from tsflab.core.paths import repository_root

    parser = argparse.ArgumentParser(description="Check tsflab.card/1 cards")
    parser.add_argument("paths", nargs="*", help="card directories (default: all)")
    parser.add_argument("--curated", action="store_true", help="also require curation (no TODO, fits)")
    args = parser.parse_args()
    root = repository_root()
    dirs = [Path(p).resolve() for p in args.paths] or [d for ds in card_directories(root).values() for d in ds]
    found = []
    for directory in dirs:
        try:
            found += problems(load(directory), curated=args.curated)
        except ValueError as exc:
            found.append(str(exc))
    for line in found:
        print(line)
    print(f"{len(dirs)} cards checked, {len(found)} problem(s)")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(_cli())
