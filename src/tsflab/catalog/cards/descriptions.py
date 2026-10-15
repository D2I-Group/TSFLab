"""Read concise, human-facing descriptions from canonical model cards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tsflab.catalog.cards.metadata import read_model_card


@dataclass(frozen=True)
class ModelCardDescription:
    """The concise prose needed for catalog discovery without loading a model."""

    summary: str


def read_model_card_description(path: Path) -> ModelCardDescription:
    """Return the first prose paragraph after a model card's level-one title.

    Model cards remain the human-readable source of method descriptions. This
    parser intentionally reads only the introductory paragraph; paper abstracts
    and citations are kept separate and are not presented as repository claims.
    """
    card = read_model_card(path)
    declared = str(card.get("summary", "")).strip()
    lines = path.read_text(encoding="utf-8").splitlines()
    title_index = next(
        (index for index, line in enumerate(lines) if line.startswith("# ")),
        None,
    )
    if title_index is None:
        raise ValueError(f"{path} has no level-one title")

    summary = declared or " ".join(_first_paragraph(lines[title_index + 1 :]))
    if not summary:
        raise ValueError(f"{path} has no introductory method description")
    placeholders = ("this model is scaffolded", "replace this text", "todo")
    if any(marker in summary.casefold() for marker in placeholders):
        raise ValueError(f"{path} still contains a placeholder method description")
    return ModelCardDescription(summary=summary)


def _first_paragraph(lines: list[str]) -> list[str]:
    """Stripped lines of the first paragraph, stopping at the next heading."""
    paragraph: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if paragraph:
                break
            continue
        if stripped.startswith("#"):
            break
        paragraph.append(stripped)
    return paragraph
