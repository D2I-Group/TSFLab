"""Write new cards in ``tsflab.card/1`` form (used by scaffolding and composition)."""

from __future__ import annotations

import json

from tsflab.catalog.cards.schema import README_SECTIONS, SCHEMA
from tsflab.catalog.cards.toml_io import dumps


def card_files(kind: str, facts: dict, description: str, sections: dict[str, str]) -> dict[str, str]:
    """Return ``{"card.toml": ..., "README.md": ...}`` with the fixed section order for ``kind``."""
    facts = {"schema": SCHEMA, "kind": kind, **{k: v for k, v in facts.items() if k not in {"schema", "kind"}}}
    order = README_SECTIONS["dataset" if kind == "dataset-family" else kind]
    unknown = set(sections) - set(order)
    if unknown:
        raise ValueError(f"unknown README sections for {kind}: {sorted(unknown)}")
    lines = [
        "---",
        f"name: {json.dumps(facts['name'])}",
        f"description: {json.dumps(' '.join(description.split()))}",
        "---",
        "",
        f"# {facts['name']}",
        "",
    ]
    for title in order:
        lines += [f"## {title}", "", sections.get(title, "").strip(), ""]
    return {"card.toml": dumps(facts, inline=("data_params",)), "README.md": "\n".join(lines).rstrip() + "\n"}
