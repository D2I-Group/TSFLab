"""Component-card location (cards follow ``tsflab.card/1``; see ``cards.schema``)."""

from __future__ import annotations

from pathlib import Path


def component_card_path(root: Path, name: str) -> Path:
    """Return the component card README path (``card.toml`` lives beside it)."""
    return root / "src" / "tsflab" / "models" / "_components" / name / "README.md"
