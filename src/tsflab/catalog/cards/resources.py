"""Shared entry points for card lookups and the repository card audit."""

from __future__ import annotations

from pathlib import Path

from tsflab.catalog.cards.components import component_card_path  # noqa: F401  (public name)
from tsflab.catalog.cards.datasets import (  # noqa: F401  (public names)
    DatasetRecord,
    dataset_card_path,
    dataset_records,
)


def audit_resource_cards(root: Path) -> list[str]:
    """Every card problem (models, components, datasets, declined papers); see ``cards.audit``."""
    from tsflab.catalog.cards.audit import audit_cards

    return audit_cards(root)
