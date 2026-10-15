"""Model-card rules that need the code catalog: architecture families and composition slots."""

from __future__ import annotations

import re

#: Architecture families; every card names at least one so retrieval by family works.
FAMILY_TAGS = frozenset(
    {
        "linear",
        "mlp",
        "transformer",
        "cnn",
        "rnn",
        "gnn",
        "ssm",
        "kan",
        "llm",
        "foundation",
        "diffusion",
        "tree",
        "statistical",
        "hybrid",
    }
)
#: The recombination slots shared with autoresearch, in this order.
COMPOSITION_SLOTS = ("normalization", "decomposition", "temporal", "channel", "head", "loss")
_PART = re.compile(r"^(component|local|loss):[A-Za-z0-9_.+/-]+$")


def _composition_problems(entries: list[object], where: str) -> list[str]:
    from tsflab.catalog.components import COMPONENT_CATALOG
    from tsflab.catalog.registry.losses import LOSS_NAME_MAP

    problems: list[str] = []
    slots = [str(entry).split("=", 1)[0] for entry in entries]
    if slots != list(COMPOSITION_SLOTS):
        return [f"{where} composition must list slots {', '.join(COMPOSITION_SLOTS)} in order"]
    components = set(COMPONENT_CATALOG.names())
    losses = set(LOSS_NAME_MAP)
    for entry in entries:
        slot, _, value = str(entry).partition("=")
        if value == "none":
            continue
        for part in value.split("+"):
            problems += _part_problems(where, slot, part, components, losses)
    return problems


def _part_problems(where: str, slot: str, part: str, components: set[str], losses: set[str]) -> list[str]:
    if not _PART.match(part):
        return [f"{where} composition {slot}: {part!r} is not component:/local:/loss: or none"]
    problems: list[str] = []
    kind, name = part.split(":", 1)
    if kind == "component" and name not in components:
        problems.append(f"{where} composition {slot}: unknown component {name!r}")
    if kind == "loss" and (slot != "loss" or name not in losses):
        problems.append(f"{where} composition {slot}: {part!r} is not a registered loss in the loss slot")
    return problems
