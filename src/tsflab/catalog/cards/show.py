"""Depth-aware ``show`` shared by the model, component, and dataset commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tsflab.catalog.cards.depth import (
    DEPTH_HELP,
    DEPTHS,
    card_payload,
    read_card,
    render_text,
)


def parse_show(prog: str, noun: str, args: list[str]) -> argparse.Namespace:
    """Parse ``<name> [--depth N] [--json]`` for any resource kind."""
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("name", help=noun)
    parser.add_argument("--depth", type=int, choices=DEPTHS, default=1, help=f"{DEPTH_HELP} (default 1)")
    parser.add_argument("--json", action="store_true", help="structured output")
    return parser.parse_args(args)


def show_card(
    root: Path,
    card_path: Path,
    parsed: argparse.Namespace,
    *,
    facts: dict[str, object],
    paths: list[str],
    legacy: dict[str, object],
) -> int:
    """Print one card at the requested depth (text, or JSON with ``--json``).

    ``facts`` are runtime facts shown after the front matter at L1; ``legacy``
    holds the structured machine fields merged into JSON from L1 upward.
    """
    card = read_card(card_path)
    if parsed.json:
        payload = card_payload(card, parsed.depth, facts=facts, paths=paths)
        if parsed.depth >= 1:
            payload = {**legacy, **payload}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        text = render_text(card, parsed.depth, facts=facts, paths=paths)
        print(text, end="" if text.endswith("\n") else "\n")
    return 0


def existing(root: Path, *candidates: str) -> list[str]:
    """Keep the repo-relative candidates that exist, preserving order."""
    return [path for path in dict.fromkeys(candidates) if path and (root / path).exists()]
