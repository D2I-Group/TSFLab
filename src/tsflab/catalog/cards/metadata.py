"""Torch-free readers for runtime specs and canonical model-card metadata."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


def _literal(node: ast.expr):
    """Read literal values and frozensets used by runtime specifications."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "frozenset":
            return tuple(ast.literal_eval(node.args[0])) if node.args else ()
    return ast.literal_eval(node)


def declared_model_fields(spec_file: Path) -> dict[str, object]:
    """Read literal top-level ``ModelSpec`` runtime fields without imports."""
    tree = ast.parse(spec_file.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "SPEC" for target in node.targets):
            continue
        if not isinstance(node.value, ast.Call):
            return {}
        fields: dict[str, object] = {}
        for keyword in node.value.keywords:
            if keyword.arg is None:
                continue
            try:
                fields[keyword.arg] = _literal(keyword.value)
            except (ValueError, TypeError):
                continue
        return fields
    return {}


_SPEC_ASSIGNMENT = re.compile(r"^SPEC\s*(?::[^=\n]*)?=", re.MULTILINE)


def declared_model_name(spec_file: Path) -> str | None:
    """Return the literal ``SPEC = ModelSpec(name=...)`` name, or ``None`` when absent.

    Catalog discovery runs this for every package, so it parses only the source
    from the top-level ``SPEC`` assignment onward and falls back to a full parse
    (:func:`declared_model_fields`) when that tail is not valid on its own.
    """
    try:
        text = spec_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    match = _SPEC_ASSIGNMENT.search(text)
    if match is None:
        return None
    try:
        statement = ast.parse(text[match.start() :]).body[0]
        value = statement.value if isinstance(statement, (ast.Assign, ast.AnnAssign)) else None
        keywords = value.keywords if isinstance(value, ast.Call) else []
        name = next((_literal(k.value) for k in keywords if k.arg == "name"), None)
    except (SyntaxError, ValueError, TypeError, IndexError):
        try:
            name = declared_model_fields(spec_file).get("name")
        except SyntaxError:
            return None
    return name if isinstance(name, str) and name else None


def _scalar(value: str) -> object:
    """Parse the deliberately small scalar subset used by model cards."""
    value = value.strip()
    if not value or value in {"null", "~"}:
        return None
    if value in {"true", "false"}:
        return value == "true"
    if value.startswith(('"', "'")):
        return json.loads(value) if value.startswith('"') else value[1:-1]
    if value.startswith("["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    try:
        return int(value)
    except ValueError:
        return value


def read_model_card(path: Path) -> dict[str, object]:
    """Return normalized metadata for a model card (``card.toml`` + README description).

    ``path`` is the card's README or its directory. Runtime consumers receive the
    same flat shape as before (``summary``, ``tagline``, ``tags``, ``composition``
    as ``slot=value`` strings, ``paper``, ``codebase``) plus the card's ``fits``,
    ``fidelity``, ``data_params``, ``issues``, and ``admission``.
    """
    from tsflab.catalog.cards.store import load

    card = load(path if path.is_dir() else path.parent)
    facts = card.facts
    if facts.get("kind") != "model":
        raise ValueError(f"{card.directory}/card.toml is not a model card")
    paper = facts.get("paper") or {}
    code = facts.get("code")
    return {
        "name": facts["name"],
        "summary": card.description,
        "tagline": card.description,
        "tags": list(facts.get("tags") or []),
        "composition": [f"{slot}={value}" for slot, value in (facts.get("composition") or {}).items()],
        "paper": {
            "title": paper.get("title", ""),
            "venue": paper.get("venue", ""),
            "year": paper.get("year"),
            "url": paper.get("url", ""),
        },
        "codebase": ({"url": code["url"], "revision": code["revision"], "license": code["license"]} if code else None),
        "fits": list(facts.get("fits") or []),
        "fidelity": facts.get("fidelity", ""),
        "data_params": dict(facts.get("data_params") or {}),
        "issues": list(facts.get("issues") or []),
        "admission": dict(facts.get("admission") or {}),
    }


def model_records(root: Path, refs: dict[str, str] | None = None) -> list[dict[str, object]]:
    """Return one record per catalog model (spec runtime fields + card metadata).

    The catalog is derived from the model packages (see
    ``tsflab.catalog.registry.models``). A discovered package whose card is
    missing, malformed, or names a different model is left out here rather than
    breaking every listing; the registry and card audits report it, and
    ``tsf model add`` refuses it, so it cannot pass ``tsf repo check --audit``.
    """
    if refs is None:
        from tsflab.catalog.registry.models import MODEL_CATALOG

        refs = MODEL_CATALOG.refs()
    records: list[dict[str, object]] = []
    for name, module_path in refs.items():
        path = root / "src" / Path(*module_path.split(".")).with_suffix(".py")
        runtime = declared_model_fields(path) if path.is_file() else {}
        card_path = path.parent / "README.md"
        try:
            metadata = read_model_card(card_path)
        except (OSError, ValueError, KeyError):
            continue
        if runtime.get("name") != name or metadata.get("name") != name:
            continue
        fields = {**runtime, **metadata}
        fields["package"] = path.parent.name
        fields["spec_file"] = str(path.relative_to(root))
        fields["model_card"] = str(card_path.relative_to(root))
        records.append(fields)
    return sorted(records, key=lambda record: str(record["name"]).casefold())
