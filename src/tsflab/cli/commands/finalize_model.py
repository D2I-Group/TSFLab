#!/usr/bin/env python3
"""Atomically admit a completed model workspace to the flat catalog.

Admission is static: structure, components, cards, and audits. Executed
verification (``--verify``) is recommended and is required before a final release.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import subprocess
import sys

from tsflab.catalog.component_audit import components_used_by
from tsflab.catalog.components import COMPONENT_CATALOG
from tsflab.catalog.cards.metadata import declared_model_fields
from tsflab.catalog.registry.models import ModelSpec
from tsflab.core.paths import repository_root, require_checkout


ROOT = repository_root()
CATALOG = ROOT / "src" / "tsflab" / "catalog" / "registry" / "models.py"


def _workspace(name: str) -> tuple[Path, dict[str, object]]:
    matches = []
    for spec_path in (ROOT / "src" / "tsflab" / "models").glob("*/spec.py"):
        fields = declared_model_fields(spec_path)
        if fields.get("name") == name:
            matches.append((spec_path.parent, fields))
    if len(matches) != 1:
        raise ValueError(f"expected one model workspace named {name!r}, found {len(matches)}")
    return matches[0]


def _preflight(name: str) -> tuple[Path, str]:
    package, fields = _workspace(name)
    module = str(fields.get("module", ""))
    expected_module = f"tsflab.models.{package.name}"
    if module != expected_module:
        raise ValueError(f"spec module must be {expected_module!r}, got {module!r}")
    required = [
        package / "__init__.py",
        package / "model.py",
        package / "spec.py",
        package / "README.md",
        package / "card.toml",
    ]
    config = ROOT / str(fields.get("config_path", ""))
    required.append(config)
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"missing model files: {', '.join(missing)}")

    marker_hits = []
    for path in (package / "model.py", package / "README.md", package / "card.toml"):
        text = path.read_text(encoding="utf-8")
        if any(marker in text for marker in ("SCAFFOLD", "PLACEHOLDER", "TODO")):
            marker_hits.append(str(path.relative_to(ROOT)))
    if marker_hits:
        raise ValueError(f"unfinished scaffold markers remain in: {', '.join(marker_hits)}")

    from tsflab.catalog.cards.store import load, problems

    card_problems = problems(load(package), curated=True)
    if card_problems:
        raise ValueError("model card is incomplete: " + "; ".join(card_problems))

    declared_components = set(fields.get("components", ()))
    unknown = declared_components - set(COMPONENT_CATALOG.names())
    if unknown:
        raise ValueError(f"unknown declared components: {', '.join(sorted(unknown))}")
    imported_components = components_used_by(package)
    if declared_components != set(imported_components):
        raise ValueError(
            "component declaration/import mismatch: "
            f"declared={sorted(declared_components)}, imported={sorted(imported_components)}"
        )

    imported = importlib.import_module(f"{module}.spec")
    spec = getattr(imported, "SPEC", None)
    if not isinstance(spec, ModelSpec) or spec.name != name:
        raise ValueError(f"{module}.spec must expose ModelSpec(name={name!r})")
    return package, module


def _insert_catalog(name: str, module: str, original: str) -> str:
    if f'"{name}":' in original or f"'{name}':" in original:
        raise ValueError(f"model {name!r} is already in MODEL_CATALOG")
    lines = original.splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("MODEL_CATALOG ="))
    close = next(index for index in range(start, len(lines)) if lines[index].rstrip() == "})")
    lines.insert(close, f'    "{name}": "{module}.spec",')
    return "\n".join(lines) + "\n"


def _run(*arguments: str) -> None:
    command = [sys.executable, "-m", "tsflab.cli.main", *arguments]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode:
        raise RuntimeError(f"failed gate: tsf {' '.join(arguments)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--verify", action="store_true",
                        help="also run `tsf model verify` and the strict runtime contracts")
    args = parser.parse_args()
    try:
        require_checkout("tsf model add")
        _, module = _preflight(args.name)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    original = CATALOG.read_text(encoding="utf-8")
    try:
        CATALOG.write_text(_insert_catalog(args.name, module, original), encoding="utf-8")
        if args.verify:
            _run("model", "verify", args.name)
        _run("model", "audit", args.name)
        # Cards and the model index are projections of the catalog, so they can
        # only be regenerated once the entry exists.
        _run("repo", "cards")
        # `model verify` already ran the strict contract; the audit covers component cards.
        _run("repo", "check", "--audit")
    except (RuntimeError, ValueError) as exc:
        CATALOG.write_text(original, encoding="utf-8")
        subprocess.run([sys.executable, "-m", "tsflab.cli.main", "repo", "cards"],
                       cwd=ROOT, check=False, capture_output=True)
        print(f"Model admission rolled back: {exc}", file=sys.stderr)
        return 1
    if args.verify:
        print(f"Added verified model {args.name!r} to the flat catalog")
    else:
        print(f"Added model {args.name!r} to the flat catalog; verification pending "
              f"(`tsf model verify {args.name}`, required before a final release)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
