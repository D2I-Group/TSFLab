"""Static audit for the flat shared-component layer."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from tsflab.catalog.components import COMPONENT_CATALOG
from tsflab.core.paths import repository_root

ROOT = repository_root()
COMPONENTS = ROOT / "src" / "tsflab" / "models" / "_components"
COMPONENT_PREFIX = "tsflab.models._components."


def _component_imports(path: Path) -> set[str]:
    imports: set[str] = set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return imports
    for node in ast.walk(tree):
        module = None
        if isinstance(node, ast.ImportFrom):
            module = node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(COMPONENT_PREFIX):
                    imports.add(alias.name.removeprefix(COMPONENT_PREFIX).split(".", 1)[0])
        if module and module.startswith(COMPONENT_PREFIX):
            imports.add(module.removeprefix(COMPONENT_PREFIX).split(".", 1)[0])
    return imports


def components_used_by(package: Path) -> tuple[str, ...]:
    """Return shared component modules imported anywhere below a package."""
    used: set[str] = set()
    for path in package.rglob("*.py"):
        if path.name == "spec.py":
            continue
        used.update(_component_imports(path))
    return tuple(sorted(used - {"audit", "catalog"}))


def component_dependency_closure(names: set[str]) -> tuple[str, ...]:
    """Resolve every cataloged component imported by the selected models._components."""
    catalog_names = set(COMPONENT_CATALOG.names())
    resolved: set[str] = set()
    pending = list(names)
    while pending:
        name = pending.pop()
        if name in resolved:
            continue
        if name not in catalog_names:
            raise KeyError(f"unknown component dependency {name!r}")
        resolved.add(name)
        module_path = COMPONENTS / name / "__init__.py"
        pending.extend(_component_imports(module_path) - resolved - {"audit", "catalog"})
    return tuple(sorted(resolved))


def _source_claim_problems() -> list[str]:
    errors: list[str] = []
    derived_source_claim = re.compile(r"\b(?:ported|copied)\s+from\b", re.IGNORECASE)
    for path in sorted(COMPONENTS.glob("*/__init__.py")):
        text = path.read_text(encoding="utf-8")
        if derived_source_claim.search(text):
            errors.append(
                f"{path.relative_to(ROOT)} claims copied or ported source; "
                "shared components must be independently implemented or carry audited provenance"
            )
    return errors


def _spec_problems() -> list[str]:
    errors: list[str] = []
    for spec in COMPONENT_CATALOG.specs():
        if not spec.contract:
            errors.append(f"component {spec.name!r} has no card role (its semantic contract)")
        if not spec.keywords:
            errors.append(f"component {spec.name!r} has no card tags (its retrieval keywords)")
        if len(set(spec.keywords)) != len(spec.keywords):
            errors.append(f"component {spec.name!r} has duplicate card tags")
        if not spec.public_symbols:
            errors.append(f"component {spec.name!r} exports no public symbols")
    return errors


def _consumer_problems(catalog_names: set[str]) -> tuple[list[str], set[str]]:
    """Imports of uncataloged components, and the set of components that are imported."""
    errors: list[str] = []
    ignored = {"__init__"}
    consumers = list((ROOT / "src" / "tsflab" / "models").rglob("*.py"))
    consumers += list(COMPONENTS.glob("*/__init__.py"))
    used: set[str] = set()
    for path in consumers:
        for name in _component_imports(path):
            if name in ignored:
                continue
            used.add(name)
            if name not in catalog_names:
                errors.append(f"{path.relative_to(ROOT)} imports uncataloged component {name!r}")
    return errors, used


def audit_components() -> list[str]:
    """Return component-package and consumer-import inconsistencies.

    The catalog is derived from the packages, so names and modules cannot drift;
    what remains is what a package or card can get wrong.
    """
    errors: list[str] = list(COMPONENT_CATALOG.problems())
    catalog_names = set(COMPONENT_CATALOG.names())
    errors += _source_claim_problems()
    errors += _spec_problems()
    consumer_errors, used = _consumer_problems(catalog_names)
    errors += consumer_errors
    for name in sorted(catalog_names - used):
        errors.append(f"component {name!r} has no model or component consumer")
    return errors


def main() -> int:
    errors = audit_components()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"Shared components OK: {len(COMPONENT_CATALOG.names())} modules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
