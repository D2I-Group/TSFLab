#!/usr/bin/env python3
"""Gate a completed model workspace into the flat catalog.

The catalog is derived from the model packages, so this command writes no
registry entry: it checks that the package is discoverable and complete, then
runs the gates (structure, components, cards, audits). Executed verification
(``--verify``) is recommended and is required before a final release.
"""

from __future__ import annotations

import argparse

from tsflab.cli.args import add_name, resolve_name
import importlib
from pathlib import Path
import subprocess
import sys

from tsflab.catalog.admission import unfinished_files
from tsflab.catalog.cards.metadata import declared_model_fields
from tsflab.catalog.component_audit import components_used_by
from tsflab.catalog.components import COMPONENT_CATALOG
from tsflab.catalog.registry.models import MODEL_CATALOG, ModelSpec
from tsflab.core.paths import repository_root, require_checkout


ROOT = repository_root()


def _workspace(name: str) -> tuple[Path, dict[str, object]]:
    """Return the discovered package of ``name`` and its literal spec fields."""
    MODEL_CATALOG.refresh()
    module_path = MODEL_CATALOG.refs().get(name)
    if module_path is None:
        skipped = MODEL_CATALOG.problems()
        detail = f"; skipped packages: {'; '.join(skipped)}" if skipped else ""
        raise ValueError(
            f"no discoverable model package declares SPEC = ModelSpec(name={name!r}){detail}"
        )
    spec_path = ROOT / "src" / Path(*module_path.split(".")).with_suffix(".py")
    return spec_path.parent, declared_model_fields(spec_path)


def _preflight(name: str) -> tuple[Path, str]:
    package, fields = _workspace(name)
    expected_module = f"tsflab.models.{package.name}"
    # ``module`` is normally left to its derived default; an explicit one must agree.
    module = str(fields.get("module", expected_module))
    if module != expected_module:
        raise ValueError(f"spec module must be {expected_module!r}, got {module!r}")
    required = [
        package / "__init__.py",
        package / "model.py",
        package / "spec.py",
        package / "README.md",
        package / "card.toml",
    ]
    config = ROOT / str(fields.get("config_path", f"configs/models/{name}.toml"))
    required.append(config)
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"missing model files: {', '.join(missing)}")

    marker_hits = [str(path.relative_to(ROOT)) for path in unfinished_files(package)]
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


def _run(*arguments: str) -> None:
    command = [sys.executable, "-m", "tsflab.cli.main", *arguments]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode:
        raise RuntimeError(f"failed gate: tsf {' '.join(arguments)}")


def _report_reuse(slug: str) -> None:
    """Name the shared components the new model's code resembles (nominations, never a failure)."""
    from tsflab.catalog.similarity import reuse_candidates

    hits = reuse_candidates(ROOT, slug)
    if not hits:
        return
    print("Reuse candidates (similar shared components; switch only if numerically identical):")
    for pair in hits:
        own, shared = (pair.right, pair.left) if pair.left.is_component else (pair.left, pair.right)
        print(f"  {own.qualname} ({own.path}:{own.line}) ~ {shared.owner.removeprefix('component:')}"
              f".{shared.qualname}  score {pair.score:.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(prog="tsf model add", description=__doc__)
    add_name(parser, "name", flag="--name", help="public model name of the scaffolded package")
    parser.add_argument("--verify", action="store_true",
                        help="also run `tsf model verify` and the strict runtime contracts")
    args = parser.parse_args()
    resolve_name(parser, args, "name")
    try:
        require_checkout("tsf model add")
        _, module = _preflight(args.name)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    try:
        if args.verify:
            _run("model", "verify", args.name)
        _run("model", "audit", args.name)
        # Regenerate the card projections (model index) now that the package is complete.
        _run("repo", "cards")
        # `model verify` already ran the strict contract; the audit covers component cards.
        _run("repo", "check", "--audit")
    except RuntimeError as exc:
        # Nothing to roll back: the package itself is the catalog entry.
        print(f"Model admission failed: {exc}; fix the package and rerun", file=sys.stderr)
        return 1
    _report_reuse(module.rpartition(".")[2])
    if args.verify:
        print(f"Admitted verified model {args.name!r} (discovered from its package)")
    else:
        print(f"Admitted model {args.name!r} (discovered from its package); verification pending "
              f"(`tsf model verify {args.name}`, required before a final release)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
