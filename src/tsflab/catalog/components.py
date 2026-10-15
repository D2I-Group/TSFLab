"""Flat metadata catalog for reusable model components, derived from the component packages.

The catalog documents shared implementation contracts without creating a
second model hierarchy. It is metadata only: models import the concrete
component modules directly, so catalog discovery never changes runtime code.

Each ``src/tsflab/models/_components/<name>/`` package with an ``__init__.py`` is
one entry, read statically (no imports):

* ``name`` is the directory name and ``module`` is ``tsflab.models._components.<name>``;
* ``contract`` is the card's ``role`` (one schema-checked sentence of at most 160
  characters in ``card.toml``; the longer README description is already the card
  summary that search ranks on);
* ``keywords`` are the card's ``tags``;
* ``public_symbols`` are the package's ``__all__`` when it defines one, otherwise its
  public top-level classes, functions, and assignments.

A package whose card is missing or malformed still gets an entry (with an empty
contract and no keywords) and is reported by :meth:`ComponentCatalog.problems`.
"""

from __future__ import annotations

import ast
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ComponentSpec:
    """Identity and semantic boundary of one reusable component module."""

    name: str
    module: str
    contract: str
    public_symbols: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComponentMatch:
    """One retrieval candidate; semantic compatibility still needs review."""

    spec: ComponentSpec
    score: int
    matched_terms: tuple[str, ...]


def public_symbols(init_file: Path) -> tuple[str, ...]:
    """Return ``__all__`` or the public top-level names of a module, read statically."""
    tree = ast.parse(init_file.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
            continue
        targets = (
            node.targets if isinstance(node, ast.Assign) else ([node.target] if isinstance(node, ast.AnnAssign) else [])
        )
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id == "__all__" and node.value is not None:
                return tuple(str(item) for item in ast.literal_eval(node.value))
            names.append(target.id)
    return tuple(dict.fromkeys(name for name in names if not name.startswith("_")))


def _component_spec(directory: Path) -> tuple[ComponentSpec, list[str]]:
    """Derive one component entry from its package and card, with any problems."""
    where = f"src/tsflab/models/_components/{directory.name}"
    problems: list[str] = []
    try:
        symbols = public_symbols(directory / "__init__.py")
    except (SyntaxError, ValueError) as exc:
        symbols = ()
        problems.append(f"{where}/__init__.py cannot be read statically: {exc}")
    contract, keywords = "", ()
    try:
        card = tomllib.loads((directory / "card.toml").read_text(encoding="utf-8"))
        contract = str(card.get("role") or "").strip()
        keywords = tuple(str(tag) for tag in card.get("tags") or ())
    except (OSError, ValueError) as exc:
        problems.append(f"{where}/card.toml cannot be read: {exc}")
    spec = ComponentSpec(directory.name, f"tsflab.models._components.{directory.name}", contract, symbols, keywords)
    return spec, problems


class ComponentCatalog:
    """Flat lookup for shared component metadata, scanned once on first use."""

    def __init__(self, specs: tuple[ComponentSpec, ...] | None = None, *, directory: Path | None = None) -> None:
        self._directory = directory or Path(__file__).resolve().parents[1] / "models" / "_components"
        self._specs: dict[str, ComponentSpec] | None = None
        self._problems: list[str] = []
        if specs is not None:
            self._specs = {spec.name: spec for spec in specs}
            if len(self._specs) != len(specs):
                raise ValueError("component names must be unique")

    def _discover(self) -> dict[str, ComponentSpec]:
        if self._specs is None:
            specs: dict[str, ComponentSpec] = {}
            problems: list[str] = []
            for init_file in sorted(self._directory.glob("*/__init__.py")):
                spec, found = _component_spec(init_file.parent)
                specs[spec.name] = spec
                problems += found
            self._specs, self._problems = specs, problems
        return self._specs

    def refresh(self) -> None:
        """Forget the cached scan (after packages or cards changed in-process)."""
        self._specs, self._problems = None, []

    def names(self) -> list[str]:
        return sorted(self._discover())

    def get(self, name: str) -> ComponentSpec:
        try:
            return self._discover()[name]
        except KeyError as exc:
            raise KeyError(f"Unknown component {name!r}") from exc

    def specs(self) -> tuple[ComponentSpec, ...]:
        specs = self._discover()
        return tuple(specs[name] for name in sorted(specs))

    def problems(self) -> list[str]:
        """Component packages whose source or card could not be read."""
        self._discover()
        return list(self._problems)

    def match(self, query: str, limit: int = 5) -> tuple[ComponentMatch, ...]:
        """Rank lexical candidates without claiming semantic equivalence."""
        terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        if not terms or limit < 1:
            return ()
        matches: list[ComponentMatch] = []
        for spec in self.specs():
            name_terms = set(re.findall(r"[a-z0-9]+", spec.name.lower()))
            symbol_terms = set(re.findall(r"[a-z0-9]+", " ".join(spec.public_symbols).lower()))
            keyword_terms = set(re.findall(r"[a-z0-9]+", " ".join(spec.keywords).lower()))
            contract_terms = set(re.findall(r"[a-z0-9]+", spec.contract.lower()))
            matched = terms & (name_terms | symbol_terms | keyword_terms | contract_terms)
            if not matched:
                continue
            score = sum(
                5 if term in name_terms else 3 if term in keyword_terms else 2 if term in symbol_terms else 1
                for term in matched
            )
            matches.append(ComponentMatch(spec, score, tuple(sorted(matched))))
        matches.sort(key=lambda item: (-item.score, item.spec.name))
        return tuple(matches[:limit])


COMPONENT_CATALOG = ComponentCatalog()
