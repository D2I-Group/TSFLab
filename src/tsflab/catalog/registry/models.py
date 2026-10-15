"""Flat model catalog derived from the model packages, and model specification contracts."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


@dataclass(frozen=True)
class ModelArtifact:
    """Pinned external artifact required by an optional model runtime path."""

    name: str
    url: str
    revision: str
    sha256: str
    filename: str
    required: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.revision or not self.filename:
            raise ValueError("artifact name, revision, and filename must be non-empty")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError(f"artifact {self.name!r} needs a lowercase SHA-256 digest")
        if not self.url.startswith(("hf://", "https://", "file://")):
            raise ValueError(f"artifact {self.name!r} URL must use hf://, https://, or file://")
        if self.url.startswith("hf://"):
            from tsflab.release.hub.uri import parse

            if parse(self.url).revision != self.revision:
                raise ValueError(f"artifact {self.name!r} hf:// URI must be pinned to its revision")
        elif self.url.startswith("https://") and self.revision not in self.url:
            raise ValueError(f"artifact {self.name!r} URL must contain its pinned revision")
        if "/" in self.filename or "\\" in self.filename:
            raise ValueError(f"artifact {self.name!r} filename must be a basename")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_class: type
    factory: Callable
    params_schema: type[BaseModel]
    #: Derived from the package of ``model_class`` when left empty: ``tsflab.models.<package>``,
    #: ``configs/models/<name>.toml``, and ``src/tsflab/models/<package>/README.md``.
    module: str = ""
    config_path: str = ""
    model_card: str = ""
    smoke_config: str | None = None
    capabilities: frozenset[str] = field(default_factory=frozenset)
    components: tuple[str, ...] = ()
    contract_task: dict[str, int | str] = field(default_factory=dict)
    contract_seeds: tuple[int, ...] = (0,)
    #: The model needs the dataset adjacency (``adj_mx``, injected by the runner) and
    #: refuses to build without it; the admission contract supplies a synthetic graph.
    requires_graph: bool = False
    artifacts: tuple[ModelArtifact, ...] = ()
    artifact_factory: Callable | None = None
    # Opt-in paper objective used only while training; contract in
    # ``tsflab.experiments.runner.objective``. ``training_setup`` prepares it once
    # from the training split (for example fitting a label basis).
    training_objective: Callable | None = None
    training_setup: Callable | None = None

    def __post_init__(self) -> None:
        if not self.module:
            object.__setattr__(self, "module", self.model_class.__module__.rpartition(".")[0])
        if not self.config_path:
            object.__setattr__(self, "config_path", f"configs/models/{self.name}.toml")
        if not self.model_card:
            object.__setattr__(self, "model_card", f"src/{self.module.replace('.', '/')}/README.md")
        output_capabilities = self.capabilities & {"quantile-output", "distribution-output"}
        if len(output_capabilities) > 1:
            raise ValueError(f"model {self.name!r} declares conflicting output capabilities")
        if not self.task_modes:
            raise ValueError(f"model {self.name!r} declares no supported task mode")
        if len(set(self.components)) != len(self.components):
            raise ValueError(f"model {self.name!r} declares duplicate components")
        if self.training_objective is not None and not callable(self.training_objective):
            raise TypeError(f"model {self.name!r} training_objective must be callable")
        if self.training_setup is not None:
            if not callable(self.training_setup):
                raise TypeError(f"model {self.name!r} training_setup must be callable")
            if self.training_objective is None:
                raise ValueError(f"model {self.name!r} declares training_setup without training_objective")
        if "inference-only" in self.capabilities:
            if "pretraining-stage" in self.capabilities:
                raise ValueError(f"model {self.name!r} cannot be inference-only and pretraining-stage")
            if self.training_objective is not None:
                raise ValueError(f"model {self.name!r} cannot be inference-only with a training objective")
        artifact_names = [artifact.name for artifact in self.artifacts]
        if len(set(artifact_names)) != len(artifact_names):
            raise ValueError(f"model {self.name!r} declares duplicate artifacts")
        if self.artifact_factory is not None and not callable(self.artifact_factory):
            raise TypeError(f"model {self.name!r} artifact_factory must be callable")
        if self.artifact_factory is not None and not self.artifacts:
            raise ValueError(f"model {self.name!r} declares artifact_factory without artifacts")

    @property
    def output_type(self) -> Literal["point", "quantile", "distribution"]:
        """Declared public output kind, derived from orthogonal capabilities."""
        if "quantile-output" in self.capabilities:
            return "quantile"
        if "distribution-output" in self.capabilities:
            return "distribution"
        return "point"

    @property
    def task_modes(self) -> frozenset[str]:
        """Return supported public data settings from orthogonal capabilities."""
        mapping = {
            "time-series": "time_series",
            "spatiotemporal": "spatiotemporal",
            "covariate": "covariate",
        }
        return frozenset(mapping[key] for key in mapping if key in self.capabilities)

    def validate_params(self, params: dict) -> dict:
        unknown = sorted(set(params) - set(self.params_schema.model_fields))
        if unknown:
            raise ValueError(
                f"Unknown parameters for {self.name}: {', '.join(unknown)}. "
                f"Read declared fields with `tsf catalog show {self.name} --params`; do not infer parameter names."
            )
        return self.params_schema.model_validate(params).model_dump()

    def build(self, cfg, params: dict):
        return self.factory(cfg, self.validate_params(params))

    def build_with_artifacts(self, cfg, params: dict, paths: dict):
        """Construct an artifact-backed model through its explicit factory."""
        if self.artifact_factory is None:
            raise ValueError(f"model {self.name!r} declares runtime artifacts but has no artifact_factory")
        return self.artifact_factory(cfg, self.validate_params(params), dict(paths))


class ModelCatalog:
    """Name -> spec-module map derived from the model packages on disk.

    Every ``src/tsflab/models/<package>/spec.py`` (package names not starting with
    ``_``) whose top-level ``SPEC = ModelSpec(name="...")`` carries a literal name is
    a catalog entry. Discovery reads the name statically, so listing models never
    imports a model; :meth:`get` imports only the requested spec module. A package
    that cannot be discovered (no literal ``SPEC`` name, a duplicate name, Python
    files without ``spec.py``) is left out and reported by :meth:`problems`, which
    the registry audit turns into failures, instead of breaking the whole catalog.
    """

    def __init__(self, refs: dict[str, str] | None = None, *, models_dir: Path | None = None) -> None:
        self._fixed = None if refs is None else dict(refs)
        self._models_dir = models_dir or Path(__file__).resolve().parents[2] / "models"
        self._scan: tuple[dict[str, str], tuple[str, ...]] | None = None
        self._loaded: dict[str, ModelSpec] = {}

    def _discover(self) -> tuple[dict[str, str], tuple[str, ...]]:
        if self._fixed is not None:
            return self._fixed, ()
        if self._scan is None:
            self._scan = _scan_packages(self._models_dir)
        return self._scan

    def refresh(self) -> None:
        """Forget the cached scan (after packages were added or removed in-process)."""
        self._scan = None
        self._loaded.clear()

    def names(self) -> list[str]:
        return sorted(self._discover()[0])

    def refs(self) -> dict[str, str]:
        return dict(self._discover()[0])

    def problems(self) -> list[str]:
        """Model packages skipped by discovery, with the reason."""
        return list(self._discover()[1])

    def get(self, name: str) -> ModelSpec:
        if name in self._loaded:
            return self._loaded[name]
        module_path = self._discover()[0].get(name)
        if module_path is None:
            available = ", ".join(self.names())
            skipped = self.problems()
            hint = f" ({len(skipped)} package(s) skipped: {'; '.join(skipped)})" if skipped else ""
            raise KeyError(f"Unknown model {name!r}{hint}. Available: {available}")
        module = importlib.import_module(module_path)
        spec = getattr(module, "SPEC", None)
        if not isinstance(spec, ModelSpec):
            raise TypeError(f"{module_path} must expose a ModelSpec named SPEC")
        if spec.name != name:
            raise ValueError(f"discovered name {name!r} disagrees with {module_path}.SPEC.name={spec.name!r}")
        self._loaded[name] = spec
        return spec


def _scan_packages(models_dir: Path) -> tuple[dict[str, str], tuple[str, ...]]:
    """Return ``{name: "tsflab.models.<package>.spec"}`` and the skipped packages."""
    from tsflab.catalog.cards.metadata import declared_model_name

    found: dict[str, list[str]] = {}
    problems: list[str] = []
    for package in sorted(models_dir.iterdir()):
        if not package.is_dir() or package.name.startswith(("_", ".")):
            continue
        where = f"src/tsflab/models/{package.name}"
        spec_file = package / "spec.py"
        if not spec_file.is_file():
            if any(package.glob("*.py")):
                problems.append(f"{where} has Python files but no spec.py")
            continue
        name = declared_model_name(spec_file)
        if name is None:
            problems.append(f"{where}/spec.py has no top-level SPEC = ModelSpec(name=<literal>)")
            continue
        found.setdefault(name, []).append(f"tsflab.models.{package.name}.spec")
    refs: dict[str, str] = {}
    for name, modules in found.items():
        if len(modules) == 1:
            refs[name] = modules[0]
        else:
            problems.append(f"model name {name!r} is declared by {', '.join(sorted(modules))}")
    return refs, tuple(problems)


MODEL_CATALOG = ModelCatalog()
