"""Static consistency checks for the model packages the flat catalog is derived from."""

from __future__ import annotations

import argparse
import ast
import sys
import tomllib
from pathlib import Path

from tsflab.catalog.admission import SCAFFOLD_MARKERS, unfinished_files
from tsflab.catalog.cards.metadata import declared_model_fields
from tsflab.catalog.component_audit import components_used_by
from tsflab.catalog.registry.models import MODEL_CATALOG
from tsflab.core.paths import repository_root


ROOT = repository_root()
MODEL_CONFIG_DIR = ROOT / "configs" / "models"
SRC = ROOT / "src"
MODELS = SRC / "tsflab" / "models"


def _cross_model_imports(package: Path) -> list[tuple[Path, int, str]]:
    """Find imports from one named model package into another named model package."""
    found: list[tuple[Path, int, str]] = []
    own_name = package.name
    for path in package.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            for module in modules:
                parts = module.split(".")
                if (
                    len(parts) >= 3
                    and parts[:2] == ["tsflab", "models"]
                    and parts[2] not in {own_name, "_components", "_slots", "_inputs"}
                ):
                    found.append((path, node.lineno, module))
    return found


def _audit_model_forward(path: Path) -> str | None:
    """Require the one public forecasting signature used by every runner."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return f"cannot parse {path.relative_to(ROOT)}: {exc}"
    model = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Model"),
        None,
    )
    forward = next(
        (
            node
            for node in model.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        ),
        None,
    ) if model is not None else None
    if forward is None:
        return f"{path.relative_to(ROOT)} must define Model.forward"
    arguments = forward.args
    names = [argument.arg for argument in arguments.posonlyargs + arguments.args]
    expected = ["self", "x_enc", "x_mark_enc", "x_dec", "x_mark_dec"]
    defaults_are_none = len(arguments.defaults) == 3 and all(
        isinstance(default, ast.Constant) and default.value is None
        for default in arguments.defaults
    )
    if (
        names != expected
        or arguments.posonlyargs
        or arguments.vararg is not None
        or arguments.kwarg is not None
        or arguments.kwonlyargs
        or not defaults_are_none
    ):
        return (
            f"{path.relative_to(ROOT)} Model.forward must be exactly "
            "(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None)"
        )
    return None


def _package_problems(name: str, package: Path, fields: dict[str, object]) -> list[str]:
    """Files, conventions, imports, and declared components of one discovered package."""
    problems: list[str] = []
    descriptive_fields = sorted(
        {"paper", "source", "evidence", "deviations", "implementation"} & fields.keys()
    )
    if descriptive_fields:
        problems.append(
            f"{package.relative_to(ROOT)}/spec.py duplicates README metadata: "
            f"{', '.join(descriptive_fields)}"
        )
    for obsolete in (package / "registry.py", package / "schema.py"):
        if obsolete.exists():
            problems.append(f"obsolete model file remains: {obsolete.relative_to(ROOT)}")
    for required in ("__init__.py", "model.py", "README.md", "card.toml"):
        if not (package / required).is_file():
            problems.append(f"{name!r} is missing {(package / required).relative_to(ROOT)}")
    for path in unfinished_files(package):
        problems.append(f"{path.relative_to(ROOT)} still has scaffold markers ({'/'.join(SCAFFOLD_MARKERS)})")
    model_file = package / "model.py"
    if model_file.is_file():
        forward_problem = _audit_model_forward(model_file)
        if forward_problem:
            problems.append(forward_problem)
    expected = {
        "module": f"tsflab.models.{package.name}",
        "config_path": f"configs/models/{name}.toml",
        "model_card": f"src/tsflab/models/{package.name}/README.md",
    }
    for field, value in expected.items():
        declared = fields.get(field, value)
        if declared != value:
            problems.append(f"{name!r} declares {field}={declared!r}, expected {value!r}")
    config_file = ROOT / expected["config_path"]
    if config_file.is_file():
        config_name = tomllib.loads(config_file.read_text(encoding="utf-8"))["model"]["name"]
        if config_name != name:
            problems.append(
                f"{config_file.relative_to(ROOT)} declares model.name={config_name!r}, expected {name!r}"
            )
    for path, line, module in _cross_model_imports(package):
        problems.append(
            f"{path.relative_to(ROOT)}:{line} imports peer model module {module!r}; "
            "extract the shared contract into components"
        )
    actual_components = components_used_by(package)
    declared_components = tuple(fields.get("components", ()))
    if declared_components != actual_components:
        problems.append(
            f"{name!r} components={declared_components!r}, imported={actual_components!r}"
        )
    return problems


def check() -> list[str]:
    """Problems in the discovered model catalog (empty when consistent).

    Names and spec modules are derived from the packages, so what is checked is
    what a package can get wrong: undiscoverable or duplicate specs, missing
    files, unfinished scaffolds, path conventions, presets, imports, and smoke configs.
    """
    problems: list[str] = list(MODEL_CATALOG.problems())
    refs = MODEL_CATALOG.refs()
    config_names = {path.stem for path in MODEL_CONFIG_DIR.glob("*.toml")}

    for path in sorted(MODELS.glob("*.py")):
        if path.name != "__init__.py":
            problems.append(f"shared code must not live at the model root: {path.relative_to(ROOT)}")
    for name in sorted(set(refs) - config_names):
        problems.append(f"model {name!r} has no configs/models/{name}.toml")
    for name in sorted(config_names - set(refs)):
        problems.append(f"configs/models/{name}.toml names no discovered model package")

    declared_smoke: dict[str, str] = {}
    for name, module_path in sorted(refs.items()):
        spec_file = SRC / Path(*module_path.split(".")).with_suffix(".py")
        fields = declared_model_fields(spec_file)
        problems += _package_problems(name, spec_file.parent, fields)
        smoke_config = fields.get("smoke_config")
        if smoke_config is not None:
            if not (ROOT / str(smoke_config)).is_file():
                problems.append(f"{name!r} declares missing smoke config {smoke_config!r}")
            prior_smoke = declared_smoke.get(str(smoke_config))
            if prior_smoke is not None:
                problems.append(f"{name!r} and {prior_smoke!r} both declare {smoke_config!r}")
            declared_smoke[str(smoke_config)] = name

    smoke_files = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "configs" / "runs").glob("smoke_*.toml")
    }
    for path in sorted(smoke_files - set(declared_smoke)):
        problems.append(f"{path} is not declared by any ModelSpec")

    return problems


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    problems = check()
    if not problems:
        print(f"OK: {len(MODEL_CATALOG.names())} discovered model packages match configs 1:1")
        return 0
    print(f"Found {len(problems)} catalog inconsistency(ies):")
    for problem in problems:
        print(f"  - {problem}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
