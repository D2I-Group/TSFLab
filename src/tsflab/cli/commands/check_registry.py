"""Static consistency checks for the flat model catalog and specifications."""

from __future__ import annotations

import argparse
import ast
import sys
import tomllib
from pathlib import Path

from tsflab.core.paths import repository_root

from tsflab.catalog.cards.metadata import declared_model_fields
from tsflab.catalog.registry.models import MODEL_CATALOG
from tsflab.catalog.component_audit import components_used_by


ROOT = repository_root()
MODEL_CONFIG_DIR = ROOT / "configs" / "models"
SRC = ROOT / "src"
MODELS = SRC / "tsflab" / "models"


def _module_file(module_path: str) -> Path:
    return SRC / Path(*module_path.split(".")).with_suffix(".py")


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
                    and parts[2] not in {own_name, "_components", "_slots"}
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


def check() -> list[str]:
    problems: list[str] = []
    refs = MODEL_CATALOG.refs()
    config_names = {path.stem for path in MODEL_CONFIG_DIR.glob("*.toml")}
    catalog_names = set(refs)

    unexpected_root_modules = sorted(
        path.relative_to(ROOT)
        for path in MODELS.glob("*.py")
        if path.name != "__init__.py"
    )
    for path in unexpected_root_modules:
        problems.append(f"shared code must not live at the model root: {path}")

    for name in sorted(catalog_names - config_names):
        problems.append(f"catalog model {name!r} has no configs/models/{name}.toml")
    for name in sorted(config_names - catalog_names):
        problems.append(f"configs/models/{name}.toml is absent from MODEL_CATALOG")

    seen_modules: dict[str, str] = {}
    declared_smoke: dict[str, str] = {}
    for name, module_path in sorted(refs.items()):
        spec_file = _module_file(module_path)
        if not spec_file.is_file():
            problems.append(f"{name!r} -> {module_path!r} is missing {spec_file}")
            continue
        if spec_file.name != "spec.py":
            problems.append(f"{name!r} must resolve to a spec.py module")
        fields = declared_model_fields(spec_file)
        descriptive_fields = sorted(
            {"paper", "source", "evidence", "deviations", "implementation"}
            & fields.keys()
        )
        if descriptive_fields:
            problems.append(
                f"{spec_file.relative_to(ROOT)} duplicates README metadata: "
                f"{', '.join(descriptive_fields)}"
            )
        declared = fields.get("name")
        if declared != name:
            problems.append(
                f"catalog key {name!r} disagrees with {spec_file.relative_to(ROOT)} name={declared!r}"
            )
        package = spec_file.parent
        for obsolete in (package / "registry.py", package / "schema.py"):
            if obsolete.exists():
                problems.append(f"obsolete model file remains: {obsolete.relative_to(ROOT)}")
        for required in (
            package / "__init__.py",
            package / "model.py",
            package / "README.md",
            package / "card.toml",
        ):
            if not required.is_file():
                problems.append(f"{name!r} is missing {required.relative_to(ROOT)}")
        model_file = package / "model.py"
        if model_file.is_file():
            forward_problem = _audit_model_forward(model_file)
            if forward_problem:
                problems.append(forward_problem)
        expected_module = f"tsflab.models.{package.name}"
        declared_module = fields.get("module", expected_module)
        if declared_module != expected_module:
            problems.append(f"{name!r} declares module={declared_module!r}, expected {expected_module!r}")
        expected_config = f"configs/models/{name}.toml"
        config_path = fields.get("config_path", expected_config)
        if config_path != expected_config:
            problems.append(f"{name!r} declares config_path={config_path!r}, expected {expected_config!r}")
        expected_card = f"src/tsflab/models/{package.name}/README.md"
        model_card = fields.get("model_card", expected_card)
        if model_card != expected_card:
            problems.append(f"{name!r} declares model_card={model_card!r}, expected {expected_card!r}")
        config_file = ROOT / str(config_path)
        if config_file.is_file():
            config_name = tomllib.loads(config_file.read_text(encoding="utf-8"))["model"]["name"]
            if config_name != name:
                problems.append(f"{config_file.relative_to(ROOT)} declares model.name={config_name!r}, expected {name!r}")
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
        smoke_config = fields.get("smoke_config")
        if smoke_config is not None:
            smoke_path = ROOT / str(smoke_config)
            if not smoke_path.is_file():
                problems.append(f"{name!r} declares missing smoke config {smoke_config!r}")
            prior_smoke = declared_smoke.get(str(smoke_config))
            if prior_smoke is not None:
                problems.append(
                    f"{name!r} and {prior_smoke!r} both declare {smoke_config!r}"
                )
            declared_smoke[str(smoke_config)] = name
        prior = seen_modules.get(module_path)
        if prior is not None:
            problems.append(f"{name!r} and {prior!r} both use {module_path!r}")
        seen_modules[module_path] = name

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
        print(f"OK: {len(MODEL_CATALOG.names())} model specs match configs 1:1")
        return 0
    print(f"Found {len(problems)} catalog inconsistency(ies):")
    for problem in problems:
        print(f"  - {problem}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
