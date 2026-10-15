"""Setuptools hook that bundles only the read-only runtime/Agent catalog."""

from __future__ import annotations

from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py


ROOT = Path(__file__).resolve().parent


class BuildWithRepositoryAssets(build_py):
    """Copy only resources needed for installed catalog and Agent discovery."""

    def run(self) -> None:
        # Setuptools may reuse build/lib between invocations. Start from an
        # empty build tree so deleted runtime modules cannot reappear in a
        # later wheel.
        shutil.rmtree(self.build_lib, ignore_errors=True)
        super().run()
        target = Path(self.build_lib) / "tsflab" / "assets"
        target.mkdir(parents=True, exist_ok=True)
        # ``egg_info`` runs before ``build_py`` and creates ``src/*.egg-info``.
        # It belongs to the distribution metadata, not to the bundled checkout
        # snapshot, so exclude it together with ordinary build detritus.
        ignored = shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.egg-info", ".DS_Store"
        )
        # Runtime Python packages are already installed by build_py. The asset
        # tree contains only catalogs required by public inspection, verified
        # configs, and Agent workflows. Tests, scripts, and human docs
        # deliberately remain checkout-only.
        for directory in (
            ".agents",
            "catalog",
            "configs",
        ):
            shutil.copytree(
                ROOT / directory,
                target / directory,
                dirs_exist_ok=True,
                ignore=ignored,
            )
        # Catalog inspection needs cards and the literal ModelSpec declarations,
        # not a second copy of every installed model implementation. Preserve
        # their relative paths so the same torch-free readers work in a checkout
        # and in a wheel.
        model_assets = ROOT / "src" / "tsflab" / "models"
        for source in model_assets.rglob("*"):
            if not source.is_file() or source.name not in {"README.md", "card.toml", "reference.md", "spec.py"}:
                continue
            destination = target / "src" / "tsflab" / "models" / source.relative_to(
                model_assets
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        for filename in (
            "AGENTS.md",
            "LICENSE",
            "README.md",
            "THIRD_PARTY_NOTICES.md",
        ):
            shutil.copy2(ROOT / filename, target / filename)
        (target / ".packaged-assets").write_text(
            "read-only TSFLab runtime and Agent assets\n", encoding="utf-8"
        )


setup(cmdclass={"build_py": BuildWithRepositoryAssets})
