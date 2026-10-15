"""Shared test setup: markers, the no-catalog-model guard, and generated fixture data.

The repository keeps infrastructure tests only. Catalog models are checked once,
at admission, by ``tsf model verify`` (strict executable contract, recorded in
each ``card.toml``), so ordinary tests must not construct them:

* everything under ``tests/e2e`` is marked ``e2e`` (run with ``-m e2e``);
* a test that genuinely needs a catalog model is marked ``model`` and runs only
  when ``TSFLAB_MODEL_TESTS=1`` (on a machine allowed to execute models);
* any other test that instantiates a class from ``tsflab.models.<model>`` fails.
  Shared infrastructure (``tsflab.models._components``, ``_foundation``,
  ``_slots``) is not a catalog model and stays allowed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
E2E = Path(__file__).resolve().parent / "e2e"


def pytest_collection_modifyitems(config, items) -> None:
    run_models = os.environ.get("TSFLAB_MODEL_TESTS") == "1"
    skip_model = pytest.mark.skip(reason="builds and runs models; set TSFLAB_MODEL_TESTS=1 to run")
    for item in items:
        if E2E in Path(str(item.fspath)).resolve().parents:
            item.add_marker(pytest.mark.e2e)
        if item.get_closest_marker("model") and not run_models:
            item.add_marker(skip_model)


def _catalog_model(cls: type) -> bool:
    parts = getattr(cls, "__module__", "").split(".")
    return len(parts) >= 3 and parts[:2] == ["tsflab", "models"] and not parts[2].startswith("_")


_GUARD = {"test": None}  # node id of the running guarded test, None when constructing is allowed


def _guarded_new(cls, *args, **kwargs):
    if _GUARD["test"] is not None and _catalog_model(cls):
        raise AssertionError(
            f"{_GUARD['test']} constructs catalog model class {cls.__module__}.{cls.__qualname__}; "
            "use a fake, or mark the test `model`"
        )
    return object.__new__(cls)


@pytest.fixture(scope="session", autouse=True)
def _install_guard():
    # Installed once and never removed: deleting an assigned ``__new__`` leaves
    # subclasses with a slot that rejects constructor arguments.
    import torch

    torch.nn.Module.__new__ = staticmethod(_guarded_new)
    yield


@pytest.fixture(autouse=True)
def no_catalog_models(request, _install_guard):
    """Fail a test that constructs a catalog model unless it is marked ``model`` or ``e2e``."""
    exempt = request.node.get_closest_marker("model") or request.node.get_closest_marker("e2e")
    _GUARD["test"] = None if exempt else request.node.nodeid
    try:
        yield
    finally:
        _GUARD["test"] = None


@pytest.fixture(scope="session", autouse=True)
def smoke_dataset() -> Path:
    """Create ``dataset/smoke/smoke.csv`` (gitignored) once, as CI starts from a clean tree."""
    target = ROOT / "dataset" / "smoke" / "smoke.csv"
    if not target.is_file():
        sys.path.insert(0, str(ROOT / "scripts"))
        import make_smoke_data

        cwd = os.getcwd()
        os.chdir(ROOT)
        try:
            make_smoke_data.main()
        finally:
            os.chdir(cwd)
    return target
