"""Admission: run a model's executable contract once and record it in its card.

The contract (``tsflab.catalog.model_contracts``, strict mode) constructs the
model from its preset, runs forward and backward (or one synthetic training step
through a declared ``training_objective``), checks shapes, finiteness, gradients,
and a ``state_dict`` round trip. The result is written to ``[admission]`` in the
model's ``card.toml``. There is no fingerprint and no staleness tracking: a model
is re-checked only when it, its preset, or a component it uses changes
(``tsf model verify --changed``), and a release requires every admission to be
``passed``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

from tsflab.catalog.cards.toml_io import dumps

INLINE = ("data_params",)
#: Wall-clock limit for one isolated contract (seconds).
TIMEOUT = 1800


def _contract(name: str) -> dict[str, str] | None:
    from tsflab.catalog.model_contracts import audit_model_contracts

    failures = audit_model_contracts([name], strict=True)
    if not failures:
        return None
    return {"stage": failures[0].stage, "error": failures[0].error}


def _isolated(name: str, timeout: int) -> dict[str, str] | None:
    """Run one contract in a fresh interpreter, so memory and crashes stay per model."""
    try:
        out = subprocess.run(
            [sys.executable, "-m", "tsflab.catalog.admission", name],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=os.environ.copy(),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"stage": "process", "error": f"timed out after {timeout}s"}
    lines = out.stdout.strip().splitlines()
    if out.returncode == 0 and lines:
        return json.loads(lines[-1])
    if out.returncode < 0:
        return {"stage": "process", "error": f"killed by signal {-out.returncode} (out of memory?)"}
    tail = (out.stderr.strip().splitlines() or ["no output"])[-1]
    return {"stage": "process", "error": f"exit {out.returncode}: {tail}"}


def run_contracts(
    names: list[str],
    jobs: int = 1,
    *,
    isolated: bool = False,
    timeout: int = TIMEOUT,
    progress: Callable[[str, dict[str, str] | None], None] | None = None,
) -> dict[str, dict[str, str] | None]:
    """Run strict contracts; ``None`` means passed.

    ``isolated`` runs each model in its own interpreter (``jobs`` at a time): a crash,
    an out-of-memory kill, or a timeout becomes that model's failure instead of
    ending the run. ``progress`` is called once per model as results arrive.
    """
    results: dict[str, dict[str, str] | None] = {}
    if not isolated:
        for name in names:
            results[name] = _contract(name)
            if progress:
                progress(name, results[name])
        return results
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_isolated, name, timeout): name for name in names}
        for future in as_completed(futures):
            name = futures[future]
            results[name] = future.result()
            if progress:
                progress(name, results[name])
    return {name: results[name] for name in names}


def _commit(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"], cwd=root, capture_output=True, text=True, check=False
        )
    except OSError:
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


#: Text ``tsf model scaffold`` leaves for the implementer; none may remain once a model
#: is verified or added (a scaffold is discoverable as soon as its ``spec.py`` exists).
SCAFFOLD_MARKERS = ("SCAFFOLD", "PLACEHOLDER", "TODO")


def unfinished_files(package: Path) -> list[Path]:
    """Model files that still carry scaffold markers (empty for a finished package)."""
    return [
        path
        for path in (package / "model.py", package / "README.md", package / "card.toml")
        if path.is_file() and any(marker in path.read_text(encoding="utf-8") for marker in SCAFFOLD_MARKERS)
    ]


def card_path(root: Path, name: str) -> Path:
    from tsflab.catalog.registry.models import MODEL_CATALOG

    module = MODEL_CATALOG.refs()[name]  # e.g. tsflab.models.patchtst.spec
    return root / "src" / Path(*module.split(".")[:-1]) / "card.toml"


def write_admission(
    root: Path, name: str, failure: dict[str, str] | None, *, note: str = "", reference: str | None = None
) -> dict[str, str]:
    """Record one admission result in the model's ``card.toml``; return the record."""
    path = card_path(root, name)
    facts = tomllib.loads(path.read_text(encoding="utf-8"))
    previous = facts.get("admission") or {}
    record = {
        "status": "passed" if failure is None else "failed",
        "date": date.today().isoformat(),
        "commit": _commit(root),
        "device": "cpu",
        "reference": reference or previous.get("reference", "none"),
        "note": note or (f"{failure['stage']}: {failure['error']}"[:300] if failure else ""),
    }
    facts["admission"] = record
    path.write_text(dumps(facts, inline=INLINE), encoding="utf-8")
    return record


def changed_models(root: Path, base: str) -> list[str]:
    """Models whose package, preset, or used component changed since ``base`` (git diff paths)."""
    from tsflab.catalog.cards.metadata import model_records
    from tsflab.catalog.component_audit import component_dependency_closure

    diff = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    paths = [line for line in diff.stdout.split() if line]
    docs = {"README.md", "card.toml", "reference.md"}
    touched_packages, touched_components, touched_presets = set(), set(), set()
    for path in paths:
        parts = Path(path).parts
        if len(parts) >= 4 and parts[:3] == ("src", "tsflab", "models") and parts[-1] not in docs:
            if parts[3] == "_components" and len(parts) >= 5:
                touched_components.add(parts[4])
            elif not parts[3].startswith("_"):
                touched_packages.add(parts[3])
        elif len(parts) == 3 and parts[:2] == ("configs", "models"):
            touched_presets.add(Path(parts[2]).stem)
    names = []
    for record in model_records(root):
        used = set(component_dependency_closure({str(c) for c in record.get("components", ())}))
        if record["package"] in touched_packages or record["name"] in touched_presets or used & touched_components:
            names.append(str(record["name"]))
    return sorted(names)


if __name__ == "__main__":  # worker for run_contracts(isolated=True)
    print(json.dumps(_contract(sys.argv[1])))
