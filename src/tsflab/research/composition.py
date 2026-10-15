"""Validate a recombination spec against the component, model, and loss catalogs.

``tsf model compose SPEC.toml`` validates a recombination spec and writes nothing
by default. It checks that every slot option names a real catalog entry, that component
public symbols import, and that any free-form block stays inside its line budget; it
also resolves the slots against the executable slot registry
(``tsflab.models._slots``) and reports whether the assignment can run. Flags:

- ``--write-config PATH --dataset D [--enc-in N]``: a run config for the ``Composed``
  model (or for the registered model when combined with ``--register``).
- ``--register NAME [--dry-run]``: scaffold a dedicated catalog model package whose
  card records the composition as provenance; ``tsf model add NAME`` admits it.

Spec (TOML)::

    name = "SeasonalRevLinear"          # new public model name
    summary = "one sentence"
    hypothesis = "profile-tied claim being tested"
    parents = ["RLinear"]               # existing models whose idea is borrowed
    [slots]                             # slot names come from profile_rules.toml
    normalization = ["component:revin"]
    decomposition = ["component:series_decomposition"]
    temporal = ["component:channel_wise_linear"]
    channel = ["local:independent"]     # local:independent | local:individual | local:mixing
head = ["component:flatten_forecast_head"]
loss = ["loss:mae"]                 # a training.loss, not a component
[params]                            # optional adapter hyperparameters (hidden, n_layers, ...)
hidden = 32
    [[free_form]]                       # optional, bounded new block
    slot = "temporal"
    name = "PhaseGate"
    description = "what it does and why no component fits"
    max_lines = 60
"""

from __future__ import annotations

import importlib
import re
import tomllib
from pathlib import Path
from typing import Any

from tsflab.data.profile import load_rules

MAX_FREE_FORM_BLOCKS = 2
MAX_FREE_FORM_LINES = 120
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")


def validate_composition(spec: dict[str, Any], root: Path) -> dict[str, Any]:
    """Return ``{"ok", "errors", "warnings", ...}`` for a parsed composition spec."""
    from tsflab.catalog.cards.metadata import model_records
    from tsflab.catalog.components import COMPONENT_CATALOG
    from tsflab.catalog.registry.losses import LOSS_NAME_MAP

    errors: list[str] = []
    warnings: list[str] = []
    models = {str(r["name"]) for r in model_records(root)}
    slots_known = load_rules().get("slot", {})
    name = spec.get("name", "")
    if not isinstance(name, str) or not _NAME.match(name):
        errors.append("name must be an alphanumeric model name starting with a letter")
    elif name in models:
        errors.append(f"model {name!r} already exists in the catalog")
    for key in ("summary", "hypothesis"):
        if not str(spec.get(key, "")).strip():
            errors.append(f"{key} is required")
    parents = list(spec.get("parents", []))
    for parent in parents:
        if parent not in models:
            errors.append(f"parent model {parent!r} is not in the catalog")

    components: list[str] = []
    losses: list[str] = []
    donors: list[str] = []
    interfaces: dict[str, list[str]] = {}
    slots = spec.get("slots", {})
    for slot, options in slots.items():
        if slot not in slots_known:
            errors.append(f"unknown slot {slot!r}; choose from {sorted(slots_known)}")
            continue
        for option in options:
            kind, _, value = str(option).partition(":")
            if kind == "component":
                if value not in COMPONENT_CATALOG.names():
                    errors.append(f"slot {slot}: unknown component {value!r}")
                    continue
                if value not in components:
                    components.append(value)
                meta = COMPONENT_CATALOG.get(value)
                try:
                    module = importlib.import_module(meta.module)
                    missing = [s for s in meta.public_symbols if not hasattr(module, s)]
                except Exception as exc:  # import failures are spec errors, not crashes
                    errors.append(f"component {value!r} failed to import: {exc}")
                    continue
                if missing:
                    errors.append(f"component {value!r} lacks public symbols {missing}")
                interfaces[value] = list(meta.public_symbols)
            elif kind == "model":
                if value not in models:
                    errors.append(f"slot {slot}: unknown donor model {value!r}")
                elif value not in donors:
                    donors.append(value)
            elif kind == "local" or str(option) == "none":
                continue  # local strategy / empty slot; executability is checked by the slot registry
            elif kind == "loss":
                if value not in LOSS_NAME_MAP:
                    errors.append(f"slot {slot}: unknown loss {value!r}; known {sorted(LOSS_NAME_MAP)}")
                elif value not in losses:
                    losses.append(value)
            else:
                errors.append(f"slot {slot}: option {option!r} needs a component:/model:/local:/loss: prefix or 'none'")
    if len(losses) > 1:
        errors.append(f"choose one training loss, got {losses}")

    free = list(spec.get("free_form", []))
    if len(free) > MAX_FREE_FORM_BLOCKS:
        errors.append(f"at most {MAX_FREE_FORM_BLOCKS} free-form blocks per method")
    for block in free:
        label = block.get("name", "?")
        if block.get("slot") not in slots_known:
            errors.append(f"free-form block {label!r}: unknown slot {block.get('slot')!r}")
        if not str(block.get("description", "")).strip():
            errors.append(f"free-form block {label!r}: description is required")
        lines = block.get("max_lines", 0)
        if not isinstance(lines, int) or not 0 < lines <= MAX_FREE_FORM_LINES:
            errors.append(f"free-form block {label!r}: max_lines must be 1..{MAX_FREE_FORM_LINES}")
    if "temporal" not in slots and not any(b.get("slot") == "temporal" for b in free):
        errors.append("fill the temporal slot with a component or a free-form block")
    if not components and not free:
        errors.append("a composition needs at least one component or free-form block")
    if not (parents or donors):
        warnings.append("no parent or donor model named; record what the design is compared against")
    if free:
        warnings.append("free-form blocks stay model-local; extract a component only after two consumers match exactly")

    from tsflab.models._slots.registry import assignment_components, check_assignment, resolve_assignment

    assignment, exec_errors = resolve_assignment(slots)
    if not exec_errors:
        exec_errors = check_assignment(assignment)
    if free:
        exec_errors.append("free-form blocks are not executable; implement them with add-model first")
    executable = {
        "ok": not exec_errors,
        "errors": exec_errors,
        "assignment": assignment if not exec_errors else {},
        "components": list(assignment_components(assignment)) if not exec_errors else [],
        "output": _head_output(assignment) if not exec_errors else None,
    }
    components_flag = ",".join(components) if components else "none"
    scaffold = (
        f'tsf model scaffold {name} --paper-title "{name} (recombination)" '
        f"--paper-url <research-report-url> --venue unpublished --year <year> "
        f"--components {components_flag}"
    )
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "name": name,
        "components": components,
        "donor_models": donors,
        "parents": parents,
        "loss": losses[0] if losses else None,
        "free_form": [b.get("name") for b in free],
        "interfaces": interfaces,
        "executable": executable,
        "scaffold": scaffold,
        "card_provenance": (
            "Composition of components: "
            + (", ".join(components) or "none")
            + "; borrowed from: "
            + (", ".join(sorted(set(parents + donors))) or "none")
            + f". Hypothesis: {spec.get('hypothesis', '')}"
        ),
    }


def _head_output(assignment: dict[str, str]) -> str:
    from tsflab.models._slots.registry import OPTIONS

    return OPTIONS["head"][assignment["head"]].output


def compose_command(args: list[str], root: Path) -> int:
    """CLI body for ``tsf model compose``."""
    import argparse
    import json
    import sys

    from tsflab.cli.args import add_option

    parser = argparse.ArgumentParser(
        prog="tsf model compose",
        description=(
            "Validate a recombination spec. Without flags it writes nothing. "
            "--write-config emits a runnable run config for the Composed model; "
            "--register NAME scaffolds a dedicated catalog model package."
        ),
    )
    parser.add_argument("spec", help="composition spec TOML")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--write-config", metavar="PATH", help="write a run config (extends base + dataset + Composed)")
    parser.add_argument(
        "--dataset",
        help="dataset config: a TOML path (e.g. configs/fixtures/smoke.toml) or a configs/datasets preset name",
    )
    parser.add_argument("--base", default="configs/base.toml", help="base config to extend (default configs/base.toml)")
    parser.add_argument("--enc-in", type=int, help="channel count of the dataset (or [params] enc_in in the spec)")
    parser.add_argument("--seq-len", type=int)
    add_option(parser, "--horizon", "pred_len", "--pred-len", type=int, help="prediction length")
    parser.add_argument("--epochs", type=int)
    add_option(
        parser,
        "--records",
        "work_dir",
        "--work-dir",
        help="run-records root, experiment.work_dir (keeps a round's records together)",
    )
    parser.add_argument("--smoke", action="store_true", help="CPU, 1 epoch, small batch (tiny fixture runs)")
    parser.add_argument(
        "--register", metavar="NAME", help="scaffold a dedicated catalog model named NAME from the spec"
    )
    parser.add_argument("--dry-run", action="store_true", help="with --register: list the files without writing")
    parser.add_argument("--root", help="repository root to write into (default: this checkout)")
    parsed = parser.parse_args(args)
    try:
        spec = tomllib.loads(Path(parsed.spec).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        print(f"cannot read spec: {exc}", file=sys.stderr)
        return 2
    result = validate_composition(spec, root)
    written: dict[str, object] = {}
    status = 0 if result["ok"] else 1
    wants_build = bool(parsed.write_config or parsed.register)
    if wants_build and result["ok"] and not result["executable"]["ok"]:
        status = 1
    if wants_build and status == 0:
        from tsflab.research import composition_build as build

        target = Path(parsed.root).resolve() if parsed.root else root
        try:
            if parsed.write_config:
                written["config"] = str(build.write_config(spec, result, parsed, target))
            if parsed.register:
                written["register"] = build.register_model(
                    spec, result, parsed.register, target, dry_run=parsed.dry_run
                )
        except ValueError as exc:
            result["errors"].append(str(exc))
            status = 1
    result["written"] = written
    if parsed.json:
        print(json.dumps(result, indent=2))
        return status
    for error in result["errors"]:
        print(f"ERROR: {error}")
    for warning in result["warnings"]:
        print(f"WARNING: {warning}")
    executable = result["executable"]
    if result["ok"]:
        print(f"OK: {result['name']} = {', '.join(result['components']) or 'free-form only'}")
        print(f"scaffold: {result['scaffold']}")
        print(f"card provenance: {result['card_provenance']}")
        if executable["ok"]:
            print(f"executable: yes ({executable['output']} output); slots {executable['assignment']}")
        else:
            for reason in executable["errors"]:
                print(f"NOT EXECUTABLE: {reason}")
        if not wants_build:
            print(
                "Validation only; add --write-config PATH to make it runnable or --register NAME to scaffold a model."
            )
    for key, value in written.items():
        print(f"{key}: {value}")
    return status
