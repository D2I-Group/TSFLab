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

    errors: list[str] = []
    warnings: list[str] = []
    models = {str(r["name"]) for r in model_records(root)}
    slots_known = load_rules().get("slot", {})
    name = spec.get("name", "")
    parents = _check_identity(spec, name, models, errors)

    slots = spec.get("slots", {})
    found = _check_slots(slots, slots_known, models, errors)
    components, losses, donors = found["components"], found["losses"], found["donors"]
    if len(losses) > 1:
        errors.append(f"choose one training loss, got {losses}")

    free = list(spec.get("free_form", []))
    _check_free_form(free, slots_known, errors)
    if "temporal" not in slots and not any(b.get("slot") == "temporal" for b in free):
        errors.append("fill the temporal slot with a component or a free-form block")
    if not components and not free:
        errors.append("a composition needs at least one component or free-form block")
    _warn(free, parents, donors, warnings)

    executable = _executable(slots, free)
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
        "interfaces": found["interfaces"],
        "executable": executable,
        "scaffold": _scaffold(name, components),
        "card_provenance": _card_provenance(spec, components, parents, donors),
    }


def _warn(free: list[Any], parents: list[Any], donors: list[str], warnings: list[str]) -> None:
    if not (parents or donors):
        warnings.append("no parent or donor model named; record what the design is compared against")
    if free:
        warnings.append("free-form blocks stay model-local; extract a component only after two consumers match exactly")


def _card_provenance(spec: dict[str, Any], components: list[str], parents: list[Any], donors: list[str]) -> str:
    return (
        "Composition of components: "
        + (", ".join(components) or "none")
        + "; borrowed from: "
        + (", ".join(sorted(set(parents + donors))) or "none")
        + f". Hypothesis: {spec.get('hypothesis', '')}"
    )


def _scaffold(name: Any, components: list[str]) -> str:
    components_flag = ",".join(components) if components else "none"
    return (
        f'tsf model scaffold {name} --paper-title "{name} (recombination)" '
        f"--paper-url <research-report-url> --venue unpublished --year <year> "
        f"--components {components_flag}"
    )


def _check_identity(spec: dict[str, Any], name: Any, models: set[str], errors: list[str]) -> list[Any]:
    """Check name, summary, hypothesis, and parents; return the parents."""
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
    return parents


def _check_slots(slots: dict[str, Any], slots_known: dict[str, Any], models: set[str], errors: list[str]) -> dict:
    """Check every slot option; collect components, losses, donors, and component interfaces."""
    found: dict[str, Any] = {"components": [], "losses": [], "donors": [], "interfaces": {}}
    for slot, options in slots.items():
        if slot not in slots_known:
            errors.append(f"unknown slot {slot!r}; choose from {sorted(slots_known)}")
            continue
        for option in options:
            _check_option(slot, option, models, found, errors)
    return found


def _check_option(slot: str, option: Any, models: set[str], found: dict[str, Any], errors: list[str]) -> None:
    kind, _, value = str(option).partition(":")
    if kind == "component":
        _check_component(slot, value, found, errors)
    elif kind == "model":
        if value not in models:
            errors.append(f"slot {slot}: unknown donor model {value!r}")
        elif value not in found["donors"]:
            found["donors"].append(value)
    elif kind == "local" or str(option) == "none":
        return  # local strategy / empty slot; executability is checked by the slot registry
    elif kind == "loss":
        _check_loss(slot, value, found["losses"], errors)
    else:
        errors.append(f"slot {slot}: option {option!r} needs a component:/model:/local:/loss: prefix or 'none'")


def _check_loss(slot: str, value: str, losses: list[str], errors: list[str]) -> None:
    from tsflab.catalog.registry.losses import LOSS_NAME_MAP

    if value not in LOSS_NAME_MAP:
        errors.append(f"slot {slot}: unknown loss {value!r}; known {sorted(LOSS_NAME_MAP)}")
    elif value not in losses:
        losses.append(value)


def _check_component(slot: str, value: str, found: dict[str, Any], errors: list[str]) -> None:
    from tsflab.catalog.components import COMPONENT_CATALOG

    if value not in COMPONENT_CATALOG.names():
        errors.append(f"slot {slot}: unknown component {value!r}")
        return
    if value not in found["components"]:
        found["components"].append(value)
    meta = COMPONENT_CATALOG.get(value)
    try:
        module = importlib.import_module(meta.module)
        missing = [s for s in meta.public_symbols if not hasattr(module, s)]
    except Exception as exc:  # import failures are spec errors, not crashes
        errors.append(f"component {value!r} failed to import: {exc}")
        return
    if missing:
        errors.append(f"component {value!r} lacks public symbols {missing}")
    found["interfaces"][value] = list(meta.public_symbols)


def _check_free_form(free: list[Any], slots_known: dict[str, Any], errors: list[str]) -> None:
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


def _executable(slots: dict[str, Any], free: list[Any]) -> dict[str, Any]:
    """Resolve the slots against the executable slot registry."""
    from tsflab.models._slots.registry import assignment_components, check_assignment, resolve_assignment

    assignment, exec_errors = resolve_assignment(slots)
    if not exec_errors:
        exec_errors = check_assignment(assignment)
    if free:
        exec_errors.append("free-form blocks are not executable; implement them with add-model first")
    if exec_errors:
        return {"ok": False, "errors": exec_errors, "assignment": {}, "components": [], "output": None}
    return {
        "ok": True,
        "errors": exec_errors,
        "assignment": assignment,
        "components": list(assignment_components(assignment)),
        "output": _head_output(assignment),
    }


def _head_output(assignment: dict[str, str]) -> str:
    from tsflab.models._slots.registry import OPTIONS

    return OPTIONS["head"][assignment["head"]].output


def compose_command(args: list[str], root: Path) -> int:
    """CLI body for ``tsf model compose``."""
    import json
    import sys

    parsed = _compose_parser().parse_args(args)
    try:
        spec = tomllib.loads(Path(parsed.spec).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        print(f"cannot read spec: {exc}", file=sys.stderr)
        return 2
    result = validate_composition(spec, root)
    written: dict[str, object] = {}
    wants_build = bool(parsed.write_config or parsed.register)
    status = _validation_status(result, wants_build)
    if wants_build and status == 0:
        status = _build(spec, result, parsed, root, written)
    result["written"] = written
    if parsed.json:
        print(json.dumps(result, indent=2))
        return status
    _print_result(result, wants_build)
    for key, value in written.items():
        print(f"{key}: {value}")
    return status


def _validation_status(result: dict[str, Any], wants_build: bool) -> int:
    if not result["ok"]:
        return 1
    return 1 if wants_build and not result["executable"]["ok"] else 0


def _compose_parser() -> Any:
    import argparse

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
    return parser


def _build(spec: dict[str, Any], result: dict[str, Any], parsed: Any, root: Path, written: dict[str, object]) -> int:
    """Write the requested config and/or model package into ``written``; return the exit status."""
    from tsflab.research import composition_build as build

    target = Path(parsed.root).resolve() if parsed.root else root
    try:
        if parsed.write_config:
            written["config"] = str(build.write_config(spec, result, parsed, target))
        if parsed.register:
            written["register"] = build.register_model(spec, result, parsed.register, target, dry_run=parsed.dry_run)
    except ValueError as exc:
        result["errors"].append(str(exc))
        return 1
    return 0


def _print_result(result: dict[str, Any], wants_build: bool) -> None:
    for error in result["errors"]:
        print(f"ERROR: {error}")
    for warning in result["warnings"]:
        print(f"WARNING: {warning}")
    if not result["ok"]:
        return
    executable = result["executable"]
    print(f"OK: {result['name']} = {', '.join(result['components']) or 'free-form only'}")
    print(f"scaffold: {result['scaffold']}")
    print(f"card provenance: {result['card_provenance']}")
    if executable["ok"]:
        print(f"executable: yes ({executable['output']} output); slots {executable['assignment']}")
    else:
        for reason in executable["errors"]:
            print(f"NOT EXECUTABLE: {reason}")
    if not wants_build:
        print("Validation only; add --write-config PATH to make it runnable or --register NAME to scaffold a model.")
