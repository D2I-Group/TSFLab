"""Harness-neutral loading, validation, and rendering of Agent task templates."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from string import Formatter
from typing import Any

from tsflab.core.paths import repository_root


def _assets_root() -> Path:
    """The `tsf init` project owning the working directory, else the repository/wheel root."""
    from tsflab.agent.modules import find_project

    found = find_project()
    if found is not None and (found[0] / ".agents" / "tasks").is_dir():
        return found[0]
    return repository_root()


ROOT = _assets_root()
TASKS = ROOT / ".agents" / "tasks"
SKILLS = ROOT / ".agents" / "skills"
REQUIRED_FIELDS = {
    "name",
    "title",
    "summary",
    "skills",
    "prompt",
    "inputs",
    "permissions",
    "budget",
    "acceptance",
}


class AgentTaskError(ValueError):
    """Raised when a task template or supplied input violates the contract."""


def _path(name: str) -> Path:
    if not name or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in name):
        raise AgentTaskError("task names must use lowercase kebab-case")
    path = TASKS / f"{name}.toml"
    if not path.is_file():
        raise AgentTaskError(f"unknown Agent task: {name}")
    return path


def load_task(name: str) -> dict[str, Any]:
    """Load one task and reject invalid task data before it reaches an Agent."""
    path = _path(name)
    task = tomllib.loads(path.read_text(encoding="utf-8"))
    errors = validate_task(task, path)
    if errors:
        raise AgentTaskError("; ".join(errors))
    return task


def _text_problems(label: str, task: dict[str, Any]) -> list[str]:
    return [
        f"{label}: {field} must be non-empty text"
        for field in ("title", "summary", "prompt")
        if not isinstance(task[field], str) or not task[field].strip()
    ]


def _table_problems(label: str, task: dict[str, Any]) -> list[str]:
    return [
        f"{label}: {field} must be a non-empty table"
        for field in ("permissions", "budget")
        if not isinstance(task[field], dict) or not task[field]
    ]


def _placeholder_problems(label: str, task: dict[str, Any], inputs: dict[str, Any]) -> list[str]:
    placeholders = {field_name for _, field_name, _, _ in Formatter().parse(str(task["prompt"])) if field_name}
    unknown = placeholders - inputs.keys()
    if unknown:
        return [f"{label}: undeclared prompt inputs {', '.join(sorted(unknown))}"]
    return []


def _skill_problems(label: str, skills: Any) -> list[str]:
    if not isinstance(skills, list) or not skills:
        return [f"{label}: skills must be a non-empty list"]
    return [
        f"{label}: unknown skill {skill!r}"
        for skill in skills
        if not isinstance(skill, str) or not (SKILLS / skill / "SKILL.md").is_file()
    ]


def _input_problems(label: str, key: str, spec: Any, task: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(spec, dict) or not isinstance(spec.get("description"), str):
        errors.append(f"{label}: input {key!r} needs a description")
    if not isinstance(spec, dict):
        return errors
    if not isinstance(spec.get("required", False), bool):
        errors.append(f"{label}: input {key!r} required must be boolean")
    if "maximum" in spec and not isinstance(spec["maximum"], int):
        errors.append(f"{label}: input {key!r} maximum must be an integer")
    if "budget_key" in spec:
        errors += _budget_key_problems(label, key, spec, task)
    return errors


def _budget_key_problems(label: str, key: str, spec: dict[str, Any], task: dict[str, Any]) -> list[str]:
    budget_key = spec["budget_key"]
    if not isinstance(budget_key, str) or budget_key not in task.get("budget", {}):
        return [f"{label}: input {key!r} budget_key must name a budget field"]
    if "maximum" not in spec:
        return [f"{label}: input {key!r} budget_key requires maximum"]
    return []


def _budget_problems(label: str, budget: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(budget, dict):
        return errors
    for key, value in budget.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            errors.append(f"{label}: budget {key!r} must be a nonnegative number")
        elif key in {"max_parallel_jobs", "max_parallel_workers", "max_wall_minutes"} and value == 0:
            errors.append(f"{label}: budget {key!r} must be positive")
    return errors


def _acceptance_problems(label: str, acceptance: Any) -> list[str]:
    if (
        not isinstance(acceptance, list)
        or not acceptance
        or not all(isinstance(item, str) and item.strip() for item in acceptance)
    ):
        return [f"{label}: acceptance must be a non-empty string list"]
    return []


def validate_task(task: dict[str, Any], path: Path) -> list[str]:
    """Return deterministic schema and cross-reference violations."""
    label = str(path.relative_to(ROOT))
    errors: list[str] = []
    missing = REQUIRED_FIELDS - task.keys()
    if missing:
        errors.append(f"{label}: missing {', '.join(sorted(missing))}")
        return errors
    if task["name"] != path.stem:
        errors.append(f"{label}: name must match the filename")
    errors.extend(_text_problems(label, task))
    errors.extend(_skill_problems(label, task["skills"]))
    inputs = task["inputs"]
    if not isinstance(inputs, dict):
        errors.append(f"{label}: inputs must be a table")
        inputs = {}
    for key, spec in inputs.items():
        errors.extend(_input_problems(label, key, spec, task))
    errors.extend(_placeholder_problems(label, task, inputs))
    errors.extend(_table_problems(label, task))
    errors.extend(_budget_problems(label, task.get("budget", {})))
    errors.extend(_acceptance_problems(label, task["acceptance"]))
    return errors


def audit_tasks() -> list[str]:
    """Validate every canonical task without rendering or executing it."""
    errors: list[str] = []
    if not TASKS.is_dir():
        return [".agents/tasks is missing"]
    for path in sorted(TASKS.glob("*.toml")):
        try:
            task = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
            continue
        errors.extend(validate_task(task, path))
    if not list(TASKS.glob("*.toml")):
        errors.append(".agents/tasks contains no task templates")
    return errors


def list_tasks() -> list[dict[str, Any]]:
    """Return the stable, lightweight task catalog."""
    records = []
    for path in sorted(TASKS.glob("*.toml")):
        task = load_task(path.stem)
        records.append(
            {
                "name": task["name"],
                "title": task["title"],
                "summary": task["summary"],
                "skills": task["skills"],
            }
        )
    return records


def _check_maximum(key: str, spec: dict[str, Any], value: Any) -> None:
    if value is None or "maximum" not in spec:
        return
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise AgentTaskError(f"input {key!r} must be an integer") from exc
    if numeric < 1 or numeric > spec["maximum"]:
        raise AgentTaskError(f"input {key!r} must be between 1 and {spec['maximum']}")


def _bind_inputs(specs: dict[str, Any], supplied: dict[str, str]) -> dict[str, str]:
    unknown = supplied.keys() - specs.keys()
    if unknown:
        raise AgentTaskError(f"unknown input(s): {', '.join(sorted(unknown))}")
    values: dict[str, str] = {}
    missing = []
    for key, spec in specs.items():
        value = supplied.get(key, spec.get("default"))
        if value is None and spec.get("required", False):
            missing.append(key)
        _check_maximum(key, spec, value)
        values[key] = "" if value is None else str(value)
    if missing:
        raise AgentTaskError(f"missing required input(s): {', '.join(sorted(missing))}")
    return values


def render_task(name: str, supplied: dict[str, str]) -> dict[str, Any]:
    """Bind user inputs and return a self-contained Agent prompt plus limits."""
    task = load_task(name)
    specs = task["inputs"]
    values = _bind_inputs(specs, supplied)
    prompt = task["prompt"].format_map(values).strip()
    budget = dict(task["budget"])
    for key, spec in specs.items():
        budget_key = spec.get("budget_key")
        if budget_key:
            budget[budget_key] = int(values[key])
    return {
        "task": task["name"],
        "title": task["title"],
        "skills": task["skills"],
        "inputs": values,
        "permissions": task["permissions"],
        "budget": budget,
        "acceptance": task["acceptance"],
        "prompt": prompt,
    }


def render_text(payload: dict[str, Any]) -> str:
    """Render a provider-neutral prompt while preserving machine-readable limits."""
    boundaries = json.dumps(
        {"permissions": payload["permissions"], "budget": payload["budget"]},
        ensure_ascii=False,
        sort_keys=True,
    )
    acceptance = "\n".join(f"- {item}" for item in payload["acceptance"])
    skills = ", ".join(payload["skills"])
    return (
        f"Task: {payload['title']} ({payload['task']})\n"
        f"Required skills: {skills}\n"
        "Execution: continue in the current Agent; use native reasoning, editing, "
        "and presentation for cognitive work. Call library APIs for validated "
        "computation, persistent evidence, budgets, locks, and recovery. CLI examples "
        "are optional adapters. This template does not authorize dispatch.\n"
        f"Boundaries: {boundaries}\n\n"
        f"{payload['prompt']}\n\nAcceptance criteria:\n{acceptance}\n"
    )
