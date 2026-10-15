"""Lightweight, file-based research round state for Agent and human workflows."""

from __future__ import annotations

import json
import os
import re
import secrets
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from tsflab.core.paths import working_root

ROUND_ENV = "TSFLAB_RESEARCH_ROUND"
ROUND_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
STATUSES = {"running", "blocked", "completed", "stopped"}
EVENT_KINDS = {
    "hypothesis",
    "decision",
    "observation",
    "run",
    "failure",
    "conclusion",
    "note",
}


class ResearchRoundError(ValueError):
    """Raised when a research round request is invalid or exceeds its budget."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def research_root() -> Path:
    """Return the research workspace without creating it."""
    configured = os.environ.get("TSFLAB_WORK_DIR")
    work_dir = Path(configured) if configured else working_root() / "work_dirs"
    return work_dir.expanduser().resolve() / "_research"


def _round_dir(round_id: str) -> Path:
    if not ROUND_ID.fullmatch(round_id):
        raise ResearchRoundError("round ids must use lowercase letters, digits, and hyphens")
    return research_root() / round_id


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ResearchRoundError(f"unknown research round: {path.parent.name}") from exc


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def _state_lock(round_id: str, timeout: float = 10.0) -> Iterator[None]:
    from tsflab.experiments.infra.storage import file_lock

    deadline = time.monotonic() + timeout
    while True:
        lock = file_lock(_round_dir(round_id) / ".lock", blocking=False)
        try:
            lock.__enter__()
            break
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise ResearchRoundError(f"research round {round_id!r} is busy")
            time.sleep(0.05)
    try:
        yield
    finally:
        lock.__exit__(None, None, None)


def _normalize_budget(budget, max_runs=None):
    import math

    result = dict(budget or {})
    if max_runs is not None:
        if result.get("max_runs") is not None and result["max_runs"] != max_runs:
            raise ResearchRoundError("max_runs conflicts with budget.max_runs")
        result["max_runs"] = max_runs
    integer_limits = {
        "max_runs",
        "max_iterations",
        "max_parallel_jobs",
        "max_tokens",
        "max_retries",
        "max_models",
        "max_candidates",
    }
    for name, value in result.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ResearchRoundError(f"budget {name} must be a finite number")
        minimum = 0 if name == "max_retries" else 1 if name in integer_limits else 0
        if value < minimum or (name not in integer_limits and value == 0):
            raise ResearchRoundError(f"budget {name} must be {'nonnegative' if name == 'max_retries' else 'positive'}")
        if name in integer_limits and (not isinstance(value, int)):
            raise ResearchRoundError(f"budget {name} must be an integer")
    return result


def create_round(
    *,
    task: str,
    goal: str,
    max_runs: int | None = None,
    round_id: str | None = None,
    budget: dict | None = None,
) -> dict:
    """Create one research round with a small mutable state record."""
    if not task.strip() or not goal.strip():
        raise ResearchRoundError("task and goal must be non-empty")
    budget = _normalize_budget(budget, max_runs)
    max_runs = budget.get("max_runs")
    candidate = round_id or (f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{secrets.token_hex(2)}")
    directory = _round_dir(candidate)
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ResearchRoundError(f"research round already exists: {candidate}") from exc
    now = _now()
    state = {
        "id": candidate,
        "task": task.strip(),
        "goal": goal.strip(),
        "status": "running",
        "max_runs": max_runs,
        "budget": budget or {},
        "iterations_used": 0,
        "gpu_hours_used": 0.0,
        "active_runs": {},
        "run_claims": {},
        "runs_used": 0,
        "created_at": now,
        "updated_at": now,
    }
    _write_json(directory / "round.json", state)
    (directory / "events.jsonl").touch()
    (directory / "logs").mkdir()
    add_event(candidate, "hypothesis", goal.strip())
    return state


def load_round(round_id: str) -> dict:
    """Load one round state record."""
    state = _read_json(_round_dir(round_id) / "round.json")
    state["budget"] = _normalize_budget(state.get("budget"), state.get("max_runs"))
    state["max_runs"] = state["budget"].get("max_runs")
    return state


def list_rounds() -> list[dict]:
    """List round state records, newest first."""
    records = []
    root = research_root()
    if not root.is_dir():
        return records
    for path in root.glob("*/round.json"):
        try:
            records.append(_read_json(path))
        except (ResearchRoundError, json.JSONDecodeError):
            continue
    return sorted(records, key=lambda item: item.get("created_at", ""), reverse=True)


def add_event(
    round_id: str,
    kind: str,
    text: str,
    *,
    details: dict | None = None,
) -> dict:
    """Append one concise research event."""
    if kind not in EVENT_KINDS:
        raise ResearchRoundError(f"unknown event kind {kind!r}")
    if not text.strip():
        raise ResearchRoundError("event text must be non-empty")
    load_round(round_id)
    event = {"time": _now(), "kind": kind, "text": text.strip()}
    if details:
        event["details"] = details
    path = _round_dir(round_id) / "events.jsonl"
    line = json.dumps(event, ensure_ascii=False, default=str) + "\n"
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, line.encode("utf-8"))
    finally:
        os.close(descriptor)
    return event


def read_events(round_id: str) -> list[dict]:
    """Read all well-formed events for one round."""
    path = _round_dir(round_id) / "events.jsonl"
    load_round(round_id)
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def set_status(round_id: str, status: str, message: str | None = None) -> dict:
    """Update a round lifecycle state and optionally append a conclusion."""
    if status not in STATUSES:
        raise ResearchRoundError(f"status must be one of {sorted(STATUSES)}")
    load_round(round_id)
    with _state_lock(round_id):
        state = load_round(round_id)
        state["status"] = status
        state["updated_at"] = _now()
        _write_json(_round_dir(round_id) / "round.json", state)
    if message:
        add_event(
            round_id,
            "conclusion" if status == "completed" else "note",
            message,
            details={"status": status},
        )
    return state


def claim_run(round_id: str, details: dict, *, active: bool = False) -> int:
    """Atomically reserve one run from a round's optional run budget."""
    load_round(round_id)
    with _state_lock(round_id):
        state = load_round(round_id)
        if state["status"] != "running":
            raise ResearchRoundError(f"research round {round_id!r} is {state['status']}, not running")
        budget = state.get("budget", {})
        if (
            budget.get("max_wall_minutes")
            and time.time() >= datetime.fromisoformat(state["created_at"]).timestamp() + budget["max_wall_minutes"] * 60
        ):
            raise ResearchRoundError("research round wall-time budget exhausted")
        running = state.setdefault("active_runs", {})
        # Reclaim finished/crashed local reservations only when both execution locks
        # are free. PID-only probes would be unsafe after PID reuse.
        from tsflab.experiments.infra.storage import file_lock

        for key, item in list(running.items()):
            directory = item.get("directory")
            if not directory:
                continue
            path = Path(directory)
            try:
                with file_lock(path / ".dispatch.lock", blocking=False), file_lock(path / ".run.lock", blocking=False):
                    runtime_path = path / "runtime.json"
                    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
                    end = runtime.get("heartbeat", time.time())
                    state["gpu_hours_used"] = (
                        state.get("gpu_hours_used", 0) + max(0, end - item["started_at"]) * item.get("gpus", 0) / 3600
                    )
                    del running[key]
            except BlockingIOError:
                pass
        live_hours = sum((time.time() - item["started_at"]) * item.get("gpus", 0) / 3600 for item in running.values())
        if budget.get("max_gpu_hours") and state.get("gpu_hours_used", 0) + live_hours >= budget["max_gpu_hours"]:
            raise ResearchRoundError("research round GPU-hour budget exhausted")
        if active and len(running) >= budget.get("max_parallel_jobs", 2**31):
            raise ResearchRoundBusy("research round parallel slots are occupied")
        used = int(state.get("runs_used", 0))
        limit = state.get("max_runs")
        existing = state.setdefault("run_claims", {}).get(details.get("run_id"))
        if existing is None and limit is not None and used >= int(limit):
            raise ResearchRoundError(f"research round {round_id!r} exhausted its {limit}-run budget")
        number = existing if existing is not None else used + 1
        if str(number) in running:
            raise ResearchRoundBusy("this run is already active in the round")
        if existing is None:
            state["runs_used"] = number
        if details.get("run_id"):
            state["run_claims"][details["run_id"]] = number
        if active:
            running[str(number)] = {
                "started_at": time.time(),
                "gpus": details.get("gpus", 0),
                "pid": os.getpid(),
                "directory": details.get("directory"),
            }

        state["updated_at"] = _now()
        _write_json(_round_dir(round_id) / "round.json", state)
    add_event(round_id, "run", f"Reserved run {number}", details=details)
    return number


def finish_run(
    round_id: str,
    number: int,
    *,
    status: str,
    run_id: str | None = None,
    metrics: dict | None = None,
    error: str | None = None,
) -> None:
    """Record the outcome of one claimed run without changing round status."""
    with _state_lock(round_id):
        state = load_round(round_id)
        reservation = state.get("active_runs", {}).pop(str(number), None)
        if reservation:
            state["gpu_hours_used"] = (
                state.get("gpu_hours_used", 0) + (time.time() - reservation["started_at"]) * reservation["gpus"] / 3600
            )
            _write_json(_round_dir(round_id) / "round.json", state)
    details = {"number": number, "status": status}
    if run_id:
        details["run_id"] = run_id
    if metrics:
        details["metrics"] = metrics
    if error:
        details["error"] = error
    kind = "run" if status == "passed" else "failure"
    text = f"Run {number} {status}" + (f": {run_id}" if run_id else "")
    add_event(round_id, kind, text, details=details)


def write_prompt(round_id: str, text: str) -> Path:
    """Store the rendered task prompt beside a round for direct Agent pickup."""
    load_round(round_id)
    path = _round_dir(round_id) / "prompt.txt"
    path.write_text(text, encoding="utf-8")
    return path


def write_log(round_id: str, name: str, text: str) -> Path:
    """Persist complete command output under a traversal-safe log name."""
    load_round(round_id)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "run"
    path = _round_dir(round_id) / "logs" / f"{safe_name}.log"
    if path.exists():
        path = path.with_name(f"{safe_name}-{secrets.token_hex(2)}.log")
    path.write_text(text, encoding="utf-8")
    return path


def events_for_run(run_id: str) -> list[dict]:
    """Return complete rounds containing one completed run."""
    matches = []
    for state in list_rounds():
        events = read_events(state["id"])
        if any((event.get("details") or {}).get("run_id") == run_id for event in events):
            matches.extend({"round": state["id"], **event} for event in events)
    return matches


class ResearchRoundBusy(ResearchRoundError):
    """The round is valid but its concurrent execution capacity is occupied."""


def claim_iteration(round_id: str, *, operation: str | None = None) -> dict:
    """Start a declared research iteration without dispatching an Agent."""
    with _state_lock(round_id):
        state = load_round(round_id)
        if state["status"] != "running":
            raise ResearchRoundError("round is not running")
        claims = state.setdefault("iteration_claims", {})
        if operation is not None and operation in claims:
            return state
        used = state.get("iterations_used", 0)
        limit = state.get("budget", {}).get("max_iterations")
        if limit is not None and used >= limit:
            raise ResearchRoundError("iteration budget exhausted")
        state["iterations_used"] = used + 1
        if operation is not None:
            claims[operation] = used + 1
        _write_json(_round_dir(round_id) / "round.json", state)
    add_event(round_id, "decision", f"Started iteration {used + 1}")
    return state
