"""Optional durable local queue. Each job owns a lock across controller lifetimes."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from tsflab.experiments.infra.contracts import Executor
from tsflab.experiments.infra.storage import file_lock, write_json


def enqueue(root, directory, *, priority=0, validate=None, executor=None):
    if executor is not None:
        from tsflab.experiments.infra.executors import load_executor

        load_executor(executor)
    if validate is None:
        from tsflab.experiments.infra.execution import status

        validate = status

    directory = str(Path(directory).resolve())
    validate(directory)  # Reject invalid experiment references before writing a job.
    root = Path(root).resolve()
    with file_lock(root / ".queue.lock"):
        for path in root.glob("*/job.json"):
            state = json.loads(path.read_text())
            if state["directory"] == directory and state["status"] in {
                "queued",
                "running",
            }:
                if state.get("executor") != executor:
                    raise ValueError("active job already uses a different executor")
                return state
        state = {
            "schema_version": 1,
            "id": uuid.uuid4().hex,
            "directory": directory,
            "priority": priority,
            "executor": executor,
            "created_at": time.time(),
            "status": "queued",
        }
        write_json(root / state["id"] / "job.json", state)
        return state


def jobs(root):
    return sorted(
        (json.loads(p.read_text()) for p in Path(root).glob("*/job.json")),
        key=lambda j: (-j["priority"], j["created_at"]),
    )


def cancel_job(directory):
    path = Path(directory)
    try:
        with file_lock(path / ".job.lock", blocking=False):
            state = json.loads((path / "job.json").read_text())
            if state["status"] not in {"queued", "running"}:
                return {
                    "id": state["id"],
                    "cancel_requested": state["status"] == "cancelled",
                    "status": state["status"],
                }
            (path / "cancel.request").touch()
            state["status"] = "cancelled"
            write_json(path / "job.json", state)
    except BlockingIOError:
        state = json.loads((path / "job.json").read_text())
        (path / "cancel.request").touch()
    return {"id": state["id"], "cancel_requested": True}


def run_job(path, *, executor: Executor | None = None):
    """Run one queued item with an injected executor, or the standard sweep adapter."""
    path = Path(path)
    with file_lock(path / ".job.lock", blocking=False):
        return _run_claimed_job(path, executor=executor)


def _run_claimed_job(path, *, executor=None):
    from tsflab.experiments.infra.contracts import FileCancellation

    state = json.loads((path / "job.json").read_text())
    if state["status"] not in {"queued", "running"}:
        return state
    state.update(status="running", pid=os.getpid(), started_at=time.time())
    write_json(path / "job.json", state)
    try:
        from tsflab.experiments.infra.results import validate_execution_result

        if executor is None:
            from tsflab.experiments.infra.executors import load_executor

            executor = load_executor(
                state.get("executor") or "tsflab.experiments.infra.execution:execute"
            )
        deadline = time.monotonic() + 15
        while True:
            try:
                result = executor(
                    state["directory"],
                    cancelled=FileCancellation(path / "cancel.request"),
                )
                result = validate_execution_result(result)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.2)
        state.update(
            status="cancelled"
            if FileCancellation(path / "cancel.request")()
            else "succeeded"
            if result["ok"]
            else "failed",
            result=dict(result),
        )
    except Exception as exc:
        state.update(
            status="failed", error={"type": type(exc).__name__, "message": str(exc)}
        )
    finally:
        state["finished_at"] = time.time()
        write_json(path / "job.json", state)
    return state


def _claimable_jobs(root):
    """Unfinished jobs nobody holds, and the number of jobs a live controller holds."""
    active = 0
    available = []
    for state in jobs(root):
        if state["status"] not in {"queued", "running"}:
            continue
        path = root / state["id"]
        try:
            with file_lock(path / ".job.lock", blocking=False):
                available.append(path)
        except BlockingIOError:
            active += 1
    return available, active


def _spawn_controller(path):
    """Start a detached controller for the job at ``path`` that inherits its held lock."""
    # Reserve ownership in the parent before spawn, transfer the
    # inherited descriptor to eliminate the startup admission race.
    import fcntl

    stream = (path / ".job.lock").open("a+")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (path / "controller.log").open("a") as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "tsflab.experiments.infra.queue",
                    str(path),
                    str(stream.fileno()),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(stream.fileno(),),
            )
        return child
    finally:
        stream.close()


def work(root, *, once=False, slots=1):
    if slots < 1:
        raise ValueError("slots must be positive")
    root = Path(root).resolve()
    children = []
    # Restarting this worker adopts live jobs through their locks; it never
    # terminates the detached controllers merely because the worker exits.
    with file_lock(root / ".worker.lock", blocking=False):
        while True:
            children = [p for p in children if p.poll() is None]
            available, active = _claimable_jobs(root)
            for path in available[: max(0, slots - active)]:
                children.append(_spawn_controller(path))
            if once:
                return jobs(root)
            time.sleep(1)


if __name__ == "__main__":
    # The parent transfers a held descriptor, so no module monkeypatching or
    # second acquisition is necessary in this detached process.
    inherited = int(sys.argv[2])
    try:
        _run_claimed_job(Path(sys.argv[1]))
    finally:
        os.close(inherited)
