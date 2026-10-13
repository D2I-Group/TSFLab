"""Deterministic session limits and immutable per-attempt evidence snapshots."""
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path


def limits(task, wall_hours, resumes):
    wall = float(wall_hours)
    n = int(resumes)
    if not math.isfinite(wall) or wall <= 0 or n < 0:
        raise ValueError("wall hours must be finite and positive; resumes must be nonnegative")
    gpu_hours = None
    if task["task"] == "autoresearch":
        match = re.fullmatch(r"([\d.]+)\s+GPU-hours", task["budget"], re.I)
        if not match:
            raise ValueError("search budget must be numeric GPU-hours")
        gpu_hours = float(match[1])
        if not math.isfinite(gpu_hours) or gpu_hours <= 0:
            raise ValueError("GPU-hours must be finite and positive")
        # A dedicated single GPU cannot consume more allocated GPU-hours than
        # session wall hours. This conservative cap also bounds non-TSFLab arms.
        wall = min(wall, gpu_hours)
        n = 0
    return {"wall_seconds": max(1, int(wall * 3600)), "max_resumes": n,
            "gpu_hours": gpu_hours, "gpu_budget_enforcement": "single_gpu_wall_upper_bound" if gpu_hours else None}


def snapshot(run, task, attempt, exit_code):
    from check_success import check_autoresearch, check_benchmark, check_reproduce
    run = Path(run)
    dest = run / "attempts" / str(attempt)
    dest.mkdir(parents=True, exist_ok=False)
    ws = run / "workspace"
    check = {"reproduce": check_reproduce, "benchmark": check_benchmark, "autoresearch": check_autoresearch}[task["task"]]
    try:
        success = check(task, ws)
    except Exception as exc:
        success = {"success": False, "check_error": f"{type(exc).__name__}: {exc}"}
    (dest / "success.json").write_text(json.dumps(success, indent=2))
    for name in ("results.json", "report.md", "ledger.md", "decisions.md"):
        source = ws / "results" / name
        if source.is_file():
            shutil.copy2(source, dest / name)
    reason = "completed" if success["success"] else (
        "timed_out" if exit_code == 124 else "agent_failed" if exit_code else "finished_without_success")
    outcome = {"attempt": attempt, "exit_code": exit_code, "reason": reason,
               "task_success": success["success"], "finished_at": time.time()}
    (dest / "outcome.json").write_text(json.dumps(outcome, indent=2))
    (run / "terminal.json").write_text(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    action, *args = sys.argv[1:]
    if action == "limits":
        task_path, wall, resumes = args
        print(json.dumps(limits(json.loads(Path(task_path).read_text()), wall, resumes)))
    elif action == "snapshot":
        run, task_path, attempt, exit_code = args
        snapshot(run, json.loads(Path(task_path).read_text()), int(attempt), int(exit_code))
    else:
        raise SystemExit(f"unknown action: {action}")
