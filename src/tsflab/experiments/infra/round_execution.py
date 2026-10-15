"""Bridge from computation attempts to the canonical research budget ledger."""

import time
from datetime import datetime


def reserve(round_id, run_id, directory, gpus, timeout_minutes, cancelled):
    from tsflab.research.rounds import ResearchRoundBusy, claim_run

    deadline = time.monotonic() + timeout_minutes * 60
    while True:
        try:
            return claim_run(
                round_id,
                {"run_id": run_id, "gpus": gpus, "directory": str(directory)},
                active=True,
            )
        except ResearchRoundBusy:
            if cancelled() or time.monotonic() >= deadline:
                raise TimeoutError("round resource wait expired")
            time.sleep(0.2)


def stop_reason(round_id):
    from tsflab.research.rounds import load_round

    state = load_round(round_id)
    if state["status"] != "running":
        return "cancelled"
    budget = state["budget"]
    now = time.time()
    live = sum((now - r["started_at"]) * r["gpus"] / 3600 for r in state.get("active_runs", {}).values())
    if (
        budget.get("max_wall_minutes")
        and now - datetime.fromisoformat(state["created_at"]).timestamp() >= budget["max_wall_minutes"] * 60
    ):
        return "timed_out"
    if budget.get("max_gpu_hours") and state.get("gpu_hours_used", 0) + live >= budget["max_gpu_hours"]:
        return "timed_out"
    return None


def settle(round_id, number, run_id, outcome):
    from tsflab.research.rounds import finish_run

    finish_run(
        round_id,
        number,
        status="passed" if outcome == "succeeded" else outcome,
        run_id=run_id,
    )
