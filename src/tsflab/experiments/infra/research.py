"""Optional evidence and budget support for the current Agent; never an Agent loop."""

from tsflab.agent.tasks import load_task, render_task, render_text
from tsflab.research.rounds import (
    add_event,
    claim_iteration,
    create_round,
    load_round,
    read_events,
)
from tsflab.research.rounds import (
    set_status as set_round_status,
)

__all__ = [
    "add_event",
    "claim_iteration",
    "create_round",
    "load_round",
    "read_events",
    "set_round_status",
    "load_task",
    "prepare_task",
]


def prepare_task(name, supplied=None, *, persist=False):
    """Optionally bind a template; persistence and dispatch are separate decisions.

    With persist=False this only returns data. The current Agent owns planning,
    interpretation and continuation. A template is not required to use services.
    """
    task = render_task(name, supplied if supplied is not None else {})
    result = {
        "task": task,
        "round": None,
        "prompt_path": None,
        "dispatch": "not-performed",
    }
    if persist:
        from tsflab.research.rounds import write_prompt

        state = create_round(
            task=task["task"],
            goal=task["prompt"],
            max_runs=task["budget"].get("max_runs"),
            budget=task["budget"],
        )
        result["round"] = state
        result["prompt_path"] = str(write_prompt(state["id"], render_text(task)))
    return result
