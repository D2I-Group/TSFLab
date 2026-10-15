"""Idempotent admission and settlement for metered external operations.

Amounts use caller-provided USD prices; no network or implicit provider calls.
Reservations must cover the maximum charge before issuing an external request.
"""

import json
import math
from pathlib import Path

from tsflab.experiments.infra.storage import file_lock, write_json


def _check_usage(tokens, cost_usd) -> None:
    if tokens < 0 or int(tokens) != tokens or not math.isfinite(cost_usd) or cost_usd < 0:
        raise ValueError("usage must contain nonnegative integral tokens and finite USD")


def _merge_limits(state, budget) -> dict:
    """Tighten the persisted limits to ``budget``; limits only ever decrease."""
    limits = state.setdefault("limits", {"tokens": None, "cost_usd": None})
    for key, requested in [
        ("tokens", getattr(budget, "max_tokens", None)),
        ("cost_usd", getattr(budget, "max_cost_usd", None)),
    ]:
        if requested is not None:
            limits[key] = requested if limits[key] is None else min(limits[key], requested)
    return limits


def _check_reservation(entries, operation, prior, item, limits) -> None:
    if prior and prior != item:
        raise ValueError("operation ID already has a different reservation or settlement")
    totals = {k: sum(v[k] for key, v in entries.items() if key != operation) + item[k] for k in ("tokens", "cost_usd")}
    for key, limit in limits.items():
        if limit is not None and totals[key] > limit:
            raise ValueError(f"{key} budget exhausted")


def _check_settlement(prior, item) -> None:
    if not prior:
        raise ValueError("reserve before invoking the external operation")
    if prior["status"] == "settled" and prior != item:
        raise ValueError("settlement is immutable")
    if any(item[k] > prior[k] for k in ("tokens", "cost_usd")):
        raise ValueError("actual usage exceeds reservation; reserve a safe upper bound")


def _record(entries, action, operation, tokens, cost_usd, limits) -> None:
    """Check one reservation or settlement against the ledger and enter it."""
    if not operation:
        raise ValueError("an idempotency operation ID is required")
    prior = entries.get(operation)
    item = {
        "tokens": tokens,
        "cost_usd": cost_usd,
        "status": "reserved" if action == "reserve" else "settled",
    }
    if action == "reserve":
        _check_reservation(entries, operation, prior, item, limits)
    elif action == "settle":
        _check_settlement(prior, item)
    else:
        raise ValueError("action must be status, reserve, or settle")
    entries[operation] = item


def account(directory, action, operation=None, *, tokens=0, cost_usd=0.0, budget=None):
    directory = Path(directory)
    _check_usage(tokens, cost_usd)
    with file_lock(directory / ".accounting.lock"):
        path = directory / "accounting.json"
        state = json.loads(path.read_text()) if path.exists() else {"schema_version": 1, "operations": {}}
        limits = _merge_limits(state, budget)
        entries = state["operations"]
        if action != "status":
            _record(entries, action, operation, tokens, cost_usd, limits)
            write_json(path, state)
        return {
            **state,
            "totals": {k: sum(v[k] for v in entries.values()) for k in ("tokens", "cost_usd")},
        }


class UsageLedger:
    """A standalone spending scope, with no experiment, queue, or tracker required."""

    def __init__(self, directory, budget=None):
        self.directory = Path(directory)
        self.budget = budget

    def reserve(self, operation, *, tokens=0, cost_usd=0.0):
        return account(
            self.directory,
            "reserve",
            operation,
            tokens=tokens,
            cost_usd=cost_usd,
            budget=self.budget,
        )

    def settle(self, operation, *, tokens=0, cost_usd=0.0):
        return account(
            self.directory,
            "settle",
            operation,
            tokens=tokens,
            cost_usd=cost_usd,
            budget=self.budget,
        )

    def status(self):
        return account(self.directory, "status")
