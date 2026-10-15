"""One vocabulary of data and task characteristics shared by every card.

Datasets declare the characteristics they have (`characteristics` in the dataset
card, measured by `tsf data analyze <preset> --write-card`); models and
components declare the characteristics they are designed for (`fits` in their
cards). `tsf catalog match <preset>` intersects the two so research starts from
what the data needs rather than from names.

Three kinds of terms:

* ``data`` terms are measurable. Each is one rule of
  ``src/tsflab/data/profile_rules.toml`` (metric, comparison, threshold, finding);
  that file stays the single source of thresholds.
* ``task`` terms describe the forecasting setup and come from the dataset preset
  and loader (graph inputs, timestamps, covariates, horizon, output type).
* ``any`` marks a generic building block with no data-specific benefit.
"""

from __future__ import annotations

from functools import lru_cache

TASK_TERMS: dict[str, str] = {
    "spatial-graph": "nodes with a known or learnable graph (traffic, air quality, grids)",
    "calendar-effects": "timestamps carry signal (hour of day, weekday, holidays)",
    "exogenous-covariates": "known external inputs besides the target",
    "long-lookback": "long input windows (hundreds of steps or more)",
    "long-horizon": "long forecast horizons (hundreds of steps)",
    "probabilistic-output": "the task needs quantiles, samples, or a distribution",
    "low-data": "few training windows; prior knowledge or pretraining helps",
    "efficiency": "tight compute or memory budgets",
}
GENERIC = {"any": "a generic building block with no data-specific benefit"}


@lru_cache(maxsize=1)
def data_terms() -> dict[str, dict[str, object]]:
    """Measurable terms from ``profile_rules.toml``: id -> metric, op, value, finding."""
    from tsflab.data.profile import load_rules

    return {
        str(rule["id"]): {
            "metric": rule["metric"],
            "op": rule["op"],
            "value": rule["value"],
            "finding": rule.get("finding", ""),
        }
        for rule in load_rules().get("rule", [])
    }


def vocabulary() -> dict[str, str]:
    """Every allowed term with a one-line meaning, data terms first."""
    terms = {name: str(spec["finding"]) for name, spec in data_terms().items()}
    terms.update(TASK_TERMS)
    terms.update(GENERIC)
    return terms


def kind_of(term: str) -> str | None:
    if term in data_terms():
        return "data"
    if term in TASK_TERMS:
        return "task"
    if term in GENERIC:
        return "generic"
    return None


def _any_problem(label: str, field: str, allow_any: bool) -> str:
    return f"{label}: {field} 'any' must stand alone" if allow_any else f"{label}: {field} cannot be 'any'"


def term_problems(label: str, field: str, terms: object, *, allow_any: bool) -> list[str]:
    """Validate a card's list of characteristic terms against the vocabulary."""
    if not isinstance(terms, list) or not terms or not all(isinstance(t, str) for t in terms):
        return [f"{label}: {field} must be a non-empty list of characteristic terms"]
    problems = []
    unknown = sorted(set(terms) - set(vocabulary()))
    if unknown:
        problems.append(
            f"{label}: {field} terms {unknown} are not in the vocabulary "
            "(tsflab.catalog.characteristics; data terms come from profile_rules.toml)"
        )
    if "any" in terms and (not allow_any or len(terms) > 1):
        problems.append(_any_problem(label, field, allow_any))
    if len(set(terms)) != len(terms):
        problems.append(f"{label}: {field} has duplicate terms")
    return problems
