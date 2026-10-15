"""Upstream issues: problems found in a model's paper or official code.

Every model card carries a curated ``## Upstream issues`` section before the
canonical block. It is the structured record of what independent
reimplementation exposed, separate from ``Differences`` (which also lists
TSFLab's own scoping choices). One bullet per issue, on one line:

    - **code-bug** · `model/diffusion.py`: noise is `randint_like(x, 0, 1)`, all zeros, so nothing is diffused. Resolution: Gaussian noise per Eq. 2-6.

``<kind>`` is one of :data:`KINDS`; the location names a file (at the pinned
revision), an equation, section, table, or script; ``Resolution:`` says what
TSFLab does instead. A card with nothing to report says so explicitly:

    None found: <what was checked, e.g. Eq. 1-9 against `models/x.py`>.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

SECTION = "Upstream issues"
KINDS: dict[str, str] = {
    "code-bug": "the official code computes something other than intended, crashes, or ignores a setting",
    "paper-code-mismatch": "the paper and the official code disagree and neither is plainly wrong",
    "paper-error": "an equation, figure, or statement in the paper is wrong or inconsistent",
    "underspecified": "the paper omits a detail an implementation needs; TSFLab's choice is stated",
    "unreleased": "official code is missing, partial, or not runnable as published",
    "leakage": "the official pipeline uses information unavailable at forecast time",
    "license": "the license is absent, conflicting, or more restrictive than it appears",
}


@dataclass(frozen=True)
class Issue:
    kind: str
    where: str
    what: str
    resolution: str


def summarize(records: dict[str, list[Issue] | None]) -> dict[str, object]:
    """Aggregate per-model issues for reporting (counts by kind, coverage)."""
    documented = {name: issues for name, issues in records.items() if issues is not None}
    kinds = Counter(issue.kind for issues in documented.values() for issue in issues)
    return {
        "models": len(records),
        "documented": len(documented),
        "with_issues": sum(bool(issues) for issues in documented.values()),
        "issues": sum(kinds.values()),
        "by_kind": dict(sorted(kinds.items())),
        "missing": sorted(name for name, issues in records.items() if issues is None),
    }


# ---------------------------------------------------------------------------
# Declined papers: reviewed for the catalog but not admitted (catalog/declined.toml)
# ---------------------------------------------------------------------------

DECLINED = "catalog/declined.toml"
DECLINE_REASONS = ("out-of-scope", "unfaithful", "composite", "duplicate")
_DECLINED_FIELDS = ("name", "title", "paper", "code", "revision", "reason", "summary", "issues")


def load_declined(root) -> tuple[list[dict], list[str]]:
    """Return ``(papers, problems)`` from ``catalog/declined.toml``; missing file is empty."""
    import tomllib
    from pathlib import Path

    path = Path(root) / DECLINED
    if not path.is_file():
        return [], []
    papers = tomllib.loads(path.read_text(encoding="utf-8")).get("paper", [])
    problems: list[str] = []
    names: set[str] = set()
    for index, paper in enumerate(papers):
        label = f"{DECLINED} paper {paper.get('name') or index}"
        missing = [key for key in _DECLINED_FIELDS if key not in paper]
        if missing:
            problems.append(f"{label} lacks {', '.join(missing)}")
            continue
        if paper["name"] in names:
            problems.append(f"{label} is listed twice")
        names.add(paper["name"])
        problems += _declined_paper_problems(label, paper)
    return papers, problems


def _declined_paper_problems(label: str, paper: dict) -> list[str]:
    problems: list[str] = []
    if paper["reason"] not in DECLINE_REASONS:
        problems.append(f"{label} reason {paper['reason']!r} is not one of {', '.join(DECLINE_REASONS)}")
    for issue in paper["issues"]:
        if issue.get("kind") not in KINDS or not str(issue.get("what") or "").strip():
            problems.append(f"{label} has an issue without a known kind and text: {issue!r}")
    return problems


def summarize_declined(papers: list[dict]) -> dict[str, object]:
    kinds = Counter(issue["kind"] for paper in papers for issue in paper.get("issues", []))
    return {
        "papers": len(papers),
        "by_reason": dict(sorted(Counter(paper.get("reason") for paper in papers).items())),
        "issues": sum(kinds.values()),
        "by_kind": dict(sorted(kinds.items())),
    }
