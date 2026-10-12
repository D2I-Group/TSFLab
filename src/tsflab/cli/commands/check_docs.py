"""Validate human documentation projections, links, and audience boundaries.

Catches the exact kind of drift that's easy to introduce by hand and easy to
miss in review: a stale generated model table (``docs/en/models.md``, rendered
from card facts), unindexed guides, prose about the retired verification system,
and Agent-only implementation details leaking into user documentation.
Pure text/file checks keep this safe and fast enough for every PR.
"""
from __future__ import annotations

import argparse
import re
import sys

from tsflab.core.paths import repository_root, require_checkout

from tsflab.catalog.cards.metadata import model_records

ROOT = repository_root()
DOC_DIR = ROOT / "docs" / "en"
OBSOLETE_DOC_TOKENS = (
    "moderntsf",  # pre-rename package name
    "tool/",
    "modern-tsf",
    "MODEL_NAME_MAP",
    "MODEL_REGISTRY",
    "registry.py",
    "schema.py",
    "src/tsflab/models/module",
    "src/tsflab/models/_external",
)
#: Tokens of the retired verification and card system (before ``tsflab.card/1``).
#: Scanned in human docs and Agent assets; cards are audited by their own schema.
REMOVED_SYSTEM_TOKENS = (
    "verification/evidence",
    "verification/models.toml",
    "verification/index.json",
    "verify --stale",
    "verify --index",
    "model-card:canonical",
    "component-card:generated",
    "dataset-card:canonical",
)
AGENT_ONLY_TOKENS = (
    ".agents/",
    ".claude/skills",
    "AGENTS.md",
    "CLAUDE.md",
    "SKILL.md",
    "tsflab.cli.commands",
)


def _first_sentence(text: str) -> str:
    text = " ".join(text.split())
    head, dot, _ = text.partition(". ")
    return (head + "." if dot else text).replace("|", "\\|")


def render_models_doc() -> str:
    """A compact table of card facts; ``tsf catalog`` is the full, searchable view."""
    records = model_records(ROOT)
    passed = sum(1 for r in records if (r.get("admission") or {}).get("status") == "passed")
    intro = (
        "# Models and methods\n\n"
        f"TSFLab exposes {len(records)} model and method entries through one flat "
        "public catalog. Architecture families are retrieval tags on each card, not "
        "directories or categories. Presets configure runs and do not create additional "
        "entries. This page is generated from the model cards; search and open them with "
        "`tsf catalog search` and `tsf catalog show <name>`.\n\n"
        "**Fidelity** says how the implementation was checked: `reference-checked` "
        "(paper and pinned official code), `paper-only`, `inferred` (the paper leaves "
        "material details open), or `composed` (assembled from catalog components). "
        "**Admission** is the recorded result of the executable contract "
        f"(`tsf model verify`); {passed} of {len(records)} entries have passed.\n\n"
        "| Name | Description | Fidelity | Admission | Card |\n"
        "|---|---|---|---|---|\n"
    )
    rows = []
    for record in records:
        package = str(record["package"])
        status = str((record.get("admission") or {}).get("status") or "pending")
        rows.append(
            f"| `{record['name']}` | {_first_sentence(str(record['summary']))} | "
            f"{record.get('fidelity') or '—'} | {status} | "
            f"[card](../../src/tsflab/models/{package}/README.md) |"
        )
    return intro + "\n".join(rows) + "\n"


def _generated_model_doc_problems() -> list[str]:
    problems: list[str] = []
    path = DOC_DIR / "models.md"
    if path.read_text(encoding="utf-8") != render_models_doc():
        problems.append(f"{path.relative_to(ROOT)} is stale; regenerate with `tsf repo cards`")
    return problems


def write_generated_model_docs() -> None:
    """Refresh the human-readable model catalog projection."""
    (DOC_DIR / "models.md").write_text(render_models_doc(), encoding="utf-8")


def _readme_count_problems() -> list[str]:
    """The catalog counts quoted in README.md must match the code."""
    from tsflab.catalog.cards.datasets import dataset_records
    from tsflab.catalog.components import COMPONENT_CATALOG

    text = (ROOT / "README.md").read_text(encoding="utf-8")
    expected = {
        r"(\d+) methods as peers": len(model_records(ROOT)),
        r"(\d+) shared components": len(COMPONENT_CATALOG.names()),
        r"(\d+) dataset presets": len(dataset_records(ROOT)),
    }
    problems = []
    for pattern, count in expected.items():
        found = re.search(pattern, text)
        if found is None or int(found.group(1)) != count:
            stated = found.group(1) if found else "nothing"
            label = pattern.split(") ", 1)[1]
            problems.append(f"README.md states {stated} {label}; the catalog has {count}")
    return problems


def _docs_index_problems() -> list[str]:
    problems = []
    readme = DOC_DIR / "README.md"
    readme_text = readme.read_text()
    linked = set(re.findall(r"\[([a-zA-Z0-9_.-]+\.md)\]", readme_text))
    for page in DOC_DIR.glob("*.md"):
        if page.name == "README.md":
            continue
        if page.name not in linked:
            problems.append(
                f"{page.relative_to(ROOT)} has no link in {readme.relative_to(ROOT)}"
            )
    return problems


def _obsolete_reference_problems() -> list[str]:
    problems: list[str] = []
    human = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", *DOC_DIR.glob("*.md")]
    agent = [ROOT / "AGENTS.md", *(ROOT / ".agents").rglob("*.md"), *(ROOT / ".agents").rglob("*.toml")]
    for path in human + agent:
        text = path.read_text(encoding="utf-8")
        tokens = (*OBSOLETE_DOC_TOKENS, *REMOVED_SYSTEM_TOKENS) if path in human else REMOVED_SYSTEM_TOKENS
        for token in tokens:
            if token in text:
                problems.append(f"{path.relative_to(ROOT)} references obsolete {token!r}")
    return problems


def _audience_boundary_problems() -> list[str]:
    problems: list[str] = []
    human_paths = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", *DOC_DIR.glob("*.md")]
    for path in human_paths:
        text = path.read_text(encoding="utf-8")
        for token in AGENT_ONLY_TOKENS:
            if token in text:
                problems.append(
                    f"{path.relative_to(ROOT)} exposes Agent-only detail {token!r}"
                )

    for path in (ROOT / ".agents").rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for token in ("docs/en/", "CONTRIBUTING.md"):
            if token in text:
                problems.append(
                    f"{path.relative_to(ROOT)} depends on human documentation {token!r}"
                )
    return problems


def _relative_link_problems() -> list[str]:
    problems: list[str] = []
    paths = [
        ROOT / "README.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "THIRD_PARTY_NOTICES.md",
    ]
    paths.extend(DOC_DIR.glob("*.md"))
    paths.extend((ROOT / ".agents").rglob("*.md"))
    paths.extend((ROOT / "src" / "tsflab" / "models").glob("*/README.md"))

    for path in paths:
        text = path.read_text(encoding="utf-8")
        for raw_target in re.findall(r"\]\(([^)]+)\)", text):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith(("#", "/", "mailto:")):
                continue
            resolved = path.parent / target
            if not resolved.exists():
                problems.append(
                    f"{path.relative_to(ROOT)} has broken relative link {raw_target!r}"
                )
    return problems


#: ``tsf <command> [word]`` in prose or code; the word is checked when it can be a subcommand.
_COMMAND_REFERENCE = re.compile(r"(?<![\w/.-])tsf ([a-z]+)(?: ([a-z][a-z-]*))?(?: ([a-z][a-z-]*))?")


def command_reference_problems(paths=None) -> list[str]:
    """Every documented ``tsf`` command must exist in the command inventory and not be retired."""
    from tsflab.cli.inventory import RETIRED, known

    if paths is None:
        paths = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", ROOT / "AGENTS.md", *DOC_DIR.glob("*.md"),
                 *(ROOT / ".agents").rglob("*.md"), *(ROOT / ".agents").rglob("*.toml")]
    problems = []
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for group, word, third in _COMMAND_REFERENCE.findall(line):
                name = f"{group} {word}".strip()
                where = f"{path.relative_to(ROOT)}:{number}"
                retired = next((old for old in (f"{name} {third}", name) if old in RETIRED), None)
                if retired:
                    problems.append(f"{where} uses retired `tsf {retired}`; use `{RETIRED[retired]}`")
                elif not known(group, word or None):
                    problems.append(f"{where} names unknown command `tsf {name}` (see tsflab.cli.inventory)")
    return problems


def check() -> list[str]:
    return (
        command_reference_problems() +
        _generated_model_doc_problems()
        + _readme_count_problems()
        + _docs_index_problems()
        + _obsolete_reference_problems()
        + _audience_boundary_problems()
        + _relative_link_problems()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write",
        action="store_true",
        help="regenerate the model catalog before checking",
    )
    args = parser.parse_args()
    if args.write:
        require_checkout("documentation regeneration")
        write_generated_model_docs()
    problems = check()
    if not problems:
        print("OK: generated model docs and indexes are consistent.")
        return 0
    print(f"Found {len(problems)} doc inconsistency(ies):")
    for p in problems:
        print(f"  - {p}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
