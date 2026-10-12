"""Validate the repository's cross-harness Agent-first asset layout."""

from __future__ import annotations

import re
from pathlib import Path

from tsflab.agent.index import render_index
from tsflab.agent.modules import MODULES, owners
from tsflab.core.paths import is_packaged_root, repository_root


ROOT = repository_root()
SKILLS = ROOT / ".agents" / "skills"
STANDARDS = ROOT / ".agents" / "STANDARDS.md"
INDEX = ROOT / ".agents" / "README.md"
ROOT_DOCS = {"STANDARDS.md", "README.md"}


def _link_target(path: Path) -> str | None:
    return str(path.readlink()) if path.is_symlink() else None


def audit_module_map(skills_on_disk: set[str]) -> list[str]:
    """Every skill and task belongs to exactly one module; the map names nothing unknown."""
    errors: list[str] = []
    task_names = {path.stem for path in (ROOT / ".agents" / "tasks").glob("*.toml")}
    for kind, on_disk in (("skills", skills_on_disk), ("tasks", task_names)):
        mapped = owners(kind)
        for name, modules in sorted(mapped.items()):
            if len(modules) > 1:
                errors.append(f"{kind[:-1]} {name!r} is listed by several modules: {', '.join(modules)}")
        unmapped = sorted(on_disk - mapped.keys())
        unknown = sorted(mapped.keys() - on_disk)
        if unmapped:
            errors.append(f"{kind} missing from tsflab.agent.modules.MODULES: {', '.join(unmapped)}")
        if unknown:
            errors.append(f"tsflab.agent.modules.MODULES names unknown {kind}: {', '.join(unknown)}")
    for module, info in MODULES.items():
        for key in ("skills", "tasks", "extras", "purpose", "commands", "context"):
            if key not in info:
                errors.append(f"module {module!r} lacks {key!r}")
    if not INDEX.is_file():
        errors.append(".agents/README.md is missing; run `tsf repo cards`")
    elif INDEX.read_text(encoding="utf-8") != render_index(ROOT / ".agents"):
        errors.append(".agents/README.md is stale; run `tsf repo cards`")
    return errors


def _module_command_problems() -> list[str]:
    """Entry commands in the module map must be commands of the public inventory."""
    from tsflab.agent.modules import MODULES
    from tsflab.cli.inventory import RETIRED, known

    errors = []
    for module, info in MODULES.items():
        for command in info["commands"]:  # type: ignore[union-attr]
            words = command.split()[1:]  # drop the leading "tsf"
            word = words[1] if len(words) > 1 else None
            if not known(words[0], word) or " ".join(words[:2]) in RETIRED:
                errors.append(f"module {module!r} lists unknown or retired command {command!r}")
    return errors


def audit_agent_assets() -> list[str]:
    """Return violations of the cross-harness Agent-first contract."""
    errors: list[str] = []
    agents_md = ROOT / "AGENTS.md"
    claude_md = ROOT / "CLAUDE.md"
    claude_skills = ROOT / ".claude" / "skills"

    if not agents_md.is_file() or agents_md.is_symlink():
        errors.append("AGENTS.md must be a real file")
    elif len(agents_md.read_text(encoding="utf-8").splitlines()) > 50:
        errors.append("AGENTS.md exceeds the 50-line always-loaded context budget")
    if not is_packaged_root(ROOT) and _link_target(claude_md) != "AGENTS.md":
        errors.append("CLAUDE.md must be a symlink to AGENTS.md")
    if not SKILLS.is_dir() or SKILLS.is_symlink():
        errors.append(".agents/skills must be the real canonical skill directory")
    if not STANDARDS.is_file() or STANDARDS.is_symlink():
        errors.append(".agents/STANDARDS.md must be the real consolidated contract")
    elif len(STANDARDS.read_text(encoding="utf-8").splitlines()) > 100:
        errors.append(".agents/STANDARDS.md exceeds the 100-line on-demand budget")
    if not is_packaged_root(ROOT) and _link_target(claude_skills) != "../.agents/skills":
        errors.append(".claude/skills must link to ../.agents/skills")
    root_agent_docs = {path.name for path in (ROOT / ".agents").glob("*.md")}
    if root_agent_docs != ROOT_DOCS:
        errors.append(".agents may contain only STANDARDS.md and the generated README.md at its root")

    for skill_file in sorted(SKILLS.rglob("SKILL.md")):
        if skill_file.parent.parent != SKILLS:
            errors.append(
                f"{skill_file.relative_to(ROOT)}: skills must be one level below .agents/skills"
            )

    seen: set[str] = set()
    for skill_file in sorted(SKILLS.glob("*/SKILL.md")):
        text = skill_file.read_text(encoding="utf-8")
        if len(text.splitlines()) > 80:
            errors.append(
                f"{skill_file.relative_to(ROOT)}: exceeds the 80-line entrypoint budget"
            )
        match = re.match(r"^---\n(?P<header>.*?)\n---\n", text, re.DOTALL)
        if match is None:
            errors.append(f"{skill_file.relative_to(ROOT)}: missing YAML frontmatter")
            continue
        header = match.group("header")
        name_match = re.search(r"^name:\s*[\"']?([^\"'\n]+)", header, re.MULTILINE)
        desc_match = re.search(
            r"^description:\s*(?P<description>.+?)\s*$", header, re.MULTILINE
        )
        if name_match is None:
            errors.append(f"{skill_file.relative_to(ROOT)}: missing name")
            continue
        name = name_match.group(1).strip()
        if len(name) > 64 or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) is None:
            errors.append(
                f"{skill_file.relative_to(ROOT)}: name must be 1-64 kebab-case characters"
            )
        if name != skill_file.parent.name:
            errors.append(
                f"{skill_file.relative_to(ROOT)}: name {name!r} does not match directory"
            )
        if name in seen:
            errors.append(f"duplicate skill name: {name}")
        seen.add(name)
        if desc_match is None:
            errors.append(f"{skill_file.relative_to(ROOT)}: missing description")
        else:
            description = desc_match.group("description").strip()
            if (
                len(description) >= 2
                and description[0] == description[-1]
                and description[0] in "\"'"
            ):
                description = description[1:-1].strip()
            if not description or len(description) > 1024:
                errors.append(
                    f"{skill_file.relative_to(ROOT)}: description must be 1-1024 characters"
                )
        compatibility_paths = (
            ".claude/skills",
            ".pi/skills",
            ".dsh/skills",
            "CLAUDE.md",
        )
        if any(path in text for path in compatibility_paths):
            errors.append(
                f"{skill_file.relative_to(ROOT)}: references a harness compatibility path"
            )
        for human_path in ("docs/en/", "CONTRIBUTING.md"):
            if human_path in text:
                errors.append(
                    f"{skill_file.relative_to(ROOT)}: depends on human documentation {human_path!r}"
                )
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if "://" in target or target.startswith(("#", "/")):
                continue
            resolved = skill_file.parent / target.split("#", 1)[0]
            if not resolved.exists():
                errors.append(
                    f"{skill_file.relative_to(ROOT)}: missing linked resource {target!r}"
                )
        for obsolete in ("tool/", "modern-tsf", "moderntsf", "MODEL_NAME_MAP", "registry.py", "schema.py"):
            if obsolete in text:
                errors.append(
                    f"{skill_file.relative_to(ROOT)}: references obsolete interface {obsolete!r}"
                )

    errors.extend(audit_module_map(seen))
    errors.extend(_module_command_problems())
    from tsflab.agent.tasks import audit_tasks

    errors.extend(audit_tasks())
    return errors


def main() -> int:
    errors = audit_agent_assets()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    count = len(list(SKILLS.glob("*/SKILL.md")))
    print(f"Cross-harness Agent-first assets OK: {count} skills")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
