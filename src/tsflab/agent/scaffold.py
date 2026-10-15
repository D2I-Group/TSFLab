"""tsf init — scaffold a standalone project that uses an installed TSFLab.

The generated project owns only its run configs, data, and outputs. Catalog
configs (``base.toml``, datasets, models) are inherited from the installed
package through ``tsflab://`` extends paths, so upgrading TSFLab upgrades the
defaults without copying them.

``--modules`` selects the module chain (data, models, experiments, release,
autoresearch; default: the whole chain). ``maintenance`` (audit, contributions)
needs a TSFLab checkout and is added only when named. The project gets an AGENTS.md generated from that
chain plus the chosen modules' skills and task templates, copied from the
installed package so they match its version (``tsf agent sync`` refreshes them
after an upgrade). The chosen modules are recorded under ``[tool.tsflab]`` in
the project's ``pyproject.toml``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tomllib
from importlib import metadata
from pathlib import Path

from tsflab.agent.index import render_index
from tsflab.agent.modules import ALL_MODULES, CHAIN, MODULES, find_project, module_skills, module_tasks, parse_modules

GENERATED_START = "<!-- tsflab:generated:start -->"
GENERATED_END = "<!-- tsflab:generated:end -->"

FILES: dict[str, str] = {
    "configs/runs/example.toml": """\
# Inherit catalog defaults from the installed TSFLab package.
extends = [
    "tsflab://configs/base.toml",
    "tsflab://configs/datasets/etth1.toml",
    "tsflab://configs/models/DLinear.toml",
]

[experiment]
description = "{name}: DLinear on ETTh1"

[sweep.task]
pred_len = [96, 192]
""",
    "dataset/README.md": """\
# dataset/

Local data files referenced by `configs/`. Inspect the catalog with
`tsf catalog list --kind dataset` and `tsf catalog show <name>`; ETTh1 is
expected at `dataset/ETT-small/ETTh1.csv` (`tsf data download etth1`).
""",
    ".gitignore": """\
dataset/*
!dataset/README.md
work_dirs/
__pycache__/
.venv/
""",
}


def _version() -> str:
    try:
        return metadata.version("tsflab")
    except metadata.PackageNotFoundError:
        return "unknown"


def extras_for(modules: list[str]) -> list[str]:
    """Pip extras needed by ``modules`` (``all`` when every module is chosen)."""
    if set(CHAIN) <= set(modules):
        return ["all"]
    extras: list[str] = []
    for name in modules:
        extras.extend(MODULES[name]["extras"])  # type: ignore[arg-type]
    return list(dict.fromkeys(extras))


def render_pyproject(name: str, modules: list[str]) -> str:
    extras = ",".join(extras_for(modules))
    listed = ", ".join(f'"{m}"' for m in modules)
    return f"""\
[project]
name = "{name}"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["tsflab[{extras}]"]

[tool.tsflab]
# Modules chosen at `tsf init`; read by `tsf agent` and `tsf catalog`.
modules = [{listed}]
assets_version = "{_version()}"
"""


def render_agents_md(name: str, modules: list[str]) -> str:
    """Project AGENTS.md generated from the module chain (kept within 50 lines)."""
    lines = [
        f"# {name} Agent Guide",
        GENERATED_START,
        f"Standalone project on TSFLab {_version()}. `.agents/skills/` holds workflows and",
        "`.agents/tasks/` bounded task templates, copied from the installed package for the modules",
        "below, indexed by module in `.agents/README.md`; `tsf agent sync` refreshes them after",
        "upgrading TSFLab. Claude Code reads `CLAUDE.md` and `.claude/skills`, which link to the same files.",
        "",
        "## Module chain",
        "Data -> Models -> Experiments -> Release, each producing context that AutoResearch",
        "consumes. Enabled here (`tsf agent modules` lists them with skills and tasks):",
    ]
    for module in modules:
        info = MODULES[module]
        lines.append(f"- {module}: {info['purpose']}.")
        lines.append(f"  Skills: {', '.join(info['skills'])}.")  # type: ignore[arg-type]
        lines.append(f"  Commands: {', '.join(f'`{c}`' for c in info['commands'])}.")  # type: ignore[union-attr]
    tasks = module_tasks(modules)
    if tasks:
        lines.append(f"Task templates: {', '.join(tasks)} (`tsf agent task list|show|render`).")
    lines += [
        "",
        "## Progressive disclosure",
        "Start at `tsf catalog` (what exists, plus this project's modules), then",
        "`tsf catalog search <terms> [--kind model|component|dataset]` (L0 lines) and",
        "`tsf catalog show <name> [--depth 1|2|3]` (L1 contract, L2 card, L3 files). Open deeper",
        "levels only when a decision needs them. `tsf <command> --help` lists each module's options.",
        "",
        "## Rules",
        "- Run configs in `configs/runs/` extend `tsflab://configs/...`; keep project changes small.",
        "- Check the environment with `tsf env`; preview with `tsf run <cfg> --dry-run` before a run.",
        "- Verify paper and official-code claims before calling an implementation faithful.",
        "- Preserve user data and `work_dirs/` outputs unless removal is requested. Publishing,",
        "  pushing, and external issues or pull requests need explicit authorization.",
    ]
    if {"data", "models"} & set(modules):
        lines.append("- Skills that add datasets or models change the shared catalog and need a TSFLab")
        lines.append("  checkout (`tsf model add`, `tsf repo cards`); here they plan and draft the change.")
    lines += [
        GENERATED_END,
        "",
        "## Project notes",
        "Add project-specific context below; `tsf agent sync` keeps it.",
        "",
    ]
    return "\n".join(lines)


def _agent_sources() -> Path:
    from tsflab.core.paths import repository_root

    return repository_root() / ".agents"


def _task_skills(source: Path, tasks: list[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for task in tasks:
        data = tomllib.loads((source / "tasks" / f"{task}.toml").read_text(encoding="utf-8"))
        result[task] = list(data.get("skills", []))
    return result


def sync_agent_assets(target: Path, modules: list[str]) -> list[str]:
    """Copy the chosen modules' skills and tasks from the installed package into ``target``."""
    source = _agent_sources()
    tasks = module_tasks(modules)
    skills = module_skills(modules, _task_skills(source, tasks))
    written: list[str] = []
    for skill in skills:
        destination = target / ".agents" / "skills" / skill
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(
            source / "skills" / skill, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")
        )
        written.append(f".agents/skills/{skill}")
    (target / ".agents" / "tasks").mkdir(parents=True, exist_ok=True)
    for task in tasks:
        shutil.copy2(source / "tasks" / f"{task}.toml", target / ".agents" / "tasks" / f"{task}.toml")
        written.append(f".agents/tasks/{task}.toml")
    (target / ".agents" / "README.md").write_text(render_index(source, modules, project=True), encoding="utf-8")
    written.append(".agents/README.md")
    return written


def _link(target: Path, link: Path, destination: str) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        return
    try:
        link.symlink_to(destination)
    except OSError:  # symlinks unavailable (e.g. some Windows setups): copy instead
        source = (link.parent / destination).resolve()
        if source.is_dir():
            shutil.copytree(source, link)
        else:
            shutil.copy2(source, link)


def init_project(
    target: Path,
    name: str | None = None,
    force: bool = False,
    modules: list[str] | None = None,
) -> list[Path]:
    """Write the project skeleton into ``target`` and return the created files."""
    name = name or target.resolve().name
    modules = list(CHAIN) if modules is None else modules
    generated = {
        **FILES,
        "pyproject.toml": render_pyproject(name, modules),
        "AGENTS.md": render_agents_md(name, modules),
        "README.md": render_readme(name, modules),
    }
    existing = [target / rel for rel in generated if (target / rel).exists()]
    if existing and not force:
        raise FileExistsError("refusing to overwrite: " + ", ".join(str(path) for path in existing))
    written = []
    for rel, template in generated.items():
        path = target / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(template.replace("{name}", name), encoding="utf-8")
        written.append(path)
    written += [target / rel for rel in sync_agent_assets(target, modules)]
    if force:
        for stale in (target / "CLAUDE.md", target / ".claude" / "skills"):
            if stale.is_symlink():
                stale.unlink()
    _link(target, target / "CLAUDE.md", "AGENTS.md")
    _link(target, target / ".claude" / "skills", "../.agents/skills")
    written += [target / "CLAUDE.md", target / ".claude" / "skills"]
    return written


def render_readme(name: str, modules: list[str]) -> str:
    extras = ",".join(extras_for(modules))
    steps = [
        "tsf env                                    # check devices, data, and output capacity",
        "tsf catalog                                # what exists: models, components, datasets",
        "tsf run configs/runs/example.toml --dry-run   # preview the expanded runs",
        "tsf run configs/runs/example.toml          # train and evaluate; outputs go to work_dirs/",
    ]
    if "release" in modules:
        steps.append("tsf result hub push <run_id> --repo <owner>/<weights-repo>   # publish a checkpoint (explicit)")
    body = "\n".join(steps)
    return f"""\
# {name}

A time-series forecasting project built on
[TSFLab](https://github.com/Diaugeia/TSFLab). Modules: {", ".join(modules)}.

```bash
uv sync                                    # install tsflab[{extras}]
{body}
```

Run configs extend the installed catalog with `tsflab://configs/...`
paths; local `extends` paths work as usual. Agents read `AGENTS.md`; skills and
task templates for the chosen modules live in `.agents/` (`tsf agent modules`,
`tsf agent sync` after upgrading TSFLab).
"""


def sync_command(argv: list[str]) -> int:
    """``tsf agent sync``: refresh a project's skills, tasks, and generated AGENTS.md."""
    parser = argparse.ArgumentParser(prog="tsf agent sync", description=sync_command.__doc__)
    parser.add_argument("--add", help="also enable these modules (comma separated)")
    args = parser.parse_args(argv)
    found = find_project()
    if found is None:
        print("error: not inside a `tsf init` project ([tool.tsflab] not found)", file=sys.stderr)
        return 2
    directory, modules = found
    try:
        if args.add:
            modules = parse_modules(",".join([*modules, *args.add.split(",")]))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    written = sync_agent_assets(directory, modules)
    pyproject = directory / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    table = tomllib.loads(text)
    name = table.get("project", {}).get("name", directory.name)
    listed = ", ".join(f'"{m}"' for m in modules)
    import re

    text = re.sub(r"(?m)^modules = \[.*\]$", f"modules = [{listed}]", text)
    text = re.sub(r'(?m)^assets_version = ".*"$', f'assets_version = "{_version()}"', text)
    pyproject.write_text(text, encoding="utf-8")
    agents = directory / "AGENTS.md"
    notes = ""
    if agents.is_file() and GENERATED_END in agents.read_text(encoding="utf-8"):
        notes = agents.read_text(encoding="utf-8").split(GENERATED_END, 1)[1]
    fresh = render_agents_md(name, modules)
    agents.write_text(
        fresh.split(GENERATED_END, 1)[0] + GENERATED_END + (notes or fresh.split(GENERATED_END, 1)[1]), encoding="utf-8"
    )
    print(f"Synced {len(written)} skill/task assets for modules: {', '.join(modules)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tsf init", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", nargs="?", default=".")
    parser.add_argument("--name", help="project name (default: directory name)")
    parser.add_argument(
        "--modules",
        default=None,
        help=f"comma-separated modules from {','.join(ALL_MODULES)} (default: {','.join(CHAIN)})",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    args = parser.parse_args(argv)
    try:
        modules = parse_modules(args.modules)
        written = init_project(Path(args.directory), args.name, args.force, modules)
    except (FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for path in written:
        print(f"created {path}")
    return 0
