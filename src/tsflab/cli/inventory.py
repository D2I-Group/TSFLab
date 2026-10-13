"""The public command inventory: one declarative table for every ``tsf`` command.

Each entry gives a name, a one-line purpose, and its usage. ``tsf --help``, the
router usage strings, the ``tsflab.cli.main`` docstring, ``tsf agent interface``,
and the documentation check (``tsf repo check --audit``) all read this table, so a
command is described in exactly one place. Pure data: importing it is cheap.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    """One leaf command, ``tsf <group> <name> <usage>``; ``name`` may be empty for the bare group."""

    name: str
    usage: str
    purpose: str


@dataclass(frozen=True)
class Group:
    """One top-level command and its leaves; ``positional`` groups also take a bare argument."""

    name: str
    section: str
    purpose: str
    commands: tuple[Command, ...]
    positional: bool = False


C = Command
GROUPS: tuple[Group, ...] = (
    Group("catalog", "Read path", "overview, search, list, show, match for models, components, datasets", (
        C("overview", "[--json]", "counts per kind and the next commands (also bare `tsf catalog`)"),
        C("search", "<terms...> [--kind model|component|dataset] [--capability C] [--limit N] [--json]",
          "ranked L0 lines: name, kind, summary, tags"),
        C("list", "[--kind model|component|dataset] [--json]", "every resource of a kind"),
        C("show", "<name...> [--kind K] [--depth {0,1,2,3}] [--json] [--params]",
          "open cards at L1 (or --depth 0-3); several names in one call; --params: model parameter schema only"),
        C("match", "<dataset...> [--extra TERM...] [--top N] [--json]",
          "models and components whose card fits match the data"),
    )),
    Group("data", "Modules", "register, prepare, profile, split, and publish datasets", (
        C("add", "<dataset> --pattern custom|single [--path FILE] [--target COL] [--force]",
          "register a CSV as a preset and card, profiled"),
        C("prepare", "[--from traffic|ultratraffic|gift-eval|tfb|dcrnn] [args...]",
          "convert or fetch data into loader layouts"),
        C("analyze", "<dataset> | --config TOML | --path FILE [--data-path FILE] [--split S] [--include-test] "
          "[--per-channel] [--plot] [--write-card] [--json [--compact]]",
          "train/val profile, TFB characteristics, and sample plot (test on request)"),
        C("splits", "<dataset> [--seq-len N] [--horizon N] [--path FILE] [--json]",
          "official split borders and scaling rule"),
        C("download", "<preset...> | --all | --list | --check [--root DIR]", "fetch published dataset files"),
        C("publish", "<preset...> [--path DIR...] [--repo OWNER/NAME]", "upload preset files (maintainers)"),
        C("audit", "", "dataset cards complete and current"),
    )),
    Group("model", "Modules", "scaffold, admit, verify, compose, and audit models", (
        C("scaffold", "<Name> --paper-title T --paper-url U --venue V --year Y --components C "
          "[--params SPEC] [--task-mode M]", "create a model package and card"),
        C("add", "<Name> [--verify]", "admit a completed package into the catalog"),
        C("artifacts", "<Name> [--fetch ARTIFACT] [--json]", "inspect or fetch pinned runtime artifacts"),
        C("verify", "<Name...> | --all | --changed [--base REF] [--jobs N] [--json]",
          "run admission contracts; record [admission]"),
        C("compose", "<spec.toml> [--json] [--write-config PATH --dataset D --enc-in N [--horizon N] "
          "[--records DIR] [--smoke]] [--register NAME [--dry-run]]", "validate or build a recombination"),
        C("similar", "[slug...] [--threshold 0.6] [--min-tokens 40] [--top N] [--json]",
          "near-duplicate code across model packages"),
        C("issues", "[Name...] [--kind KIND] [--json | --summary]", "upstream paper and code issues in cards"),
        C("audit", "[Name...] [--components] [--release] [--json | --summary]", "card and admission audit"),
    )),
    Group("run", "Modules", "run experiments; --smoke, --dry-run, --backend local|queue|slurm", (
        C("", "<config.toml...> | --model NAME [--set K=V...] [--sweep K=LIST...] [--dry-run] [--round ID] "
          "[--eval-split val|test] [--policy TOML] [--jobs N|auto] [--gpus LIST|auto] [--output JSON] "
          "[--json [--compact]]",
          "validate and run a sweep (--prepare-only persists it; --output keys the request)"),
        C("evaluate", "<run_dir...> --eval-split val|test [--json]",
          "score frozen checkpoints of finished runs on one split, without training"),
        C("status", "<directory> [--json]", "state of a run or sweep"),
        C("wait", "<directory> [--timeout S] [--json]", "block until a background sweep ends"),
        C("cancel", "<directory> [--json]", "cancel a run or sweep"),
        C("resume", "<directory> [--json]", "resume an interrupted sweep"),
        C("--smoke", "--all | --model NAME | --config PATH... [--jobs N]", "run smoke configs concurrently"),
        C("--backend", "queue|slurm ACTION DIRECTORY [args...]", "submit a prepared sweep to a queue or Slurm"),
    ), positional=True),
    Group("env", "Modules", "audit and set up the environment; storage and usage", (
        C("audit", "[--config TOML] [--policy TOML] [--json]", "environment and run preflight (default)"),
        C("doctor", "[--json]", "detect this machine; print install and run profiles"),
        C("setup", "[--json]", "install the matching profile"),
        C("storage", "status|cleanup <directory> [--policy TOML] [--apply]", "disk use and retention"),
        C("usage", "<action> <directory> [--operation OP] [--tokens N] [--cost-usd X]",
          "record and check token and cost budgets"),
    )),
    Group("result", "Modules", "aggregate, rank, plot, report, board, submit, leaderboard, hub", (
        C("aggregate", "<dataset> [--records DIR] [--out CSV] [--filter EXPR] [--collapse]",
          "merge the run CSVs of a dataset"),
        C("rank", "<dataset> [--records DIR] [--out-mse CSV] [--out-mae CSV] [--out-long CSV]",
          "rank models per horizon and seed"),
        C("plot", "--csv CSV --x FIELD --y FIELD --size FIELD [--out PNG]", "bubble chart of a results CSV"),
        C("report", "<dataset> [--records DIR] [--horizon N] [--top N] [--out MD]", "Markdown benchmark report"),
        C("predictions", "--config TOML [--num-samples N] [--channel C] [--out PNG]", "plot forecasts against truth"),
        C("board", "<dataset> [--horizon N] [--top N] [--metric M] [--records DIR] [--json]",
          "best methods: published board plus local records"),
        C("submit", "[<dataset>] [--model M] [--run-id ID | --latest | --all] [--records DIR] [--out DIR]",
          "package run records as submission bundles"),
        C("leaderboard", "[--records DIR] [--out JSON] [--primary-metric M]",
          "build leaderboard.json from submissions"),
        C("hub", "{pack,push,push-top,list,pull,results,init} [args...]",
          "checkpoint and results repositories on the Hub"),
    )),
    Group("realtime", "Modules", "rolling real-time tracks: update, open, forecast, score", (
        C("list", "", "tracks and their last observation"),
        C("update", "--track T [--bootstrap | --no-fetch] [--pull] [--push] [--repo OWNER/NAME]",
          "append a data release"),
        C("open", "--track T [--now ISO]", "open the next round"),
        C("baselines", "--track T [--round R]", "write and validate baseline forecasts"),
        C("forecast", "--track T --model M [--round R] [--set SECTION.KEY=VALUE...] [--records DIR]",
          "train a catalog model; write and validate its forecast"),
        C("validate", "FILE... [--received-at ISO]", "check forecast files produced elsewhere"),
        C("score", "--track T", "score rounds whose truth arrived"),
        C("weekly", "--track T... [--pull] [--push]", "update, score, and open with baselines"),
        C("replay", "--track T --end DATE [--weeks N] [--models M...] [--out JSON]",
          "backtest the protocol on history"),
    )),
    Group("research", "Modules", "research rounds: start, list, show, note, status, iteration, reference, screen", (
        C("start", "--goal TEXT [--task NAME] [--max-runs N] [--max-iterations N] [--policy TOML] [--id ID]",
          "open a budgeted round"),
        C("list", "[--json]", "rounds and their budgets"),
        C("show", "<id>", "state and events of a round"),
        C("note", "<id> --text TEXT [--kind KIND]", "append a structured event"),
        C("status", "<id> <status> [--message TEXT]", "update the round status"),
        C("iteration", "<id> [--operation ID]", "claim the next bounded iteration"),
        C("reference", "<id> RUN_DIR...", "reuse saved validation baselines without new training claims"),
        C("screen", "<id> --baseline RUN_DIR... --candidate RUN_DIR... [--metric mse|mae] [--min-improvement X] [--stop-on-fail]",
          "check matched validation runs before confirmation"),
    )),
    Group("repo", "Repository and Agents", "check (mergeable gate), cards (derived indexes), schema", (
        C("check", "[--scope full|changed|release] [--base REF] [--only STEP...] [--list] [--fail-fast] [--json]",
          "the single mergeable gate; release scope requires every admission"),
        C("check --audit", "", "static audits only (agent assets, components, complexity, cards, docs)"),
        C("check --contracts", "construct|forward|backward|strict [--models NAME...]",
          "execute model tensor contracts"),
        C("cards", "", "regenerate derived indexes (.agents/README.md, docs/en/models.md); "
          "cards themselves are never generated"),
        C("schema", "[--check] [--out DIR]", "export the TSF-Core JSON Schema"),
    )),
    Group("agent", "Repository and Agents", "task templates, interface discovery, modules, sync", (
        C("task", "{list,show,render,start,validate} [args...]",
          "bounded task templates; start prepares a round and prompt, it does not dispatch an Agent"),
        C("interface", "[show|schema|api] [--module SECTION] [--json]",
          "execution recipes, policy schema, Python API modules"),
        C("modules", "[--json]", "modules, skills, and tasks of this project"),
        C("sync", "", "refresh project skills and tasks from the installed TSFLab"),
    )),
    Group("init", "Repository and Agents", "scaffold a standalone project for chosen modules", (
        C("", "[directory] [--name NAME] [--modules LIST] [--force]", "create the project files"),
    ), positional=True),
)

#: Retired commands: what replaces them. The router prints one line naming the
#: replacement and forwards when the replacement is a direct equivalent.
RETIRED: dict[str, str] = {
    "data inspect": "tsf data analyze",
    "data plot": "tsf data analyze --plot",
    "agent interface modules": "tsf agent interface api",
}

BY_NAME: dict[str, Group] = {group.name: group for group in GROUPS}


def leaf_names(group: str) -> list[str]:
    """Subcommand words of ``group`` (flags and the bare form excluded)."""
    return [c.name.split()[0] for c in BY_NAME[group].commands if c.name and not c.name.startswith("-")]


def group_usage(group: str) -> str:
    """The ``tsf <group> --help`` text: a choice line, then one usage line and purpose per command."""
    entry = BY_NAME[group]
    names = list(dict.fromkeys(leaf_names(group)))
    head = f"usage: tsf {group} {{{','.join(names)}}} [args...]" if names else f"usage: tsf {group} [args...]"
    lines = [head, f"{entry.purpose}.", ""]
    for command in entry.commands:
        call = " ".join(part for part in ("tsf", group, command.name, command.usage) if part)
        lines.append(f"  {call}\n      {command.purpose}")
    return "\n".join(lines)


def overview() -> str:
    """The ``tsf --help`` text and the ``tsflab.cli.main`` docstring."""
    lines = [f"Public Agent CLI for TSFLab: {len(GROUPS)} commands, one per module.", "",
             "Usage:", "    tsf <command> [args...]", "    tsf --format json <command> [args...]"
             "   (versioned JSON envelope; --format json may appear anywhere)"]
    section = None
    for group in GROUPS:
        if group.section != section:
            section = group.section
            lines += ["", f"{section}:"]
        lines.append(f"    {group.name:<16} {group.purpose}")
    lines += ["", "Run ``tsf <command> --help`` for each command's usage lines (one table, "
              "tsflab.cli.inventory, feeds all help).",
              "Shared spellings: dataset or model name positional, --horizon, --out, --records, --json."]
    return "\n".join(lines) + "\n"


def purpose(group: str, name: str) -> str:
    """The one-line purpose of ``tsf <group> <name>``."""
    return next(c.purpose for c in BY_NAME[group].commands if c.name == name)


def retired_note(command: str, forwarding: bool) -> str:
    """The one line a retired command prints, naming its replacement."""
    tail = "; forwarding" if forwarding else ""
    return f"note: `tsf {command}` is replaced by `{RETIRED[command]}`{tail}"


def records() -> list[dict[str, str]]:
    """Every command as ``{"command", "usage", "purpose"}`` for structured discovery."""
    return [
        {"command": " ".join(part for part in ("tsf", group.name, command.name) if part),
         "usage": command.usage, "purpose": command.purpose}
        for group in GROUPS for command in group.commands
    ]


def known(group: str, word: str | None) -> bool:
    """Whether ``tsf <group> [word]`` names a public command (``word`` a subcommand or argument)."""
    entry = BY_NAME.get(group)
    if entry is None:
        return False
    return word is None or entry.positional or word in leaf_names(group)
