"""Repository consistency and executable model-contract commands."""

from __future__ import annotations

import sys

from tsflab.cli.runtime import passthrough


def regenerate_cards(args: list[str]) -> int:
    """Regenerate derived indexes (Agent module index, docs); cards themselves are never generated."""
    if args:
        print("tsf repo cards takes no arguments", file=sys.stderr)
        return 2
    from tsflab.core.paths import require_checkout

    try:
        root = require_checkout("tsf repo cards")
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    from tsflab.agent.index import render_index

    index = root / ".agents" / "README.md"
    fresh = render_index(root / ".agents")
    if not index.is_file() or index.read_text(encoding="utf-8") != fresh:
        index.write_text(fresh, encoding="utf-8")
        print("Regenerated .agents/README.md (module index)")
    return passthrough("check_docs.py", ["--write"])


def run_audit() -> int:
    """Run the static repository audits in-process; return a nonzero code on failure."""
    from tsflab.agent.assets import main as audit_agent_assets
    from tsflab.catalog.cards.audit import audit_cards as _audit_cards
    from tsflab.core.paths import repository_root

    def audit_cards() -> int:
        errors = _audit_cards(repository_root())
        for error in errors:
            print(f"ERROR: {error}")
        if not errors:
            print("Cards OK (tsflab.card/1)")
        return 1 if errors else 0

    checks = [
        ("agent-assets", audit_agent_assets),
        ("components", lambda: __import__("tsflab.catalog.component_audit", fromlist=["main"]).main()),
        ("complexity", lambda: __import__("tsflab.catalog.complexity_audit", fromlist=["main"]).main()),
        ("cards", audit_cards),
        ("model-catalog", lambda: passthrough("check_registry.py", [])),
        ("documentation", lambda: passthrough("check_docs.py", [])),
    ]
    results = []
    for name, check in checks:
        code = check()
        results.append(code)
        print(f"{'PASS' if code == 0 else 'FAIL'} {name}")
    return 1 if any(results) else 0


def run_contracts(level: str, models: list[str] | None) -> int:
    """Execute model tensor contracts at ``level`` (construct, forward, backward, strict)."""
    from tsflab.catalog.model_contracts import audit_model_contracts
    from tsflab.catalog.registry.models import MODEL_CATALOG

    strict = level == "strict"
    backward = level in {"backward", "strict"}
    forward = backward or level == "forward"
    names = models or MODEL_CATALOG.names()
    failed = audit_model_contracts(names=names, forward=forward, backward=backward, strict=strict)
    for failure in failed:
        print(f"FAIL {failure.stage} {failure.model}: {failure.error}")
    label = {"strict": "strict-checked", "backward": "backward-checked",
             "forward": "forward-checked"}.get(level, "constructed")
    print(f"{label.capitalize()} {len(names) - len(failed)}/{len(names)} models")
    return 1 if failed else 0


def repository_command(args: list[str]) -> int:
    """Check repository contracts, regenerate cards, or export the TSF-Core schema."""
    usage = "usage: tsf repo {check,cards,schema} [args...]"
    if not args or args[0] in {"-h", "--help", "help"}:
        print(usage)
        print("  check   the single mergeable gate: [--scope full|changed] [--only STEP...] [--json]")
        print("          --audit runs the static audits only; --contracts construct|forward|backward|strict")
        print("          [--models NAME...] executes model tensor contracts")
        print("  cards   regenerate component/dataset cards, the model documentation index, and .agents/README.md")
        print("  schema  export TSF-Core JSON Schema [--check] [--out DIR]")
        return 0
    action, rest = args[0], args[1:]
    if action == "check":
        from tsflab.cli.commands.gate import check_command

        return check_command(rest)
    if action == "cards":
        return regenerate_cards(rest)
    if action == "schema":
        from tsflab.core.export import main as schema_main

        return schema_main(rest)
    print(usage, file=sys.stderr)
    return 2
