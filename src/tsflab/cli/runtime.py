"""Shared process and model-slug helpers for public CLI commands."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from tsflab.core.paths import repository_root, working_root

ROOT = repository_root()
WORK_ROOT = working_root()
RUN_CONFIG_DIR = ROOT / "configs" / "runs"


def module_slug(name: str) -> str:
    """Normalize a public model name to the repository's lowercase module slug."""
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


def module_for_model(name: str) -> str:
    """Resolve a model name to its package module through the lazy catalog."""
    try:
        from tsflab.catalog.registry.models import MODEL_CATALOG

        path = MODEL_CATALOG.refs().get(name)
        if path:
            return path.split(".")[2]
    except Exception:
        pass
    return module_slug(name)


def passthrough(script: str, rest: list[str]) -> int:
    """Run one internal command module behind the stable public CLI surface."""
    module = script.removesuffix(".py")
    argv = [sys.executable, "-m", f"tsflab.cli.commands.{module}", *rest]
    from tsflab.cli.output import CAPTURE_CHILD_OUTPUT

    if CAPTURE_CHILD_OUTPUT.get():
        result = subprocess.run(argv, cwd=WORK_ROOT, capture_output=True, text=True)
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        return result.returncode
    return subprocess.run(argv, cwd=WORK_ROOT).returncode


def _config_path(cfg: str) -> Path:
    config_path = Path(cfg)
    if not config_path.is_absolute() and not (WORK_ROOT / config_path).exists():
        packaged_candidate = ROOT / config_path
        if packaged_candidate.exists():
            config_path = packaged_candidate
    return config_path


def _record_round(round_id: str, cfg: str, returncode: int, output: str) -> None:
    from tsflab.research.rounds import add_event, write_log

    log = write_log(round_id, Path(cfg).stem, output)
    add_event(
        round_id,
        "observation" if returncode == 0 else "failure",
        f"Command for {cfg} exited with status {returncode}",
        details={"config": cfg, "exit_code": returncode, "log": str(log)},
    )


def run_config(cfg: str, env_extra: dict | None = None) -> tuple[str, int, str]:
    """Execute one run configuration and return its path, status, and output tail."""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    config_path = _config_path(cfg)
    argv = [sys.executable, "-m", "tsflab.experiments.run_config", "--config", str(config_path)]
    proc = subprocess.run(
        argv,
        cwd=WORK_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    tail = next((line.strip() for line in reversed(output.splitlines()) if line.strip()), "")
    round_id = env.get("TSFLAB_RESEARCH_ROUND")
    if round_id:
        _record_round(round_id, cfg, proc.returncode, output)
    return cfg, proc.returncode, tail
