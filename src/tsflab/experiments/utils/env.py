"""Capture the execution environment + git provenance for a run.

These helpers feed ``tsflab.core.RunEnv`` so every result records *where* and *with
what code* it was produced. Both functions are defensive — they never raise and
return only the fields they could resolve, so a missing torch / non-git checkout
degrades gracefully instead of breaking a training run.
"""

from __future__ import annotations

import platform
import subprocess
import sys


def collect_env(device=None) -> dict:
    """Best-effort snapshot of the Python / torch / CUDA / GPU environment.

    Parameters
    ----------
    device : optional
        A ``torch.device`` (or anything with a ``.type``); only used to prefer
        the active CUDA device index when naming the GPU.

    Returns
    -------
    dict
        Subset of ``{python, platform, torch, cuda, gpu, framework_version}``.
    """
    env: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:  # torch is optional at this layer; never let it break a run
        import torch

        env["torch"] = torch.__version__
        if torch.cuda.is_available():
            env["cuda"] = torch.version.cuda
            try:
                idx = getattr(device, "index", None)
                idx = idx if isinstance(idx, int) else torch.cuda.current_device()
                env["gpu"] = torch.cuda.get_device_name(idx)
            except Exception:
                pass
    except Exception:
        pass
    try:
        from importlib.metadata import version

        env["framework_version"] = version("tsflab")
    except Exception:
        pass
    return env


def collect_git(cwd: str | None = None) -> dict:
    """Best-effort git SHA + dirty flag for the repo at ``cwd`` (default: CWD).

    Returns ``{}`` when there is no git, no repo, or git errors out.
    """
    out: dict = {}

    def _git(*args: str) -> subprocess.CompletedProcess | None:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return None

    head = _git("rev-parse", "HEAD")
    if head is not None and head.returncode == 0:
        out["git_sha"] = head.stdout.strip()
    status = _git("status", "--porcelain")
    if status is not None and status.returncode == 0:
        out["git_dirty"] = bool(status.stdout.strip())
    return out
