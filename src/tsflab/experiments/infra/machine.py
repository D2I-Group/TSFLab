"""Machine profiles: detect this host and recommend install and run presets.

Pure mapping functions (``cuda_from_driver``, ``install_profile``,
``run_profile``) take facts and never touch a device, so they are unit-tested
statically. ``doctor()`` gathers the facts (``nvidia-smi``, ``uv``, and an
``import torch`` that allocates nothing) and is read-only: it never installs.
``setup()`` installs the profile that the driver supports and checks that torch
sees the GPU; it uses only the standard library, so ``scripts/setup_env.py`` runs
it before TSFLab is installed.

torch 2.14.1 publishes CUDA wheels for cu126 and cu130 only (no cu128), so every
CUDA 12.6-12.9 driver uses ``cuda126`` and CUDA >= 13.0 uses ``cuda130``.
``UV_TORCH_BACKEND`` / ``--torch-backend`` apply to the ``uv pip`` interface only;
``uv sync`` always installs the locked build (cu130 on Linux).
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys

EXTRAS = ("models", "data", "experiments")

# name -> uv backend tag, minimum CUDA version the driver must report, and
# minimum Linux driver (informational).
INSTALL_PROFILES = {
    "cpu": {"backend": "cpu", "min_cuda": None, "min_driver": None},
    "cuda126": {"backend": "cu126", "min_cuda": (12, 6), "min_driver": 560},
    "cuda130": {"backend": "cu130", "min_cuda": (13, 0), "min_driver": 580},
}
RUN_PROFILES = {
    "laptop-cpu": "configs/execution/laptop-cpu.toml",
    "single-gpu": "configs/execution/single-gpu.toml",
    "multi-gpu": "configs/execution/multi-gpu.toml",
}
JOBS_PER_GPU = 2  # matches max_processes_per_gpu in the GPU run profiles

_CUDA_HEADER = re.compile(r"CUDA[A-Za-z ]*Version:\s*(\d+)\.(\d+)")


def parse_cuda_version(text: str | None) -> tuple[int, int] | None:
    """Parse ``12.8`` or an ``nvidia-smi`` header (``CUDA [UMD ]Version: 13.4``)."""
    if not text:
        return None
    match = _CUDA_HEADER.search(text) or re.fullmatch(
        r"\s*(\d+)\.(\d+)(?:\.\d+)?\s*", text
    )
    return (int(match.group(1)), int(match.group(2))) if match else None


def cuda_from_driver(driver: str | None) -> tuple[int, int] | None:
    """Fallback when the header lacks a CUDA version: Linux driver major -> CUDA."""
    try:
        major = int(str(driver).split(".")[0])
    except ValueError:
        return None
    for floor, cuda in ((580, (13, 0)), (575, (12, 9)), (570, (12, 8)),
                        (560, (12, 6)), (550, (12, 4))):
        if major >= floor:
            return cuda
    return (12, 0) if major >= 525 else None


def install_profile(cuda: tuple[int, int] | None, system: str = "Linux") -> str:
    """Map the driver's maximum CUDA version to a torch 2.14.1 install profile."""
    if system != "Linux" or cuda is None:
        return "cpu"  # macOS (MPS) and Windows use the PyPI build; no NVIDIA driver -> cpu
    if cuda >= INSTALL_PROFILES["cuda130"]["min_cuda"]:
        return "cuda130"
    if cuda >= INSTALL_PROFILES["cuda126"]["min_cuda"]:
        return "cuda126"
    return "cpu"  # driver older than CUDA 12.6 cannot run any torch 2.14.1 CUDA wheel


def install_commands(profile: str, system: str = "Linux", extras=EXTRAS) -> list[str]:
    """Exact uv commands for one install profile (run from the repository root)."""
    if profile == "cuda130" or (profile == "cpu" and system != "Linux"):
        flags = "".join(f" --extra {name}" for name in extras)
        return [f"uv sync --frozen --python 3.12{flags}"]
    backend = INSTALL_PROFILES[profile]["backend"]
    spec = f".[{','.join(extras)}]" if extras else "."
    return [
        "uv venv --python 3.12",
        f'uv pip install --torch-backend {backend} -e "{spec}" pytest',
        "export UV_NO_SYNC=1  # keep `uv run` from re-syncing the locked cu130 torch",
    ]


def run_profile(gpu_count: int) -> str:
    """Recommend the execution policy for the number of usable GPUs."""
    if gpu_count <= 0:
        return "laptop-cpu"
    return "single-gpu" if gpu_count == 1 else "multi-gpu"


def run_command(profile: str, gpu_count: int) -> str:
    command = f"uv run tsf run <experiment.toml> --policy {RUN_PROFILES[profile]}"
    if profile == "multi-gpu":
        gpus = ",".join(str(index) for index in range(gpu_count))
        command += f" --gpus {gpus} --jobs {gpu_count * JOBS_PER_GPU}"
    return command


def torch_build_tag(cuda: str | None) -> str:
    """``torch.version.cuda`` (``12.6``) -> uv backend tag (``cu126``)."""
    parsed = parse_cuda_version(cuda)
    return f"cu{parsed[0]}{parsed[1]}" if parsed else "cpu"


def _run(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _nvidia() -> dict:
    if shutil.which("nvidia-smi") is None:
        return {"driver": None, "cuda": None}
    header = _run(["nvidia-smi"]) or ""
    driver = (_run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
              or "").strip().splitlines()
    driver = driver[0].strip() if driver else None
    cuda = parse_cuda_version(header) or cuda_from_driver(driver)
    return {"driver": driver, "cuda": f"{cuda[0]}.{cuda[1]}" if cuda else None}


def _torch() -> dict:
    """Import torch only; queries availability but creates no tensor or model."""
    try:
        import torch
    except Exception as exc:  # missing or broken install
        return {"installed": False, "error": f"{type(exc).__name__}: {exc}"}
    facts = {
        "installed": True,
        "version": torch.__version__,
        "cuda": torch.version.cuda,
        "build": torch_build_tag(torch.version.cuda),
        "cuda_available": False,
        "device_count": 0,
        "mps_available": False,
    }
    try:
        facts["cuda_available"] = bool(torch.cuda.is_available())
        facts["device_count"] = torch.cuda.device_count() if facts["cuda_available"] else 0
        facts["mps_available"] = bool(torch.backends.mps.is_available())
    except Exception as exc:
        facts["error"] = f"{type(exc).__name__}: {exc}"
    return facts


def doctor() -> dict:
    """Read-only machine report with the matching install and run profiles."""
    from tsflab.experiments.infra.hardware import gpu_inventory

    system = platform.system()
    nvidia = _nvidia()
    gpus = gpu_inventory()
    uv = ((_run(["uv", "--version"]) or "").split() + [None, None])[1]  # "uv 0.9.10 (...)"
    torch = _torch()
    install = install_profile(parse_cuda_version(nvidia["cuda"]), system)
    usable = len(gpus) if install != "cpu" else 0
    run = run_profile(usable)
    expected = INSTALL_PROFILES[install]["backend"]
    warnings = []
    if not uv:
        warnings.append("uv not found: install it from https://docs.astral.sh/uv/")
    if nvidia["driver"] and install == "cpu" and system == "Linux":
        warnings.append(
            f"NVIDIA driver {nvidia['driver']} reports CUDA {nvidia['cuda']}; torch "
            "2.14.1 CUDA wheels need driver >= 560 (CUDA 12.6); upgrade the driver"
        )
    if torch.get("installed") and system == "Linux" and torch["build"] != expected:
        warnings.append(
            f"installed torch build {torch['build']} does not match profile "
            f"{install} ({expected}); reinstall with the commands below"
        )
    if torch.get("installed") and install != "cpu" and not torch["cuda_available"]:
        warnings.append("torch cannot use the GPU; check the build and driver")
    if not torch.get("installed"):
        warnings.append("torch is not importable; run the install commands below")
    return {
        "schema_version": 1,
        "ok": not warnings,
        "machine": {
            "os": f"{system} {platform.release()}",
            "arch": platform.machine(),
            "python": sys.version.split()[0],
            "uv": uv,
            "cpu_count": os.cpu_count(),
            "nvidia_driver": nvidia["driver"],
            "driver_cuda": nvidia["cuda"],
            "gpus": [f"{gpu['index']}: {gpu['name']}" for gpu in gpus],
        },
        "torch": torch,
        "install_profile": {
            "name": install,
            "backend": expected,
            "commands": install_commands(install, system),
        },
        "run_profile": {
            "name": run,
            "policy": RUN_PROFILES[run],
            "command": run_command(run, usable),
            "device": "cuda" if usable else ("mps" if torch.get("mps_available") else "cpu"),
        },
        "warnings": warnings,
    }


def setup(root: str = ".", extras=EXTRAS, dry_run: bool = False, run=subprocess.run) -> dict:
    """Install the torch build that this machine's driver supports, then verify it.

    Detects the driver's CUDA version, picks the install profile, and runs its uv
    commands in ``root``; a CUDA profile must end with torch seeing the GPU.
    """
    system = platform.system()
    nvidia = _nvidia()
    profile = install_profile(parse_cuda_version(nvidia["cuda"]), system)
    commands = [c for c in install_commands(profile, system, extras) if not c.startswith("export ")]
    report = {"driver": nvidia["driver"], "driver_cuda": nvidia["cuda"], "profile": profile,
              "commands": commands, "ok": True, "torch": None}
    if dry_run:
        return report
    for command in commands:
        if run(command, shell=True, cwd=root).returncode != 0:
            return {**report, "ok": False, "failed": command}
    python = os.path.join(root, ".venv", "bin", "python")
    probe = ("import json, torch; print(json.dumps({'version': torch.__version__, 'cuda': torch.version.cuda, "
             "'cuda_available': torch.cuda.is_available()}))")
    result = run([python, "-c", probe], cwd=root, capture_output=True, text=True)
    torch = None
    if result.returncode == 0 and result.stdout.strip():
        import json

        torch = json.loads(result.stdout.strip().splitlines()[-1])
    ok = torch is not None and (profile == "cpu" or torch["cuda_available"])
    return {**report, "torch": torch, "ok": ok,
            "no_sync": profile == "cuda126" or (profile == "cpu" and system == "Linux")}


def format_setup(report: dict) -> str:
    lines = [f"driver {report['driver'] or 'none'} (CUDA {report['driver_cuda'] or 'none'}) -> profile {report['profile']}"]
    lines += [f"  {command}" for command in report["commands"]]
    if report.get("failed"):
        lines.append(f"failed: {report['failed']}")
    elif report.get("torch"):
        torch = report["torch"]
        lines.append(f"torch {torch['version']} (CUDA {torch['cuda']}), cuda_available={torch['cuda_available']}")
    if report.get("no_sync"):
        lines.append("export UV_NO_SYNC=1  # `uv run` must not re-sync the locked cu130 torch on this machine")
    lines.append("ready" if report["ok"] else "not ready: see above, then `tsf env doctor`")
    return "\n".join(lines)


def format_doctor(report: dict) -> str:
    machine, torch = report["machine"], report["torch"]
    lines = [
        f"OS        {machine['os']} ({machine['arch']}), {machine['cpu_count']} CPUs",
        f"Python    {machine['python']}; uv {machine['uv'] or 'missing'}",
        f"NVIDIA    driver {machine['nvidia_driver'] or 'none'}, "
        f"CUDA {machine['driver_cuda'] or 'none'}, {len(machine['gpus'])} GPU(s)",
    ]
    lines += [f"          {gpu}" for gpu in machine["gpus"]]
    if torch.get("installed"):
        lines.append(
            f"torch     {torch['version']} ({torch['build']}), "
            f"cuda_available={torch['cuda_available']}, devices={torch['device_count']}"
        )
    else:
        lines.append("torch     not installed")
    install, run = report["install_profile"], report["run_profile"]
    lines += ["", f"Install profile: {install['name']} (uv torch backend {install['backend']})"]
    lines += [f"  {command}" for command in install["commands"]]
    lines += [
        "",
        f"Run profile: {run['name']} ({run['policy']}); set "
        f'[experiment.runtime] device = "{run["device"]}" in the run config',
        f"  {run['command']}",
    ]
    lines += [f"warning: {text}" for text in report["warnings"]]
    return "\n".join(lines)
