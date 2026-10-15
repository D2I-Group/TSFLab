"""Optional shared-filesystem Slurm transport using stable parsable CLI fields."""

import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

from tsflab.experiments.infra.storage import file_lock, write_json


def _sbatch_args(root: Path, minutes, partition, account, gpus) -> list[str]:
    args = [
        "sbatch",
        "--parsable",
        "--time",
        str(minutes),
        "--chdir",
        str(Path.cwd()),
        "--output",
        str(root / "slurm-%j.log"),
    ]
    if partition:
        args += ["--partition", partition]
    if account:
        args += ["--account", account]
    if gpus:
        args += ["--gpus", str(gpus)]
    return args


def _submit(root: Path, record: Path, *, partition, account, gpus, minutes) -> dict:
    """Submit the prepared run or sweep as one Slurm job and persist its receipt."""
    if record.exists():
        raise ValueError(
            "this experiment already has a Slurm submission; inspect its status"
        )
    if (
        not (root / "sweep.json").exists()
        and not (root / "manifest.json").exists()
    ):
        raise ValueError("expected a prepared run or sweep directory")
    if minutes < 1 or gpus < 0:
        raise ValueError("minutes must be positive; gpus must be nonnegative")
    script = root / "slurm.sh"
    command = shlex.join(
        [sys.executable, "-m", "tsflab.cli.main", "run", "resume", str(root)]
    )
    script.write_text("#!/bin/sh\nset -eu\nexec " + command + "\n")
    args = _sbatch_args(root, minutes, partition, account, gpus)
    write_json(
        record,
        {
            "schema_version": 1,
            "status": "submission_pending",
            "directory": str(root),
        },
    )
    response = subprocess.run(
        [*args, str(script)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    if not re.fullmatch(r"\d+(;[\w.-]+)?", response):
        raise ValueError(
            f"unrecognized sbatch receipt: {response!r}; inspect cluster before retrying"
        )
    job, _, cluster = response.partition(";")
    state = {
        "schema_version": 1,
        "job_id": job,
        "cluster": cluster,
        "directory": str(root),
    }
    write_json(record, state)
    return state


def _job_status(state: dict, cluster_args: list[str]) -> dict:
    output = subprocess.run(
        [
            "sacct",
            *cluster_args,
            "--jobs",
            state["job_id"],
            "--noheader",
            "--parsable2",
            "--format=JobIDRaw,State,ExitCode",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    return {
        **state,
        "records": [
            dict(zip(("job_id", "state", "exit_code"), line.strip().split("|")[:3]))
            for line in output.splitlines()
            if line.strip()
        ],
    }


def slurm(directory, action, *, partition=None, account=None, gpus=0, minutes=60):
    root = Path(directory).resolve()
    with file_lock(root / ".slurm.lock"):
        record = root / "slurm.json"
        if action == "submit":
            return _submit(root, record, partition=partition, account=account, gpus=gpus, minutes=minutes)
        state = json.loads(record.read_text())
        if not state.get("job_id"):
            raise ValueError(
                "submission receipt is uncertain; reconcile the job ID with the cluster before retrying"
            )
        cluster_args = ["--clusters", state["cluster"]] if state["cluster"] else []
        if action == "cancel":
            subprocess.run(
                ["scancel", *cluster_args, state["job_id"]],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return {**state, "cancel_requested": True}
        if action != "status":
            raise ValueError("action must be submit, status, or cancel")
        return _job_status(state, cluster_args)
