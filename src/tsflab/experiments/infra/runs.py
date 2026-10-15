"""One run identity with immutable attempt history and recoverable local state."""

import json
import os
import secrets
import time
from dataclasses import asdict
from pathlib import Path

from tsflab.experiments.infra.fingerprints import (
    code_fingerprint,
    dataset_fingerprint,
    dependency_fingerprint,
)
from tsflab.experiments.infra.policy import ExecutionPolicy
from tsflab.experiments.infra.storage import (
    canonical_hash,
    file_lock,
    write_json,
)
from tsflab.experiments.infra.tracking import Tracker

TERMINAL = {"succeeded", "failed", "cancelled", "timed_out", "interrupted"}


def prepare_run(config, raw, sweep_keys=None, *, directory=None) -> Path:
    if directory:
        return Path(directory).resolve()
    run_id = f"{config.model.name}-{time.time_ns()}-{secrets.token_hex(3)}"
    directory = Path(config.experiment.work_dir).expanduser().resolve() / "_runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    snapshot = config.model_dump(mode="json")
    write_json(
        directory / "manifest.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "status": "queued",
            "stage": "queued",
            "config": snapshot,
            "config_sha256": canonical_hash(snapshot),
            "dependencies": dependency_fingerprint(),
            "data": dataset_fingerprint(config),
            "code_sha256": code_fingerprint(),
            "created_at": time.time(),
            "attempts": [],
            "raw": raw,
            "sweep_keys": sweep_keys or [],
        },
    )
    return directory


def read_run(directory) -> dict:
    return json.loads((Path(directory) / "manifest.json").read_text())


def verify_resume(directory, config) -> dict:
    state = read_run(directory)
    if state["config_sha256"] != canonical_hash(config.model_dump(mode="json")):
        raise ValueError("resume configuration changed; start a new run")
    if state["code_sha256"] != code_fingerprint():
        raise ValueError("resume source/dependency fingerprint changed; start a new run")
    if state.get("dependencies") != dependency_fingerprint():
        raise ValueError("resume installed scientific dependencies changed; restore the original environment")
    if state["data"] != dataset_fingerprint(config):
        raise ValueError("resume dataset content changed; start a new run")
    return state


class RunSession:
    def __init__(self, config, raw, sweep_keys=None):
        requested = os.environ.get("TSFLAB_RUN_DIR")
        self.directory = prepare_run(config, raw, sweep_keys, directory=requested)
        self.config = config
        self.policy = ExecutionPolicy.model_validate_json(os.environ.get("TSFLAB_POLICY", "{}"))
        self.state = read_run(self.directory)
        self.run_id = self.state["run_id"]
        self.resume = bool(self.state["attempts"])
        self.tracker = None

    def __enter__(self):
        self.lock = file_lock(self.directory / ".run.lock", blocking=False)
        self.lock.__enter__()
        try:
            self.state = verify_resume(self.directory, self.config)
            if self.state["status"] == "succeeded":
                raise ValueError("run already succeeded; use its existing result")
            from tsflab.experiments.utils.env import collect_env, collect_git

            managed_attempt = os.environ.get("TSFLAB_ATTEMPT")
            if managed_attempt:
                attempt = self.state["attempts"][-1]
                if attempt["number"] != int(managed_attempt):
                    raise ValueError("execution attempt identity mismatch")
                self.resume = attempt["number"] > 1
            else:
                attempt = {
                    "number": len(self.state["attempts"]) + 1,
                    "started_at": time.time(),
                }
                self.state["attempts"].append(attempt)
            attempt.update(
                pid=os.getpid(),
                environment={**collect_env(), **collect_git()},
                policy=self.policy.model_dump(mode="json"),
            )
            for key in ("error", "diagnosis"):
                self.state.pop(key, None)
            self.state["status"] = "running"
            self.stage("preflight")
            self.tracker = Tracker(
                self.directory,
                self.run_id,
                self.state["config"],
                self.policy.tracking,
                attempt["number"],
            )
            return self
        except BaseException:
            self.lock.__exit__(None, None, None)
            raise

    def stage(self, stage):
        self.state["stage"] = stage
        self.state["updated_at"] = time.time()
        write_json(self.directory / "manifest.json", self.state)

    def result(self, result):
        write_json(self.directory / "result.json", asdict(result))
        self.state["status"] = "succeeded"
        self.stage("completed")

    def __exit__(self, kind, error, traceback):
        try:
            if error:
                self.state["status"] = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
                self.state["error"] = f"{type(error).__name__}: {error}"
            attempt = self.state["attempts"][-1]
            attempt.update(
                status=self.state["status"],
                finished_at=time.time(),
                stage=self.state["stage"],
            )
            if error:
                attempt["error"] = self.state["error"]
            write_json(self.directory / "attempts" / f"{attempt['number']}.json", attempt)
            self.stage(self.state["stage"])
            if self.tracker:
                self.tracker.close(failed=error is not None)
        finally:
            self.lock.__exit__(kind, error, traceback)
