"""End to end: real subprocess sweeps, timeouts, cancellation, and queue recovery.

Each test trains the CRIB smoke configuration (``configs/runs/smoke_crib.toml``)
in a child process on the CPU fixture data, so the module lives under
``tests/e2e`` and runs only with ``-m e2e``.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tsflab.experiments.infra.policy import ExecutionPolicy


def test_real_subprocess_sweep_and_resume_skip_completed(tmp_path):
    from tsflab.experiments.config.loader import load_config
    from tsflab.experiments.infra.execution import execute, prepare_sweep, status
    from tsflab.experiments.infra.runs import read_run

    loaded = load_config("configs/runs/smoke_crib.toml")
    loaded[0].config.experiment.work_dir = str(tmp_path)
    policy = ExecutionPolicy()
    directory = prepare_sweep(loaded, policy)
    result = execute(directory, policy)
    assert result["ok"], status(directory)
    path = Path(result["runs"][0]["directory"])
    assert (path / "checkpoints" / "latest.pth").is_file()
    assert (path / "attempt-1.log").is_file()
    saved = read_run(path)
    record = json.loads(
        next((tmp_path / "weather" / "CRIB" / "records").glob("*.json")).read_text()
    )
    assert record["config"]["snapshot"] and record["config"]["config_sha256"]
    resumed = execute(directory)
    assert resumed["ok"] and resumed["skipped"] == 1
    assert len(read_run(path)["attempts"]) == len(saved["attempts"]) == 1


def test_real_process_timeout_preserves_failure_and_can_resume(tmp_path):
    from tsflab.experiments.config.loader import load_config
    from tsflab.experiments.infra.execution import execute, prepare_sweep
    from tsflab.experiments.infra.runs import read_run

    loaded = load_config("configs/runs/smoke_crib.toml")
    loaded[0].config.experiment.work_dir = str(tmp_path)
    policy = ExecutionPolicy.model_validate({"budget": {"run_timeout_minutes": 0.001}})
    directory = prepare_sweep(loaded, policy)
    result = execute(directory, policy)
    assert not result["ok"] and result["runs"][0]["status"] == "timed_out"
    result = execute(directory, ExecutionPolicy())
    assert result["ok"]
    assert read_run(result["runs"][0]["directory"])["status"] == "succeeded"


def test_real_cancellation_and_single_run_resume_preserve_sweep_context(tmp_path):
    import time

    from tsflab.experiments.config.loader import load_config
    from tsflab.experiments.infra.execution import cancel, execute, prepare_sweep
    from tsflab.experiments.infra.runs import read_run

    loaded = load_config("configs/runs/smoke_crib.toml")
    loaded[0].config.experiment.work_dir = str(tmp_path)
    directory = prepare_sweep(loaded, ExecutionPolicy())
    run_path = Path(json.loads((directory / "sweep.json").read_text())["runs"][0])
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(execute, directory)
        deadline = time.monotonic() + 15
        while not (run_path / "runtime.json").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (run_path / "runtime.json").exists()
        cancel(run_path)
        result = future.result(timeout=20)
    assert result["runs"][0]["status"] == "cancelled"
    resumed = execute(run_path)
    assert resumed["directory"] == str(directory)
    assert resumed["ok"]
    assert len(read_run(run_path)["attempts"]) == 2


def test_local_queue_recovers_after_controller_crash(tmp_path):
    import os
    import signal
    import time

    from tsflab.experiments.config.loader import load_config
    from tsflab.experiments.infra.execution import prepare_sweep
    from tsflab.experiments.infra.queue import enqueue, jobs, work

    loaded = load_config("configs/runs/smoke_crib.toml")
    loaded[0].config.experiment.work_dir = str(tmp_path / "outputs")
    sweep = prepare_sweep(loaded, ExecutionPolicy())
    queue = tmp_path / "queue"
    enqueue(queue, sweep)
    work(queue, once=True)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        state = jobs(queue)[0]
        if state["status"] == "running":
            break
        time.sleep(0.05)
    assert state["status"] == "running"
    os.kill(state["pid"], signal.SIGKILL)
    time.sleep(0.3)
    work(queue, once=True)
    while time.monotonic() < deadline:
        state = jobs(queue)[0]
        if state["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.1)
    assert state["status"] == "succeeded", state


def test_validation_split_run_reports_no_test_metrics(tmp_path, capsys):
    from tsflab.experiments.config.loader import load_config
    from tsflab.experiments.runner.run_one import run_one

    loaded = load_config("configs/runs/smoke_crib.toml")[0]
    loaded.config.experiment.work_dir = str(tmp_path)
    loaded.config.evaluation.split = "val"
    run_one(loaded.config, loaded.raw)
    out = capsys.readouterr().out
    assert "Validation metrics |" in out and "Test metrics" not in out
    model_dir = tmp_path / "weather" / "CRIB"
    assert "split" in (model_dir / "performance.csv").read_text().splitlines()[0]
    assert not list(model_dir.glob("records/*.json"))
