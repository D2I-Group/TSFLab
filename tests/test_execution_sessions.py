"""Blocking run waits, sweep resume, evaluation splits, and submission checks."""
import json
from pathlib import Path

import pytest
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def test_wait_blocks_until_sweep_finishes(monkeypatch):
    from tsflab.experiments.infra import execution
    states = iter([{"runs": [{"status": "running"}, {"status": "succeeded"}]},
                   {"runs": [{"status": "succeeded"}, {"status": "succeeded"}]}])
    sleeps = []
    monkeypatch.setattr(execution, "status", lambda _: next(states))
    result = execution.wait("unused", timeout=60, sleep=sleeps.append)
    assert result["ok"] and result["done"]
    assert len(sleeps) == 1


def test_wait_timeout_does_not_cancel(monkeypatch):
    from tsflab.experiments.infra import execution
    monkeypatch.setattr(execution, "status", lambda _: {"status": "running"})
    monkeypatch.setattr(execution, "cancel", lambda _: pytest.fail("must not cancel"))
    assert not execution.wait("unused", timeout=0)["done"]
    monkeypatch.setattr(execution, "status", lambda _: {"status": "failed"})
    result = execution.wait("unused")
    assert not result["ok"] and result["done"]
    with pytest.raises(ValueError, match="timeout"):
        execution.wait("unused", timeout=float("nan"))


def test_wait_returns_saved_metrics_without_manifest_or_log_dump(tmp_path):
    from tsflab.experiments.infra.execution import wait
    (tmp_path / "manifest.json").write_text(json.dumps({
        "run_id": "r1", "status": "succeeded", "raw": {"long": "private context"},
        "config": {"model": {"name": "DLinear"}, "dataset": {"name": "Weather"},
                   "evaluation": {"split": "val"}, "task": {"pred_len": 96}}}))
    (tmp_path / "result.json").write_text(json.dumps({
        "metrics": {"mse": 0.2, "mae": 0.3}, "train_time_sec": 12, "test_time_sec": 1, "n_windows": 75,
        "checkpoint_path": "large-artifact"}))
    report = wait(tmp_path)
    assert report["ok"]
    row = report["results"][0]
    assert (row["mse"], row["mae"]) == (0.2, 0.3)
    assert row["model"] == "DLinear" and row["split"] == "val" and row["train_s"] == 12
    assert "raw" not in row and "config" not in row and "checkpoint_path" not in row


def test_wait_reports_missing_artifact_and_does_not_finish_during_retry(tmp_path, monkeypatch):
    from tsflab.experiments.infra import execution
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "succeeded"}))
    report = execution.wait(tmp_path)
    assert not report["ok"] and report["unfinished"][0]["error"] == "completed run has no result artifact"
    active = iter([True, False])
    monkeypatch.setattr(execution, "controller_active", lambda _: next(active))
    sleeps = []
    assert not execution.wait(tmp_path, sleep=sleeps.append)["ok"]
    assert len(sleeps) == 1


def test_completion_identifies_search_candidates_and_frozen_source():
    from tsflab.experiments.infra.execution import completion_report
    rows = []
    for lr, seed in ((0.001, 42), (0.0003, 43)):
        rows.append(completion_report({
            "status": "failed", "error": "budget exhausted",
            "sweep_keys": ["training.optimizer.lr", "experiment.random_seed", "model.params.hidden_dim"],
            "config": {"training": {"optimizer": {"lr": lr}},
                       "experiment": {"random_seed": seed},
                       "model": {"name": "StateFlow", "params": {"hidden_dim": 32}},
                       "evaluation": {"checkpoint_path": "/frozen/best.pth", "checkpoint_sha256": "abc"}}}))
    assert [r["learning_rate"] for r in rows] == [0.001, 0.0003]
    assert [r["parameters"]["experiment.random_seed"] for r in rows] == [42, 43]
    assert all(r["parameters"]["model.params.hidden_dim"] == 32 for r in rows)
    assert all(r["source_checkpoint_sha256"] == "abc" for r in rows)
    assert all(r["source_checkpoint"] == "/frozen/best.pth" and r["error"] for r in rows)


def test_completed_sweep_resume_returns_all_saved_metrics(tmp_path, monkeypatch):
    from tsflab.experiments.infra import execution
    from tsflab.experiments.infra.policy import ExecutionPolicy
    runs = []
    for i in range(2):
        path = tmp_path / f"run{i}"
        path.mkdir()
        (path / "manifest.json").write_text(json.dumps({"config": {}, "status": "succeeded"}))
        (path / "result.json").write_text(json.dumps({"metrics": {"mse": i + 0.1}}))
        runs.append(str(path))
    (tmp_path / "sweep.json").write_text(json.dumps({"runs": runs, "policy": {}}))
    monkeypatch.setattr(execution, "validated_config", lambda _: object())
    monkeypatch.setattr(execution, "verify_resume", lambda path, _: execution.read_run(path))
    monkeypatch.setattr(execution, "preflight", lambda *a: {
        "ok": True, "resolved_policy": ExecutionPolicy().model_dump(mode="json")})
    report = execution.execute(tmp_path)
    assert report["ok"] and report["skipped"] == 2
    assert [r["metrics"]["mse"] for r in report["runs"]] == [0.1, 1.1]


@pytest.mark.parametrize("extra,expected", [([], "val"), (["--eval-split", "test"], "test")])
def test_search_split_applies_to_resolved_and_raw_config(monkeypatch, tmp_path, extra, expected):
    from tsflab.cli.commands.execution import run_command
    from tsflab.experiments.config import loader
    from tsflab.experiments.infra import execution
    config = tmp_path / "run.toml"
    config.write_text("")
    item = SimpleNamespace(config=SimpleNamespace(evaluation=SimpleNamespace(split="test")), raw={})
    monkeypatch.setenv("TSFLAB_EVAL_SPLIT", "val")
    monkeypatch.setattr(loader, "load_config", lambda _: [item])
    monkeypatch.setattr(execution, "preflight", lambda *a: {"ok": False})
    assert run_command([str(config), "--json", *extra]) == 1
    assert item.config.evaluation.split == expected
    assert item.raw["evaluation"]["split"] == expected


def test_validation_schema_and_loaders_never_construct_test(monkeypatch):
    from tsflab.experiments.config.schema.evaluation import EvaluationConfig
    from tsflab.experiments.config.loader import load_config
    from tsflab.experiments.runner import run_one as runner
    assert EvaluationConfig(split="val").split == "val"
    config = load_config(str(ROOT / "configs/runs/smoke_mtlinear.toml"))[0].config
    config.evaluation.split = "val"
    flags = []
    def build(*args, **kwargs):
        assert kwargs["drop_last"] == config.training.drop_last
        flag = args[4]
        assert flag != "test", "search must not construct test data"
        flags.append(flag)
        return object(), object()
    monkeypatch.setattr(runner, "build_data_loader", build)
    result = runner._build_loaders(config)
    assert flags == ["train", "val"]
    assert result[2] is result[4] and result[3] is result[5]


def test_submission_rejects_validation_results(tmp_path):
    from tsflab.cli.commands.submit import build_submission
    record = tmp_path / "record.json"
    record.write_text(json.dumps({"config": {"snapshot": {"evaluation": {"split": "val"}}}}))
    with pytest.raises(ValueError, match="validation-only"):
        build_submission(record, submitter="smoke", out_root=tmp_path / "submissions")


def test_validation_loss_weights_the_final_partial_batch(monkeypatch):
    import torch
    from tsflab.experiments.runner import trainer
    monkeypatch.setattr(trainer, "make_decoder_input", lambda *args: None)
    monkeypatch.setattr(trainer, "call_forecaster", lambda model, x, *args: model(x))
    monkeypatch.setattr(trainer, "slice_prediction_target", lambda x, y, *args: (x, y))
    batches = [(torch.zeros(2, 1, 1), torch.zeros(2, 1, 1), None, None),
               (torch.zeros(1, 1, 1), torch.full((1, 1, 1), 3.0), None, None)]
    value = trainer.validate(torch.nn.Identity(), batches, torch.device("cpu"),
                             torch.nn.MSELoss(), 0, 1, "M")
    assert value == pytest.approx(3.0)


def test_validation_board_excludes_test_records_and_published_board(monkeypatch):
    from tsflab.cli.commands import result_board as module
    docs = [(Path("val"), {"dataset_id": "smoke", "model": "DLinear",
        "config": {"snapshot": {"evaluation": {"split": "val"}}}, "results": [{"horizon": 6}]}),
        (Path("test"), {"dataset_id": "smoke", "model": "DLinear",
        "config": {"snapshot": {"evaluation": {"split": "test"}}}, "results": [{"horizon": 6}]})]
    monkeypatch.setattr(module, "local_runs", lambda *a: docs)
    seen = []
    monkeypatch.setattr(module, "aggregate", lambda rows: seen.extend(rows) or {})
    monkeypatch.setattr(module, "_board_path", lambda *a: pytest.fail("validation must not fetch test board"))
    assert module.board("smoke", horizon=None, top=5, metric="mse", records=[],
                        board_file=None, split="val")["split"] == "val"
    assert len(seen) == 1 and seen[0]["config"]["snapshot"]["evaluation"]["split"] == "val"


def test_compact_preflight_keeps_failures_and_coverage(monkeypatch, tmp_path, capsys):
    from tsflab.cli.commands.execution import run_command
    from tsflab.experiments.config import loader
    from tsflab.experiments.infra import execution
    path = tmp_path / "run.toml"
    path.write_text("")
    monkeypatch.setattr(loader, "load_config", lambda _: [SimpleNamespace(config=object())])
    monkeypatch.delenv("TSFLAB_EVAL_SPLIT", raising=False)
    failure = {"name": "data", "status": "failed", "detail": "missing dataset"}
    monkeypatch.setattr(execution, "preflight", lambda *args: {"schema_version": 1,
        "ok": False, "total_runs": 1, "budget_ok": True, "resolved_policy": {"large": "dump"},
        "runs": [{"ok": False, "errors": ["missing dataset"]}],
        "environments": [{"ok": False, "checks": [
            {"name": "torch", "status": "passed", "detail": "version"}, failure]}]})
    output = tmp_path / "results" / "completion.json"
    assert run_command([str(path), "--dry-run", "--json", "--compact", "--output", str(output)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert json.loads(output.read_text()) == result
    assert result["total_runs"] == 1 and not result["ok"]
    assert result["environments"][0]["checks"] == [failure]
    assert "resolved_policy" not in result and "errors" in result["runs"][0]


def test_duplicate_cli_launch_waits_for_owner_and_preserves_conflicting_output(tmp_path,monkeypatch,capsys):
    from tsflab.cli.commands.execution import run_command
    from tsflab.experiments.config import loader
    from tsflab.experiments.infra import execution
    config=tmp_path/'run.toml';config.write_text('')
    output=tmp_path/'completion.json';output.write_text('{"original":true}')
    monkeypatch.delenv('TSFLAB_EVAL_SPLIT',raising=False)
    monkeypatch.setattr(loader,'load_config',lambda _:[SimpleNamespace(config=object())])
    monkeypatch.setattr(execution,'preflight',lambda *a:{'ok':True})
    keys=[]
    def prepare(*a,**kw):keys.append(kw['request_key']);return tmp_path/'sweep'
    monkeypatch.setattr(execution,'prepare_sweep',prepare)
    monkeypatch.setattr(execution,'execute',lambda *a,**k:(_ for _ in ()).throw(BlockingIOError()))
    monkeypatch.setattr(execution,'wait',lambda directory,timeout:{'ok':True,'directory':str(directory),'runs':[]})
    assert run_command([str(config),'--json','--output',str(output)])==0
    assert keys==[str(output.resolve())]
    capsys.readouterr();original=output.read_text()
    monkeypatch.setattr(execution,'prepare_sweep',lambda *a,**k:(_ for _ in ()).throw(execution.RequestConflictError('different matrix')))
    assert run_command([str(config),'--json','--output',str(output)])==2
    assert output.read_text()==original
