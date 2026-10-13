"""Regression checks for cost accounting and blocking run waits."""
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from types import SimpleNamespace


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]


def test_resumed_usage_is_not_added_twice():
    analyze = load_script("cost_analyze", ROOT / "experiments/agent-cost/analyze.py")
    events = [(i, {"type": "turn.completed", "usage": {"input_tokens": n,
                "cached_input_tokens": n - 10, "output_tokens": 5}})
              for i, n in enumerate([100, 200, 200])]
    assert sum(analyze.analyze_codex(events, [])[-1].values()) == 205


def test_separate_conversations_are_counted_and_empty_actions_keep_usage(tmp_path):
    analyze = load_script("cost_analyze", ROOT / "experiments/agent-cost/analyze.py")
    events = []
    for chain, size in [("a", 100), ("b", 50), ("a", 200)]:
        events.extend([(0, {"type": "thread.started", "thread_id": chain}),
                       (1, {"type": "turn.completed", "usage": {"input_tokens": size, "output_tokens": 5}})])
    result = analyze.analyze_codex(events, [])
    assert sum(result[-1].values()) == 260
    assert sum(sum(v.values()) for v in result[0].values()) == 260
    with (tmp_path / "usage-events.jsonl").open("w") as f:
        for chain, size in [("a", 100), ("b", 50), ("a", 200), ("a", 200)]:
            f.write(json.dumps({"timestamp": "2026-10-10T00:00:00Z", "usage_chain": chain,
                "payload": {"type": "token_count", "info": {"total_token_usage": {
                    "input_tokens": size, "output_tokens": 5, "total_tokens": size + 5}}}}) + "\n")
    usage, stages = analyze.codex_rollout_usage(tmp_path, [])
    assert sum(usage.values()) == sum(stages.values()) == 260


def test_core_filter_keeps_scientific_work_and_mixed_commands():
    core = load_script("core_stats", ROOT / "experiments/agent-cost/core_stats.py")
    assert core.overhead("/bin/bash -lc 'nvidia-smi'") == "monitoring"
    assert core.overhead("uv run tsf run wait work_dirs/sweep --timeout 1800") is None
    assert core.overhead("uv run tsf run wait work_dirs/sweep --timeout 1800", "core-v1") == "monitoring"
    assert core.overhead("tsf run wait work_dirs/sweep --timeout 0") == "monitoring"
    assert core.overhead("tsf run wait work_dirs/sweep --timeout=0.0") == "monitoring"
    assert core.overhead("tsf run status work_dirs/sweep") == "monitoring"
    for cmd in ["tail -10 performance.csv", "nvidia-smi && python train.py",
                "python train.py", "cat results/results.json", "rg error src"]:
        assert core.overhead(cmd) is None


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


def test_search_session_stops_without_supervision_and_snapshots_evidence(tmp_path):
    harness = tmp_path / "harness"
    harness.mkdir()
    source = ROOT / "experiments/agent-cost"
    for name in ("run_session.sh", "session_control.py", "check_success.py", "render_prompt.py"):
        shutil.copy2(source / name, harness / name)
    shutil.copytree(source / "prompts", harness / "prompts")
    for name, body in {"make_env.sh": '#!/bin/bash\nmkdir -p "$2"\n',
                       "package.sh": '#!/bin/bash\nexit 0\n'}.items():
        path = harness / name
        path.write_text(body)
        path.chmod(0o755)
    bins = tmp_path / "bin"
    bins.mkdir()
    (bins / "nvidia-smi").write_text("#!/bin/bash\nexit 0\n")
    (bins / "nvidia-smi").chmod(0o755)
    agent = bins / "codex"
    agent.write_text('''#!/usr/bin/env python3
import json,sys
from pathlib import Path
if "--version" in sys.argv:
    print("fake-codex")
else:
    Path("results").mkdir(exist_ok=True)
    Path("results/results.json").write_text("[]")
    Path("results/report.md").write_text("No winner; stop normally.")
    print(json.dumps({"type":"thread.started","thread_id":"test"}))
    print(json.dumps({"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":5}}))
''')
    agent.chmod(0o755)
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"task": "autoresearch", "dataset": "Weather", "target_method": "Crossformer",
                               "pred_lens": [96], "split": "chronological", "budget": "8 GPU-hours"}))
    run = tmp_path / "run"
    env = {**os.environ, "PATH": f"{bins}:{os.environ['PATH']}", "AGENT": "codex", "TRAIN_GPU": "0",
           "DATA_DIR": str(tmp_path), "TSFLAB_COMMIT": "fixture", "CODEX_BASE_URL": "http://localhost/unused",
           "TIMEOUT_H": "0.01", "MAX_RESUMES": "5"}
    env.pop("MAX_BUDGET_USD", None)
    subprocess.run(["bash", str(harness / "run_session.sh"), str(task), "tsflab", str(run)],
                   env=env, check=True, timeout=15, capture_output=True, text=True)
    assert not (run / "stream.1.jsonl").exists()
    assert not (run / "interventions.tsv").exists()
    assert json.loads((run / "terminal.json").read_text())["reason"] == "finished_without_success"
    assert (run / "attempts/0/results.json").read_text() == "[]"
    assert json.loads((run / "limits.json").read_text())["max_resumes"] == 0


def test_search_budget_and_api_discovery():
    control = load_script("session_control", ROOT / "experiments/agent-cost/session_control.py")
    assert control.limits({"task": "autoresearch", "budget": "8 GPU-hours"}, 10, 5)["wall_seconds"] == 28800
    from tsflab.experiments.infra import api
    assert api.wait is not None
    for bad in (float("nan"), float("inf"), -1):
        with pytest.raises(ValueError):
            control.limits({"task": "reproduce"}, bad, 0)


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


def test_claude_resumed_cost_is_estimate_and_not_counted_twice(tmp_path):
    events = [{"t": i, "d": {"type": "result", "session_id": session,
               "total_cost_usd": cost}}
              for i, (session, cost) in enumerate([("a", 1), ("a", 2), ("b", 3)])]
    (tmp_path / "stream.0.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    result = subprocess.run([str(ROOT / ".venv/bin/python"),
        str(ROOT / "experiments/agent-cost/analyze.py"), str(tmp_path)],
        check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert report["cost_usd_reported"] == 5
    assert report["actual_billed_usd"] is None
    assert report["authentication_mode"] == "unknown"


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


def test_source_snapshot_is_hash_pinned(tmp_path):
    import hashlib, os, subprocess, tarfile
    from pathlib import Path
    source = tmp_path / 'source'; source.mkdir()
    (source/'AGENTS.md').write_text('frozen agent guide\n')
    archive = tmp_path / 'source.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(source/'AGENTS.md', arcname='AGENTS.md')
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ, TSFLAB_SRC=str(repo), TSFLAB_SNAPSHOT=str(archive),
               TSFLAB_SNAPSHOT_SHA256=hashlib.sha256(archive.read_bytes()).hexdigest(), SKIP_INSTALL='1')
    script = repo/'experiments/agent-cost/make_env.sh'
    output = tmp_path/'valid'
    result = subprocess.run(['bash',str(script),'tsflab',str(output),'HEAD'],env=env,capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    assert (output/'AGENTS.md').read_text() == 'frozen agent guide\n'
    env['TSFLAB_SNAPSHOT_SHA256'] = '0'*64
    result = subprocess.run(['bash',str(script),'tsflab',str(tmp_path/'invalid'),'HEAD'],env=env,capture_output=True,text=True)
    assert result.returncode != 0 and 'hash mismatch' in result.stderr


def test_review_gate_skips_healthy_reused_implementations():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1]/'experiments/agent-cost/review_gate.py'
    spec = importlib.util.spec_from_file_location('review_gate_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    valid = {'runs':[{'status':'succeeded','metrics':{'mse':1.0,'mae':.5}}]}
    assert not module.review_decision(valid)['review_required']
    assert module.review_decision(valid, implementation_changed=True)['review_required']
    assert module.review_decision({'cells':[{'status':'failed','mse':float('nan')}]})['review_required']


def test_claude_native_usage_prefers_deduplicated_result_over_partial_messages():
    analyze = load_script('native_claude_cost', ROOT / 'experiments/agent-cost/analyze.py')
    events = [(0, {'type':'assistant','message':{'id':'m1','usage':{'input_tokens':9,'output_tokens':2}}}),
              (1, {'type':'result','session_id':'s1','usage':{'input_tokens':10,'cache_read_input_tokens':90,'output_tokens':5}}),
              (2, {'type':'result','session_id':'s1','usage':{'input_tokens':10,'cache_read_input_tokens':90,'output_tokens':5}})]
    assert analyze.claude_native_usage(events) == {'input':10,'cache_read':90,'cache_write':0,'output':5}
    assert analyze.claude_native_usage([]) is None


def test_aggregate_uses_native_total_and_does_not_publish_estimated_discovery(tmp_path):
    aggregate = load_script('native_aggregate', ROOT / 'experiments/agent-cost/aggregate.py')
    (tmp_path/'metrics.json').write_text(json.dumps({'usage_total':{'input':10,'cache_read':90,'output':5},
        'tokens_by_action':{'read':{'input':9000000}},'discovery':{'token_share':.9}}))
    row = aggregate.row(tmp_path)
    assert row['tokens_m'] == 0.0  # 105 tokens rounded to three decimal million tokens.
    assert row['discovery_share'] is None
    (tmp_path/'metrics.json').write_text(json.dumps({'tokens_by_action':{'read':{'input':9000000}}}))
    assert aggregate.row(tmp_path)['tokens_m'] is None
    assert aggregate.row(tmp_path)['cost_usd_luna'] is None


def test_paper_summary_propagates_unknown_tokens_without_treating_as_zero():
    paper = load_script('native_paper_tables', ROOT / 'experiments/agent-cost/paper_tables.py')
    row = dict(completion=1.,valid=1.,errs=[],wall_h=.1,gpu_h=.1,tokens_m=None,cost=1.,turns=2,
        interventions=0,stage_tokens_m={k:None for k in paper.STAGES},produced=True)
    summary=paper.summarize([row])
    assert summary['tokens_m'] is None
    assert all(value is None for value in summary['stage_tokens_m'].values())


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


def test_claude_resumed_result_uuid_distinguishes_invocations():
    analyze=load_script('claude_resume_uuids',ROOT/'experiments/agent-cost/analyze.py')
    events=[(i,{'type':'result','session_id':'same','uuid':identity,'usage':{'input_tokens':10,'output_tokens':7}}) for i,identity in enumerate(['first','second','second'])]
    assert sum(analyze.claude_native_usage(events).values())==34
