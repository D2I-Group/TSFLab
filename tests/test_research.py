"""tsflab.research: rounds, composition specs, slot pipelines, compose/register, and the result board."""

from __future__ import annotations

import json
import py_compile
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tsflab.cli.commands.result_board import board
from tsflab.models._slots.registry import OPTIONS, check_assignment, resolve_assignment
from tsflab.research.composition import compose_command, validate_composition
from tsflab.research.rounds import (
    ResearchRoundError,
    add_event,
    claim_run,
    create_round,
    events_for_run,
    finish_run,
    list_rounds,
    load_round,
    read_events,
    set_status,
    write_log,
    write_prompt,
)

# ---------------------------------------------------------------------------
# Research rounds
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_work_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TSFLAB_WORK_DIR", str(tmp_path / "work"))


def test_round_lifecycle_preserves_small_structured_memory() -> None:
    state = create_round(task="experiment", goal="Compare A against B", max_runs=2)
    round_id = state["id"]
    assert load_round(round_id)["runs_used"] == 0
    assert list_rounds()[0]["id"] == round_id

    add_event(round_id, "decision", "Use one representative horizon")
    first = claim_run(round_id, {"model": "Linear", "dataset": "synthetic"})
    finish_run(round_id, first, status="passed", run_id="run-one", metrics={"mse": 1.0})
    second = claim_run(round_id, {"model": "DLinear", "dataset": "synthetic"})
    finish_run(round_id, second, status="failed", error="fixture failure")

    with pytest.raises(ResearchRoundError, match="exhausted"):
        claim_run(round_id, {"model": "NLinear"})

    prompt = write_prompt(round_id, "Run the bounded comparison.\n")
    log = write_log(round_id, "config/unsafe", "complete output")
    second_log = write_log(round_id, "config/unsafe", "second output")
    assert prompt.read_text() == "Run the bounded comparison.\n"
    assert log.name == "config_unsafe.log"
    assert "complete output" in log.read_text()
    assert second_log != log
    assert second_log.read_text() == "second output"

    completed = set_status(round_id, "completed", "Evidence supports A")
    assert completed["status"] == "completed"
    events = read_events(round_id)
    assert {event["kind"] for event in events} >= {
        "hypothesis",
        "decision",
        "run",
        "failure",
        "conclusion",
    }
    assert not any("schema" in event for event in events)
    exported = events_for_run("run-one")
    assert len(exported) == len(events)
    assert all(event["round"] == round_id for event in exported)


def test_agent_task_start_writes_a_directly_readable_prompt(capsys) -> None:
    from tsflab.cli.commands.agent_tasks import agent_command

    code = agent_command(
        [
            "task",
            "start",
            "experiment",
            "--set",
            "question=Does normalization improve MSE?",
            "--set",
            "max_runs=2",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    round_id = payload["round"]["id"]
    assert payload["round"]["max_runs"] == 2
    assert payload["dispatch"] == "not-performed"
    assert payload["task"]["task"] == "experiment"
    assert "Does normalization improve MSE?" in payload["task"]["prompt"]
    assert load_round(round_id)["task"] == "experiment"


def test_sweep_association_counts_resolved_runs(monkeypatch) -> None:
    import importlib

    sweep_module = importlib.import_module("tsflab.experiments.runner.run_sweep")
    state = create_round(task="experiment", goal="Count resolved runs", max_runs=1)
    monkeypatch.setenv("TSFLAB_RESEARCH_ROUND", state["id"])
    loaded = SimpleNamespace(
        config_name="fixture",
        config=SimpleNamespace(
            model=SimpleNamespace(name="Linear"),
            dataset=SimpleNamespace(name="synthetic"),
            experiment=SimpleNamespace(random_seed=0),
            task=SimpleNamespace(pred_len=4),
        ),
        raw={},
        sweep_keys=[],
    )
    result = SimpleNamespace(run_id="run-1", metrics={"mse": 0.5})
    monkeypatch.setattr(sweep_module, "run_one", lambda *args: result)

    assert sweep_module.run_sweep([loaded]) == [result]
    assert load_round(state["id"])["runs_used"] == 1
    assert any(event.get("details", {}).get("run_id") == "run-1" for event in read_events(state["id"]))

    with pytest.raises(ResearchRoundError, match="exhausted"):
        sweep_module.run_sweep([loaded])


# ---------------------------------------------------------------------------
# Composition spec validation
# ---------------------------------------------------------------------------


ROOT = Path(__file__).resolve().parents[1]


def _profile_spec(**over):
    spec = {
        "name": "SeasonalRevLinear",
        "summary": "RevIN plus decomposition around a linear map.",
        "hypothesis": "strong-seasonality: period-aware split beats RLinear.",
        "parents": ["RLinear"],
        "slots": {
            "normalization": ["component:revin"],
            "decomposition": ["component:series_decomposition"],
            "temporal": ["component:channel_wise_linear"],
            "loss": ["loss:mae"],
        },
    }
    spec.update(over)
    return spec


def test_valid_composition_yields_scaffold_and_provenance():
    result = validate_composition(_profile_spec(), ROOT)
    assert result["ok"], result["errors"]
    assert result["components"] == ["revin", "series_decomposition", "channel_wise_linear"]
    assert "--components revin,series_decomposition,channel_wise_linear" in result["scaffold"]
    assert "RLinear" in result["card_provenance"]


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda s: s["slots"].update(temporal=["component:no_such_block"]), "unknown component"),
        (lambda s: s["slots"].update(loss=["loss:mae", "loss:mse"]), "one training loss"),
        (lambda s: s["slots"].update(channel=["model:NoSuchModel"]), "unknown donor"),
        (lambda s: s["slots"].update(bogus=["component:revin"]), "unknown slot"),
        (lambda s: s.update(name="RLinear"), "already exists"),
        (lambda s: s.update(parents=["Nope"]), "not in the catalog"),
        (lambda s: s["slots"].update(head=["revin"]), "prefix"),
        (
            lambda s: s.update(free_form=[{"slot": "temporal", "name": "A", "description": "d", "max_lines": 500}]),
            "max_lines",
        ),
        (lambda s: s["slots"].pop("temporal"), "temporal slot"),
    ],
)
def test_invalid_compositions_are_rejected(mutate, fragment):
    spec = _profile_spec()
    mutate(spec)
    result = validate_composition(spec, ROOT)
    assert not result["ok"]
    assert any(fragment in e for e in result["errors"]), result["errors"]


def test_free_form_block_fills_a_slot_within_budget():
    spec = _profile_spec(free_form=[{"slot": "temporal", "name": "PhaseGate", "description": "gate", "max_lines": 60}])
    del spec["slots"]["temporal"]
    result = validate_composition(spec, ROOT)
    assert result["ok"], result["errors"]
    assert result["free_form"] == ["PhaseGate"]


# ---------------------------------------------------------------------------
# Slot registry, pipelines, compose, register, and board
# ---------------------------------------------------------------------------


def _spec(**slots):
    base = {
        "normalization": ["component:revin"],
        "decomposition": ["component:series_decomposition"],
        "temporal": ["component:channel_wise_linear"],
        "loss": ["loss:mae"],
    }
    base.update(slots)
    return {
        "name": "SeasonalRevLinear",
        "summary": "RevIN plus decomposition around a linear map.",
        "hypothesis": "seasonal split beats RLinear.",
        "parents": ["RLinear"],
        "slots": base,
    }


def _assignment(**over):
    a = {
        "normalization": "revin",
        "decomposition": "none",
        "temporal": "linear",
        "channel": "independent",
        "head": "flatten_forecast_head",
        "loss": "mse",
    }
    a.update(over)
    return a


def test_every_adapter_component_is_cataloged():
    from tsflab.catalog.components import COMPONENT_CATALOG

    for options in OPTIONS.values():
        for option in options.values():
            assert set(option.components) <= set(COMPONENT_CATALOG.names()), option


@pytest.mark.parametrize(
    "over, fragment",
    [
        ({"decomposition": "wavelet"}, "not executable"),
        ({"temporal": "mixer_block"}, "set channel to mixing"),
        ({"channel": "mixing"}, "only provided by mixer_block"),
        ({"head": "gaussian_parameter_head", "loss": "nll_gaussian", "channel": "individual"}, "shared weights only"),
        ({"head": "quantile_head"}, "needs loss 'quantile'"),
        ({"loss": "quantile"}, "needs a quantile_head"),
    ],
)
def test_unsupported_combinations_fail_with_a_reason(over, fragment):
    problems = check_assignment(_assignment(**over))
    assert any(fragment in p for p in problems), problems


def test_resolution_accepts_aliases_defaults_and_rejects_sweeps():
    resolved, errors = resolve_assignment(
        {"temporal": ["component:patchtst"], "normalization": ["component:revin"], "channel": ["local:individual"]}
    )
    assert not errors and resolved["temporal"] == "tst_transformer" and resolved["head"] == "flatten_forecast_head"
    _, errors = resolve_assignment({"temporal": ["component:mamba", "component:mixer_block"]})
    assert any("exactly one option" in e for e in errors)


def test_compose_reports_executability():
    result = validate_composition(_spec(), ROOT)
    assert result["ok"] and result["executable"]["ok"], result
    assert result["executable"]["assignment"]["loss"] == "mae"
    bad = validate_composition(_spec(temporal=["component:mixer_block"]), ROOT)
    assert bad["ok"] and not bad["executable"]["ok"]


def test_write_config_is_a_runnable_config(tmp_path):
    spec_path = tmp_path / "spec.toml"
    spec_path.write_text(
        'name = "SeasonalRevLinear"\nsummary = "s"\nhypothesis = "h"\nparents = ["RLinear"]\n'
        '[slots]\nnormalization = ["component:revin"]\ntemporal = ["component:channel_wise_linear"]\nloss = ["loss:mae"]\n'
        "[params]\nhidden = 8\n",
        encoding="utf-8",
    )
    out = tmp_path / "cfg" / "run.toml"
    rc = compose_command(
        [
            str(spec_path),
            "--write-config",
            str(out),
            "--dataset",
            "configs/fixtures/smoke.toml",
            "--enc-in",
            "6",
            "--pred-len",
            "12",
            "--smoke",
            "--work-dir",
            str(tmp_path / "work"),
        ],
        ROOT,
    )
    assert rc == 0
    cfg = tomllib.loads(out.read_text())
    assert [p.split("/")[-1] for p in cfg["extends"]] == ["base.toml", "smoke.toml", "Composed.toml"]
    assert all((out.parent / p).resolve().is_file() for p in cfg["extends"])
    assert cfg["model"]["name"] == "Composed"
    assert cfg["model"]["params"]["temporal"] == "channel_wise_linear" and cfg["model"]["params"]["hidden"] == 8
    assert cfg["training"]["loss"] == "mae" and cfg["task"]["pred_len"] == 12
    params = cfg["model"]["params"]
    from tsflab.catalog.registry.models import MODEL_CATALOG

    assert MODEL_CATALOG.get("Composed").validate_params(params)["enc_in"] == 6


def test_write_config_refuses_probabilistic_head_without_register(tmp_path):
    spec_path = tmp_path / "s.toml"
    spec_path.write_text(
        'name = "QRev"\nsummary = "s"\nhypothesis = "h"\nparents = ["RLinear"]\n[slots]\n'
        'temporal = ["component:channel_wise_linear"]\nhead = ["component:quantile_head"]\nloss = ["loss:quantile"]\n',
        encoding="utf-8",
    )
    rc = compose_command(
        [
            str(spec_path),
            "--write-config",
            str(tmp_path / "r.toml"),
            "--dataset",
            "configs/fixtures/smoke.toml",
            "--enc-in",
            "6",
        ],
        ROOT,
    )
    assert rc == 1 and not (tmp_path / "r.toml").exists()


def test_register_scaffolds_a_package_and_dry_run_writes_nothing(tmp_path):
    spec_path = tmp_path / "spec.toml"
    spec_path.write_text(
        'name = "QRev"\nsummary = "quantile head on a linear map"\nhypothesis = "calibrated intervals"\n'
        'parents = ["RLinear"]\n[slots]\nnormalization = ["component:revin"]\ntemporal = ["component:channel_wise_linear"]\n'
        'head = ["component:quantile_head"]\nloss = ["loss:quantile"]\n',
        encoding="utf-8",
    )
    root = tmp_path / "repo"
    assert compose_command([str(spec_path), "--register", "QRevLinear", "--dry-run", "--root", str(root)], ROOT) == 0
    assert not root.exists()
    assert compose_command([str(spec_path), "--register", "QRevLinear", "--root", str(root)], ROOT) == 0
    package = root / "src/tsflab/models/qrevlinear"
    for name in ("__init__.py", "model.py", "spec.py"):
        py_compile.compile(str(package / name), doraise=True)
    assert "quantile-output" in (package / "spec.py").read_text()
    assert (root / "configs/models/QRevLinear.toml").is_file()
    from tsflab.catalog.cards.models import _composition_problems
    from tsflab.catalog.cards.store import load, problems, validated

    card = load(package)
    assert problems(card) == []
    facts = validated(card)
    assert facts.fidelity == "composed" and facts.admission.status == "pending"
    assert facts.composition["head"] == "component:quantile_head"
    composition = [f"{slot}={value}" for slot, value in facts.composition.items()]
    assert not _composition_problems(composition, "QRev")


def test_board_merges_committed_rows_with_local_records(tmp_path):
    board_file = tmp_path / "leaderboard.json"
    board_file.write_text(
        json.dumps(
            {
                "generated_at": "2026-01-01",
                "tracks": {
                    "time_series": {
                        "datasets": {
                            "Toy": {
                                "horizons": {
                                    "12": [
                                        {"model": "A", "mse": 0.5, "mae": 0.4, "n_runs": 1},
                                        {"model": "B", "mse": 0.3, "mae": 0.3, "n_runs": 1},
                                    ]
                                }
                            }
                        }
                    }
                },
            }
        )
    )
    records = tmp_path / "work" / "Toy" / "Composed" / "records"
    records.mkdir(parents=True)
    (records / "r1.json").write_text(
        json.dumps(
            {
                "record_id": "r1",
                "model": "Composed",
                "dataset_id": "Toy",
                "track": "time_series",
                "results": [{"horizon": 12, "metrics": {"mse": 0.2, "mae": 0.2}}],
                "config": {
                    "seq_len": 96,
                    "snapshot": {
                        "training": {"epochs": 1},
                        "model": {"params": {"temporal": "linear", "normalization": "revin"}},
                    },
                },
            }
        )
    )
    result = board("toy", horizon="12", top=3, metric="mse", records=[tmp_path / "work"], board_file=str(board_file))
    rows = result["horizons"][0]["rows"]
    assert [r["model"] for r in rows] == ["Composed[revin/-/linear/-]", "B", "A"]
    assert rows[0]["source"] == "local" and rows[0]["epochs"] == 1 and rows[1]["source"] == "board"


@pytest.mark.model  # runs the composition executor
def test_point_pipelines_cover_every_temporal_normalization_and_decomposition_option():
    import torch

    from tsflab.models._slots.adapters import Pipeline

    x = torch.randn(2, 48, 3)
    for temporal in ("linear", "channel_wise_linear", "tst_transformer", "mamba", "gated_dilated_conv"):
        for norm in ("none", "revin", "last_value_center"):
            for decomposition in ("none", "series_decomposition"):
                assignment = _assignment(normalization=norm, decomposition=decomposition, temporal=temporal)
                y = Pipeline(assignment, 3, 48, 12, patch_len=8, stride=4)(x)
                assert y.shape == (2, 12, 3) and torch.isfinite(y).all()
    y = Pipeline(_assignment(temporal="mixer_block", channel="mixing"), 3, 48, 12)(x)
    assert y.shape == (2, 12, 3)


@pytest.mark.model  # runs the composition executor
def test_probabilistic_pipelines_emit_their_output_kind():
    import torch

    from tsflab.models._slots.adapters import Pipeline

    x = torch.randn(2, 48, 3)
    q = Pipeline(_assignment(head="quantile_head", loss="quantile", normalization="last_value_center"), 3, 48, 12)(x)
    assert q.shape == (2, 12, 3, 9) and bool((q.diff(dim=-1) >= 0).all())
    g = Pipeline(_assignment(head="gaussian_parameter_head", loss="nll_gaussian"), 3, 48, 12)(x)
    assert g.shape == (2, 12, 3, 2) and bool((g[..., 1] > 0).all())


def test_params_schema_rejects_invalid_slot_combinations():
    from pydantic import ValidationError

    from tsflab.catalog.registry.models import MODEL_CATALOG

    spec = MODEL_CATALOG.get("Composed")
    with pytest.raises(ValidationError, match="mixer_block"):
        spec.validate_params({"enc_in": 3, "temporal": "mixer_block"})
    with pytest.raises(ValidationError, match="not executable"):
        spec.validate_params({"enc_in": 3, "decomposition": "wavelet"})


def test_params_schema_forbids_extras_and_explains_removed_options():
    from pydantic import ValidationError

    from tsflab.catalog.registry.models import MODEL_CATALOG

    schema = MODEL_CATALOG.get("Composed").params_schema
    with pytest.raises(ValidationError, match="Extra inputs"):
        schema.model_validate({"enc_in": 3, "bogus": 1})
    with pytest.raises(ValidationError, match="--register"):
        schema.model_validate({"enc_in": 3, "head": "quantile_head"})
    with pytest.raises(ValidationError, match="--register"):
        schema.model_validate({"enc_in": 3, "head": "gaussian_parameter_head"})
    with pytest.raises(ValidationError, match="not executable"):
        schema.model_validate({"enc_in": 3, "decomposition": "wavelet"})
