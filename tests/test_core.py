"""tsflab.core: repository paths, module hygiene, exported schemas, and leaderboard aggregation."""

from __future__ import annotations

import ast
from pathlib import Path

from tsflab.core.export import DEFAULT_SCHEMA_DIR, _check
from tsflab.core.leaderboard import aggregate, load_submissions, validate_submission
from tsflab.core.paths import is_packaged_root, repository_root, require_checkout

ROOT = Path(__file__).resolve().parents[1]


def test_repository_resources_resolve_to_the_checkout() -> None:
    assert repository_root() == ROOT
    assert not is_packaged_root()
    assert require_checkout("test") == ROOT


def test_python_modules_have_descriptions() -> None:
    missing = [
        str(path.relative_to(ROOT))
        for path in sorted((ROOT / "src").rglob("*.py"))
        if not ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert missing == []


def test_committed_json_schemas_match_the_models(capsys) -> None:
    assert _check(DEFAULT_SCHEMA_DIR) == 0
    assert "up to date" in capsys.readouterr().out


def test_schema_drift_is_reported(tmp_path, capsys) -> None:
    assert _check(tmp_path) == 1
    assert "missing: index.json" in capsys.readouterr().err


def _record(model: str, mse: float, seed: int = 0, horizon: int = 12) -> dict:
    return {
        "record_id": f"{model}-{seed}",
        "model": model,
        "dataset_id": "toy",
        "track": "time_series",
        "results": [{"horizon": horizon, "metrics": {"mse": mse, "mae": mse / 2}}],
    }


def test_leaderboard_averages_runs_and_ranks_by_the_primary_metric() -> None:
    tracks = aggregate(
        [
            _record("A", 0.4),
            _record("A", 0.6, seed=1),
            _record("B", 0.3),
            {"manifest": {"submission_id": "s1"}, "records": [_record("C", float("nan"))]},
        ]
    )
    rows = tracks["time_series"]["datasets"]["toy"]["horizons"]["12"]
    assert [(row["model"], row["rank"]) for row in rows] == [("B", 1), ("A", 2), ("C", 3)]
    a = rows[1]
    assert a["mse"] == 0.5 and a["n_runs"] == 2 and a["mse_std"] > 0
    assert a["submission_ids"] == ["A-0", "A-1"]
    assert rows[2]["mse"] is None  # non-finite metrics never rank above real ones


def test_submission_validation_requires_a_tsflab_binding(tmp_path) -> None:
    errors = validate_submission({"model": "A", "results": []})
    assert any("binding missing" in error for error in errors)
    assert len(errors) > 1  # schema violations are reported too
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "submission.json").write_text("{not json", encoding="utf-8")
    valid, rejected = load_submissions(tmp_path)
    assert valid == [] and "unreadable" in rejected[tmp_path / "bad" / "submission.json"][0]
