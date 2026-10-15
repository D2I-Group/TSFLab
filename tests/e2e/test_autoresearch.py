"""End to end: analyze -> compose x2 -> write configs -> smoke-run -> board -> register winner.

CPU, fixture dataset only (``dataset/smoke/smoke.csv``, 400 rows), one epoch per
run. Everything under ``tests/e2e`` is marked ``e2e`` (``pytest -m e2e``); it is
meant to take well under 60 s.
"""

from __future__ import annotations

import json
from pathlib import Path

from tsflab.cli.main import main

ROOT = Path(__file__).resolve().parents[2]

SPEC = """name = "{name}"
summary = "{name} recombination"
hypothesis = "smoke: {name} vs the RLinear baseline"
parents = ["RLinear"]
[slots]
normalization = ["component:revin"]
decomposition = ["{decomposition}"]
temporal = ["component:channel_wise_linear"]
loss = ["loss:mae"]
"""
BASELINE = """extends = ["{base}", "{data}", "{model}"]
[experiment]
description = "baseline RLinear"
work_dir = "{work}"
[experiment.runtime]
device = "cpu"
num_workers = 0
[task]
seq_len = 96
label_len = 0
pred_len = 12
[training]
epochs = 1
batch_size = 16
loss = "mae"
patience = 1
[model.params]
enc_in = 6
"""


def _records(work: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(work.rglob("records/*.json"))]


def test_autoresearch_chain(tmp_path, capsys):
    work = tmp_path / "work"
    # 1. profile the fixture dataset
    assert (
        main(["data", "analyze", "--path", "dataset/smoke/smoke.csv", "--out", str(tmp_path / "profile"), "--json"])
        == 0
    )
    profile = json.loads((tmp_path / "profile" / "profile.json").read_text())
    assert "recommendations" in profile and "slots" in profile["recommendations"]
    capsys.readouterr()

    # 2-3. compose two slot assignments and write runnable configs
    configs = []
    for name, decomposition in (("PlainRevLinear", "none"), ("SeasonalRevLinear", "component:series_decomposition")):
        spec = tmp_path / f"{name}.toml"
        spec.write_text(SPEC.format(name=name, decomposition=decomposition), encoding="utf-8")
        cfg = tmp_path / f"{name}_run.toml"
        assert (
            main(
                [
                    "model",
                    "compose",
                    str(spec),
                    "--write-config",
                    str(cfg),
                    "--dataset",
                    "configs/fixtures/smoke.toml",
                    "--enc-in",
                    "6",
                    "--seq-len",
                    "96",
                    "--pred-len",
                    "12",
                    "--smoke",
                    "--work-dir",
                    str(work),
                ]
            )
            == 0
        )
        configs.append(cfg)
    baseline = tmp_path / "baseline.toml"
    rel = lambda p: str(Path(ROOT / p).resolve())  # noqa: E731
    baseline.write_text(
        BASELINE.format(
            base=rel("configs/base.toml"),
            data=rel("configs/fixtures/smoke.toml"),
            model=rel("configs/models/RLinear.toml"),
            work=work,
        ),
        encoding="utf-8",
    )
    configs.append(baseline)

    # 4. smoke-run all three on CPU
    assert main(["run", "--smoke", "--config", *map(str, configs), "--jobs", "2"]) == 0

    # 5. read the records and the board
    records = _records(work)
    assert sorted(r["model"] for r in records) == ["Composed", "Composed", "RLinear"]
    capsys.readouterr()
    assert (
        main(
            [
                "result",
                "board",
                "--dataset",
                "weather",
                "--horizon",
                "12",
                "--top",
                "5",
                "--records",
                str(work),
                "--json",
            ]
        )
        == 0
    )
    rows = json.loads(capsys.readouterr().out)["horizons"][0]["rows"]
    local = [r for r in rows if r["source"] == "local"]
    assert {r["model"] for r in local} >= {"RLinear"} and len(local) == 3

    # 6. the winner is registered (dry run, then into a temp tree)
    winner = min((r for r in local if r["model"].startswith("Composed")), key=lambda r: r["mse"])
    winner_cfg = configs[0] if "none" in winner["model"] else configs[1]
    spec = tmp_path / ("PlainRevLinear.toml" if winner_cfg == configs[0] else "SeasonalRevLinear.toml")
    assert main(["model", "compose", str(spec), "--register", "WinnerRevLinear", "--dry-run"]) == 0
    assert not (ROOT / "src/tsflab/models/winnerrevlinear").exists()
    temp_repo = tmp_path / "repo"
    assert main(["model", "compose", str(spec), "--register", "WinnerRevLinear", "--root", str(temp_repo)]) == 0
    assert (temp_repo / "src/tsflab/models/winnerrevlinear/spec.py").is_file()
