"""tsflab.agent: module map, generated index, task templates, assets, and tsf init projects."""

from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

import pytest

from tsflab.agent.modules import CHAIN, find_project, module_skills, parse_modules
from tsflab.agent.scaffold import (
    extras_for,
    init_project,
    render_agents_md,
    sync_command,
)
from tsflab.agent.tasks import (
    AgentTaskError,
    audit_tasks,
    list_tasks,
    load_task,
    render_task,
)
from tsflab.cli.main import main

# ---------------------------------------------------------------------------
# Module map and task templates
# ---------------------------------------------------------------------------


class ModuleMapTests(unittest.TestCase):
    def test_every_skill_and_task_has_exactly_one_module(self) -> None:
        from tsflab.agent import assets
        from tsflab.agent.assets import audit_module_map
        from tsflab.agent.modules import ALL_MODULES, CHAIN, owners

        on_disk = {path.parent.name for path in assets.SKILLS.glob("*/SKILL.md")}
        self.assertEqual(audit_module_map(on_disk), [])
        self.assertTrue(all(len(modules) == 1 for modules in owners("skills").values()))
        self.assertEqual(ALL_MODULES, (*CHAIN, "maintenance"))
        self.assertIn("unmapped-skill", " ".join(audit_module_map(on_disk | {"unmapped-skill"})))
        self.assertIn("ghost", " ".join(audit_module_map(on_disk - {"audit"} | {"ghost"})))

    def test_index_is_generated_from_the_map(self) -> None:
        from tsflab.agent import assets
        from tsflab.agent.index import render_index

        self.assertEqual(assets.INDEX.read_text(encoding="utf-8"), render_index(assets.ROOT / ".agents"))


class AgentTaskTests(unittest.TestCase):
    def test_catalog_is_valid_and_contains_bounded_workflows(self) -> None:
        self.assertEqual(audit_tasks(), [])
        self.assertEqual(
            {record["name"] for record in list_tasks()},
            {
                "autoresearch",
                "contribution",
                "experiment",
                "intake",
                "maintenance",
            },
        )

    def test_render_binds_inputs_and_preserves_boundaries(self) -> None:
        payload = render_task(
            "autoresearch",
            {"question": "Does RevIN improve PatchTST?", "max_runs": "4"},
        )
        self.assertIn("Does RevIN improve PatchTST?", payload["prompt"])
        self.assertIn("4 total runs", payload["prompt"])
        self.assertEqual(payload["budget"]["max_runs"], 4)
        self.assertEqual(payload["permissions"]["model_code"], "no-change-without-separate-authorization")
        self.assertEqual(payload["skills"], ["run-autoresearch"])

    def test_maintenance_supports_bounded_periodic_scans(self) -> None:
        payload = render_task("maintenance", {"mode": "curation"})
        self.assertIn("repository-wide repeated implementation scan", payload["prompt"])
        self.assertEqual(payload["budget"]["max_component_extractions"], 2)
        self.assertEqual(
            payload["permissions"]["repository"],
            "write-selected-models-components-cards-and-generated-projections",
        )

    def test_every_template_has_a_directly_renderable_demo(self) -> None:
        for record in list_tasks():
            task = load_task(record["name"])
            supplied = {
                key: "1" if "maximum" in spec else "demo"
                for key, spec in task["inputs"].items()
                if spec.get("required", False)
            }
            payload = render_task(record["name"], supplied)
            self.assertTrue(payload["prompt"])
            self.assertEqual(payload["task"], record["name"])
            self.assertTrue(payload["permissions"])
            self.assertTrue(payload["budget"])

    def test_missing_or_unknown_inputs_fail_closed(self) -> None:
        with self.assertRaisesRegex(AgentTaskError, "missing required"):
            render_task("contribution", {})
        with self.assertRaisesRegex(AgentTaskError, "unknown input"):
            render_task("intake", {"surprise": "write everything"})
        with self.assertRaisesRegex(AgentTaskError, "between 1 and 12"):
            render_task("autoresearch", {"question": "test", "max_runs": "13"})

    def test_intake_is_read_only_by_default_with_a_preimplementation_gate(self) -> None:
        payload = render_task("intake", {"paper_url": "https://arxiv.org/abs/1", "model_name": "Example"})
        self.assertIn("before writing code, confirm", payload["prompt"])
        self.assertIn("Read-only", payload["inputs"]["approval"])
        self.assertEqual(payload["budget"]["max_models"], 1)

    def test_contribution_defaults_to_draft_only(self) -> None:
        payload = render_task("contribution", {"target": "#12"})
        self.assertIn("#12", payload["prompt"])
        self.assertIn("Draft only", payload["inputs"]["authorization"])
        self.assertEqual(payload["permissions"]["external_actions"], "none-unless-authorized")

    def test_numeric_inputs_narrow_machine_readable_budgets(self) -> None:
        cases = [
            ("experiment", {"question": "test", "max_runs": "2"}, "max_runs", 2),
            (
                "experiment",
                {
                    "question": "reproduce the reported primary table",
                    "paper_url": "https://arxiv.org/abs/1",
                    "max_runs": "3",
                },
                "max_runs",
                3,
            ),
            ("intake", {"candidate_limit": "4"}, "max_candidates", 4),
            ("maintenance", {"batch_size": "1"}, "max_models", 1),
        ]
        for name, supplied, key, expected in cases:
            with self.subTest(name=name):
                self.assertEqual(render_task(name, supplied)["budget"][key], expected)

    def test_cli_routes_task_validation(self) -> None:
        self.assertEqual(main(["agent", "task", "validate"]), 0)


# ---------------------------------------------------------------------------
# Module selection and tsf init
# ---------------------------------------------------------------------------


COMMANDS = {
    "catalog",
    "data",
    "model",
    "run",
    "env",
    "result",
    "realtime",
    "research",
    "repo",
    "agent",
    "init",
}


def test_module_selection_and_extras() -> None:
    assert parse_modules(None) == list(CHAIN)
    assert parse_modules("models,data") == ["data", "models"]
    with pytest.raises(ValueError):
        parse_modules("data,bogus")
    assert extras_for(list(CHAIN)) == ["all"]
    assert extras_for(["data", "release"]) == ["data", "hub", "realtime"]
    assert "audit" in module_skills(["models"], {"intake": ["audit"]})


def test_maintenance_is_opt_in() -> None:
    assert "maintenance" not in parse_modules(None)
    assert parse_modules("maintenance,data") == ["data", "maintenance"]
    assert extras_for([*CHAIN, "maintenance"]) == ["all"]
    assert extras_for(["maintenance"]) == []


def test_generated_agents_md_fits_the_context_budget() -> None:
    assert len(render_agents_md("p", list(CHAIN)).splitlines()) <= 50
    assert len(render_agents_md("p", [*CHAIN, "maintenance"]).splitlines()) <= 50


def test_init_scaffolds_selected_modules(tmp_path, monkeypatch) -> None:
    project = tmp_path / "proj"
    init_project(project, modules=["data", "autoresearch"])
    table = tomllib.loads((project / "pyproject.toml").read_text())
    assert table["tool"]["tsflab"]["modules"] == ["data", "autoresearch"]
    assert table["project"]["dependencies"] == ["tsflab[data,autoresearch]"]
    skills = {p.name for p in (project / ".agents" / "skills").iterdir()}
    assert skills == {"add-dataset", "inspect-dataset", "run-autoresearch"}
    assert (project / ".agents" / "tasks" / "autoresearch.toml").is_file()
    index = (project / ".agents" / "README.md").read_text()
    assert "run-autoresearch" in index and "add-model" not in index and "audit" not in index
    assert not (project / ".agents" / "tasks" / "experiment.toml").exists()
    agents = (project / "AGENTS.md").read_text()
    assert "tsf catalog" in agents and "run-autoresearch" in agents and "add-model" not in agents
    assert (project / "CLAUDE.md").read_text() == agents
    assert "tsflab://" in (project / "configs/runs/example.toml").read_text()
    assert find_project(project / "configs") == (project.resolve(), ["data", "autoresearch"])
    with pytest.raises(FileExistsError):
        init_project(project)

    monkeypatch.chdir(project)
    (project / "AGENTS.md").write_text(agents + "my note\n")
    assert sync_command(["--add", "models"]) == 0
    refreshed = (project / "AGENTS.md").read_text()
    assert "my note" in refreshed and "add-model" in refreshed
    assert (project / ".agents" / "tasks" / "intake.toml").is_file()
    assert tomllib.loads((project / "pyproject.toml").read_text())["tool"]["tsflab"]["modules"] == [
        "data",
        "models",
        "autoresearch",
    ]


# ---------------------------------------------------------------------------
# tsf init inherits packaged configs
# ---------------------------------------------------------------------------


def test_init_project_scaffolds_and_inherits_package_configs(tmp_path: Path) -> None:
    from tsflab.experiments.config.loader import _resolve_extends

    written = init_project(tmp_path / "proj")
    assert (tmp_path / "proj" / "configs" / "runs" / "example.toml") in written
    with pytest.raises(FileExistsError):
        init_project(tmp_path / "proj")
    import tomllib

    run = tmp_path / "proj" / "configs" / "runs" / "example.toml"
    merged = _resolve_extends(tomllib.loads(run.read_text()), str(run.parent))
    assert merged["dataset"]["name"] == "ETTh1"
    assert merged["experiment"]["description"] == "proj: DLinear on ETTh1"


def test_agent_assets_are_canonical() -> None:
    from tsflab.agent.assets import audit_agent_assets

    assert audit_agent_assets() == []
