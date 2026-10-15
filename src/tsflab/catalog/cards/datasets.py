"""Dataset cards: curated descriptive facts plus a generated runtime block.

A dataset card mirrors a model card. Everything a person had to research
(domain, source, license, statistics, protocol, pitfalls) is curated prose and
front matter that this module never rewrites. Everything derivable from the
executable preset (loader, config, task modes, parameters, files) lives in one
marked block and in a few front-matter keys that are regenerated.

Front matter is flat, one ``key: <JSON value>`` pair per line, so cards stay
greppable and an agent can read level 0/1 facts without the body. Cards for a
family of series sharing a loader (GIFT-Eval) get one family card as well.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

from tsflab.catalog.cards.schema import BENCHMARKS, DOMAINS, REDISTRIBUTION  # noqa: F401  (public names)

#: Keys every preset card must carry; ``OPTIONAL_KEYS`` may be omitted.
#: ``protocol`` is the TSFLab protocol; ``literature_protocol`` is context only.
OPTIONAL_KEYS = ("realtime_track", "literature_protocol", "characteristics", "characteristics_basis")
STATS_BASIS = ("measured", "source-reported", "mixed")
CHANNEL_KINDS = ("channels", "nodes", "series", "stations")
TODO = "TODO"


@dataclass(frozen=True)
class DatasetRecord:
    """Resolved metadata for one runnable dataset configuration preset."""

    name: str
    config: str
    loader: str
    alias: str
    path: str
    dataset_id: str
    params: dict[str, object]
    task: dict[str, object]
    track: str
    task_modes: tuple[str, ...]
    #: True when configs/hub/datasets.json pins at least one file for this preset.
    published: bool = False


def _task_modes_for_loader(loader: str) -> tuple[str, ...]:
    """Resolve modes from the executable registry instead of duplicating them."""
    from tsflab.catalog.registry.datasets import (
        DATASET_REGISTRY,
        ordered_task_modes,
        register_dataset_by_name,
    )

    register_dataset_by_name(loader)
    return ordered_task_modes(DATASET_REGISTRY.get(loader).task_modes)


def _is_published(root: Path, name: str, manifest: dict) -> bool:
    """Whether the pinned Hub manifest holds files for preset ``name``."""
    from tsflab.release.hub.datasets import published_files

    try:
        return bool(published_files(name, manifest, root))
    except (FileNotFoundError, ValueError, KeyError):
        return False


def dataset_records(root: Path) -> tuple[DatasetRecord, ...]:
    """Read every dataset preset without importing its runtime dependencies."""
    config_root = root / "configs" / "datasets"
    records: list[DatasetRecord] = []
    manifest_file = root / "configs" / "hub" / "datasets.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.is_file() else {"files": {}}
    for path in sorted(config_root.rglob("*.toml")):
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        dataset = payload.get("dataset", {})
        relative = path.relative_to(config_root)
        name = relative.with_suffix("").as_posix()
        records.append(
            DatasetRecord(
                name=name,
                config=path.relative_to(root).as_posix(),
                loader=str(dataset.get("name", "")),
                alias=str(dataset.get("alias", name)),
                path=str(dataset.get("path", "")),
                dataset_id=str(dataset.get("id", "")),
                params=dict(dataset.get("params", {})),
                task=dict(payload.get("task", {})),
                track=str(dataset.get("track", "")),
                task_modes=_task_modes_for_loader(str(dataset.get("name", ""))),
                published=_is_published(root, name, manifest),
            )
        )
    return tuple(records)


def dataset_card_path(root: Path, name: str) -> Path:
    """Return the canonical dataset-card path."""
    return root / "catalog" / "datasets" / Path(name) / "README.md"


def family_names(records: tuple[DatasetRecord, ...]) -> tuple[str, ...]:
    """Loaders whose presets select a series by ``id`` own one family card."""
    return tuple(sorted({record.loader for record in records if record.dataset_id}))


# ---------------------------------------------------------------------------
# Front matter
# ---------------------------------------------------------------------------


def split_card(text: str) -> tuple[dict[str, object], str]:
    """Split a card into ordered front-matter values and its body.

    Values are JSON, so lists and numbers stay typed without a YAML dependency.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("card has no front matter")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError("card front matter is unterminated") from exc
    front: dict[str, object] = {}
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, raw = line.partition(":")
        if not separator or key != key.strip() or not key:
            raise ValueError(f"invalid front-matter line: {line!r}")
        try:
            front[key] = json.loads(raw.strip())
        except json.JSONDecodeError as exc:
            raise ValueError(f"front-matter value of {key!r} is not JSON") from exc
    return front, "\n".join(lines[end + 1 :]).strip("\n")


def read_dataset_facts(path: Path) -> dict[str, object]:
    """Return the level-0/1 front-matter facts of one card."""
    return split_card(path.read_text(encoding="utf-8"))[0]


def _placeholder(key: str) -> object:
    return [] if key in {"tags", "seq_lens", "pred_lens", "related"} else TODO


# ---------------------------------------------------------------------------
# Generated blocks
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Rendering and writing
# ---------------------------------------------------------------------------


def _read(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.is_file() else None


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def dataset_facts(root: Path, names: list[str] | None = None) -> dict[str, dict[str, object]]:
    """Read level-0/1 facts for presets (and families) that already have a card."""
    wanted = names or [record.name for record in dataset_records(root)]
    facts: dict[str, dict[str, object]] = {}
    for name in wanted:
        path = dataset_card_path(root, name)
        if path.is_file():
            try:
                facts[name] = read_dataset_facts(path)
            except ValueError:
                continue
    return facts


def write_characteristics(root: Path, config: str, fired: list[str], basis: str) -> Path:
    """Record measured data characteristics in the preset's card.

    Measured (data) terms are replaced by ``fired``; curated task terms already in
    the card are kept, and ``spatial-graph`` is added for spatiotemporal presets.
    """
    import json

    from tsflab.catalog.characteristics import TASK_TERMS, data_terms

    config = Path(config).as_posix()
    record = next((r for r in dataset_records(root) if Path(r.config).as_posix() == config), None)
    if record is None:
        raise ValueError(f"no dataset card resolves to preset {config!r}")
    path = dataset_card_path(root, record.name)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    end = lines.index("---", 1)
    front = lines[1:end]
    current: list[str] = []
    for line in front:
        if line.startswith("characteristics: "):
            current = list(json.loads(line.split(": ", 1)[1]))
    task = [term for term in current if term in TASK_TERMS]
    if "spatiotemporal" in record.task_modes and "spatial-graph" not in task:
        task.append("spatial-graph")
    measured = [term for term in fired if term in data_terms()]
    terms = measured + [term for term in task if term not in measured]
    kept = [line for line in front if not line.startswith(("characteristics: ", "characteristics_basis: "))]
    kept += [f"characteristics: {json.dumps(terms)}", f"characteristics_basis: {json.dumps(basis)}"]
    path.write_text("\n".join(["---", *kept, *lines[end:]]) + ("\n" if text.endswith("\n") else ""), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# tsflab.card/1 readers and writers (card.toml + README description)
# ---------------------------------------------------------------------------


def read_dataset_facts(path: Path) -> dict[str, object]:  # noqa: F811 - replaces the README parser
    """Flat retrieval facts of one dataset card (``path``: its README or directory)."""
    from tsflab.catalog.cards.store import load

    card = load(path if path.is_dir() else path.parent)
    f = card.facts
    source, shape, protocol = f.get("source", {}), f.get("shape", {}), f.get("protocol", {})
    return {
        "name": f["name"],
        "kind": f["kind"],
        "summary": card.description,
        "domain": f.get("domain", ""),
        "topic": f.get("topic", ""),
        "benchmarks": list(f.get("benchmarks", [])),
        "tags": list(f.get("tags", [])),
        "characteristics": list(f.get("characteristics", [])),
        "source": source.get("name", ""),
        "source_url": source.get("url", ""),
        "license": source.get("license", ""),
        "license_url": source.get("license_url", ""),
        "redistribution": source.get("redistribution", ""),
        "conditions": source.get("conditions", ""),
        "frequency": shape.get("frequency", ""),
        "time_span": shape.get("time_span", ""),
        "length": shape.get("length", ""),
        "channels": shape.get("channels", ""),
        "channel_kind": shape.get("channel_kind", ""),
        "target": shape.get("target", ""),
        "missing_values": shape.get("missing_values", ""),
        "stats_basis": shape.get("stats_basis", ""),
        "protocol": protocol.get("protocol", ""),
        "split": protocol.get("split", ""),
        "seq_lens": list(protocol.get("seq_lens", [])),
        "pred_lens": list(protocol.get("pred_lens", [])),
        "related": list(f.get("related", [])),
        "realtime_track": f.get("realtime_track", ""),
    }


def write_characteristics(root: Path, config: str, fired: list[str], basis: str) -> Path:  # noqa: F811
    """Record measured data characteristics in the preset's ``card.toml``.

    Measured (data) terms are replaced by ``fired``; curated task terms already in
    the card are kept, and ``spatial-graph`` is added for spatiotemporal presets.
    """
    import tomllib

    from tsflab.catalog.cards.toml_io import dumps
    from tsflab.catalog.characteristics import TASK_TERMS, data_terms

    config = Path(config).as_posix()
    record = next((r for r in dataset_records(root) if Path(r.config).as_posix() == config), None)
    if record is None:
        raise ValueError(f"no dataset card resolves to preset {config!r}")
    path = dataset_card_path(root, record.name).with_name("card.toml")
    facts = tomllib.loads(path.read_text(encoding="utf-8"))
    task = [term for term in facts.get("characteristics", []) if term in TASK_TERMS]
    if "spatiotemporal" in record.task_modes and "spatial-graph" not in task:
        task.append("spatial-graph")
    measured = [term for term in fired if term in data_terms()]
    facts["characteristics"] = measured + [term for term in task if term not in measured]
    facts["characteristics_basis"] = basis
    ordered = {
        k: facts[k]
        for k in (
            "schema",
            "kind",
            "name",
            "domain",
            "topic",
            "benchmarks",
            "tags",
            "characteristics",
            "characteristics_basis",
        )
        if k in facts
    }
    ordered.update({k: v for k, v in facts.items() if k not in ordered})
    path.write_text(dumps(ordered), encoding="utf-8")
    return path
