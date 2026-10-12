"""Export the TSF-Core pydantic models to JSON Schema.

The generated ``schema/*.json`` files are the *only* contract TSFLab Leaderboard consumes —
it never imports this package's Python. Export is deterministic (sorted keys,
stable ``$id`` / version injection) so the committed files diff cleanly and a
``--check`` CI gate can reject "changed the model, forgot to re-export".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .constants import SCHEMA_VERSION
from .dataset_spec import DatasetSpec
from .realtime import ForecastSubmission, RoundScore, RoundSpec
from .run_record import RunRecord
from .submission import SubmissionReport

# Default location: committed alongside this package.
DEFAULT_SCHEMA_DIR = Path(__file__).parent / "schema"

_ID_BASE = "https://diaugeia.ai/schema"


def iter_models() -> list[tuple[str, type]]:
    """Return (name, model) pairs for every exported contract model."""
    return [
        ("dataset_spec", DatasetSpec),
        ("run_record", RunRecord),
        ("submission_report", SubmissionReport),
        ("round_spec", RoundSpec),
        ("forecast_submission", ForecastSubmission),
        ("round_score", RoundScore),
    ]


def _render(name: str, model: type) -> str:
    schema = model.model_json_schema()
    schema["$id"] = f"{_ID_BASE}/{name}.schema.json"
    schema["x-tsf-schema-version"] = SCHEMA_VERSION
    return json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _render_index() -> str:
    index = {
        "schema_version": SCHEMA_VERSION,
        "schemas": {name: f"{name}.schema.json" for name, _ in iter_models()},
    }
    return json.dumps(index, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def export_schemas(out_dir: str | Path = DEFAULT_SCHEMA_DIR) -> list[Path]:
    """Write every model's JSON Schema (+ index.json) into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, model in iter_models():
        path = out / f"{name}.schema.json"
        path.write_text(_render(name, model), encoding="utf-8")
        written.append(path)
    index_path = out / "index.json"
    index_path.write_text(_render_index(), encoding="utf-8")
    written.append(index_path)
    return written


def _check(out_dir: Path) -> int:
    """Compare freshly-rendered schemas against the committed files."""
    rendered: dict[str, str] = {f"{n}.schema.json": _render(n, m) for n, m in iter_models()}
    rendered["index.json"] = _render_index()
    drift: list[str] = []
    for fname, content in rendered.items():
        path = out_dir / fname
        if not path.exists():
            drift.append(f"missing: {fname}")
        elif path.read_text(encoding="utf-8") != content:
            drift.append(f"out of date: {fname}")
    if drift:
        print("schema-export --check FAILED — committed schemas are stale:", file=sys.stderr)
        for line in drift:
            print(f"  - {line}", file=sys.stderr)
        print("Run `tsf repo schema` and commit the result.", file=sys.stderr)
        return 1
    print(f"schema-export --check OK ({len(rendered)} files up to date).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tsf repo schema", description="Export TSF-Core models to JSON Schema.")
    parser.add_argument(
        "--out", dest="out_dir", metavar="DIR", default=str(DEFAULT_SCHEMA_DIR),
        help="Output directory (default: package schema/)."
    )
    parser.add_argument(  # former spelling
        "--out-dir", dest="out_dir", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="CI mode: verify committed schemas match the models; exit 1 on drift.",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    if args.check:
        return _check(out_dir)

    written = export_schemas(out_dir)
    print(f"Wrote {len(written)} schema file(s) to {out_dir}:")
    for path in written:
        print(f"  - {path.name}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
