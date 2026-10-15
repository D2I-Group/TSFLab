#!/usr/bin/env python3
"""Build data/model-meta.json from a TSFLab checkout.

Each model lives at ``<tsflab>/src/tsflab/models/<name>/README.md`` with a YAML
front matter block carrying ``model`` / ``year`` / ``venue`` / ``arxiv`` /
``paper_title``. We harvest those into a flat ``{model: {...}}`` map keyed by
the canonical model name (the ``model:`` field), which matches the model names
used in ``leaderboard.json``.

This metadata powers the "method evolution" chart (publication year vs MSE).
It is kept as a separate static file so the HF-regenerated ``leaderboard.json``
stays untouched.

Usage:
    python pipeline/build_model_meta.py            # defaults to this monorepo's root
    python pipeline/build_model_meta.py /path/to/TSFLab --out data/model-meta.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys


def _field(front_matter: str, key: str) -> str | None:
    m = re.search(rf'^{key}:\s*"?([^"\n]*)"?\s*$', front_matter, re.M)
    if not m:
        return None
    val = m.group(1).strip().strip('"')
    return val or None


def parse_models(tsflab_root: str) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    pattern = os.path.join(tsflab_root, "src", "tsflab", "models", "*", "README.md")
    for readme in sorted(glob.glob(pattern)):
        text = open(readme, encoding="utf-8", errors="ignore").read()
        fm = re.match(r"^---\n(.*?)\n---", text, re.S)
        if not fm:
            continue
        block = fm.group(1)
        name = _field(block, "name")
        if not name:
            continue
        year = _field(block, "year")
        entry = {
            "year": int(year) if year and year.isdigit() else None,
            "venue": _field(block, "venue"),
            "arxiv": _field(block, "paper"),
            "paper": _field(block, "paper_title"),
        }
        meta[name] = entry
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "tsflab",
        nargs="?",
        default=os.path.join(os.path.dirname(__file__), "..", "..", ".."),
        help="Path to a TSFLab checkout (default: the enclosing monorepo)",
    )
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "model-meta.json"),
        help="Output JSON path",
    )
    args = ap.parse_args()

    meta = parse_models(args.tsflab)
    if not meta:
        print(f"No models found under {args.tsflab}/src/tsflab/models/*/README.md", file=sys.stderr)
        return 1

    out = os.path.abspath(args.out)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    with_year = sum(1 for v in meta.values() if v["year"])
    print(f"Wrote {len(meta)} models ({with_year} with year) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
