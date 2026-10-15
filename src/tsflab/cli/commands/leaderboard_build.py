#!/usr/bin/env python3
"""tsf result leaderboard — aggregate submissions into a leaderboard.json.

Validation and aggregation are the shared TSF-Core implementation used by the
TSFLab Leaderboard site (``tsflab.core.leaderboard``), so a local
build ranks exactly as the published board does. No torch, no LLM.

    uv run tsf result leaderboard --records apps/web/submissions
    uv run tsf result leaderboard --records work_dirs/_submissions --out leaderboard.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from tsflab.cli.args import add_option
from tsflab.core.leaderboard import PRIMARY_METRIC, aggregate, load_submissions

# Tracks that always appear, in this order, so empty tracks read as "open for
# submissions" rather than silently disappearing.
CANONICAL_TRACKS = ("time_series", "spatiotemporal", "covariate", "realtime")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tsf result leaderboard", description=__doc__)
    add_option(
        ap,
        "--records",
        "source",
        "--source",
        default="work_dirs/_submissions",
        help="submission-records root searched for submission.json files (default work_dirs/_submissions)",
    )
    ap.add_argument("--out", default="leaderboard.json", help="Output path")
    ap.add_argument(
        "--primary-metric", default=PRIMARY_METRIC, help="Metric used to rank (lower is better; default: mse)"
    )
    args = ap.parse_args(argv)

    source = Path(args.source)
    if not source.is_dir():
        sys.exit(f"error: source dir not found: {source}")
    valid, rejected = load_submissions(source)
    tracks = aggregate((doc for _, doc in valid), args.primary_metric)
    ordered = {t: tracks.get(t, {"datasets": {}}) for t in CANONICAL_TRACKS}
    ordered.update({t: b for t, b in tracks.items() if t not in ordered})
    rejections = [f"{p}: {'; '.join(errs[:3])}" for p, errs in rejected.items()]
    leaderboard = {
        "schema_version": "1.1",
        "generated_at": datetime.now(UTC).isoformat(),
        "primary_metric": args.primary_metric,
        "n_submissions": len(valid),
        "n_rejected": len(rejected),
        "rejections": rejections,
        "tracks": ordered,
    }
    Path(args.out).write_text(json.dumps(leaderboard, indent=2, ensure_ascii=False), encoding="utf-8")
    cells = sum(len(rows) for b in ordered.values() for d in b["datasets"].values() for rows in d["horizons"].values())
    print(f"Built {args.out}")
    print(f"  submissions: {len(valid)} accepted, {len(rejected)} rejected")
    print(f"  tracks: {', '.join(ordered)} · ranked entries: {cells}")
    for msg in rejections:
        print(f"  reject: {msg}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
