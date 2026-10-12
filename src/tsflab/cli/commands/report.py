#!/usr/bin/env python3
"""Generate a Markdown benchmark report for a dataset.

Pulls together the aggregated results, a Top-N leaderboard, and a performance
bubble chart into one shareable Markdown file. Pure standard library; reuses the
internal aggregate and plot commands under the hood.

Example
-------
    uv run tsf result report ETTh1 --horizon 96
"""
from __future__ import annotations

import argparse

from tsflab.cli.args import add_name, add_option, resolve_name
import csv
import subprocess
import sys
from pathlib import Path

from tsflab.core.paths import working_root

ROOT = working_root()
def _run_tool(script: str, args: list[str]) -> subprocess.CompletedProcess:
    module = script.removesuffix(".py")
    return subprocess.run(
        [sys.executable, "-m", f"tsflab.cli.commands.{module}", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    if not path.exists():
        return [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def _fmt(x, n: int = 4) -> str:
    try:
        return f"{float(x):.{n}f}"
    except (TypeError, ValueError):
        return str(x)


def _md_table(headers: list[str], rows: list[dict]) -> str:
    if not rows:
        return "_(no data)_"
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(prog="tsf result report", description="Generate a Markdown benchmark report for a dataset.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_name(ap, "dataset", flag="--dataset", help="dataset name (<records>/<dataset>/)")
    add_option(ap, "--records", "work_dir", "--work-dir", default="work_dirs",
               help="run-records root (default work_dirs)")
    add_option(ap, "--horizon", "pred_len", "--pred-len", default=None, help="only this prediction length")
    ap.add_argument("--top", type=int, default=10, help="Top-N models in the leaderboard")
    ap.add_argument("--out", default=None, help="Output .md path (default: work_dirs/<dataset>/report.md)")
    ap.add_argument("--no-plot", action="store_true", help="Skip the bubble chart")
    args = ap.parse_args()
    resolve_name(ap, args, "dataset")

    ds = args.dataset
    out_dir = Path(args.work_dir) / ds
    out_dir.mkdir(parents=True, exist_ok=True)
    agg_csv = out_dir / "results_all.csv"
    bubble_svg = out_dir / "bubble.svg"
    out_md = Path(args.out) if args.out else out_dir / "report.md"

    # 1. Aggregate per-run CSVs.
    agg_args = [ds, "--records", args.work_dir, "--out", str(agg_csv),
                "--prof-fields", "latency_avg_ms,throughput_samples_sec,total_params,peak_vram_mb"]
    if args.pred_len:
        agg_args += ["--filter", f"pred_len={args.pred_len}"]
    aggregated = _run_tool("aggregate_results.py", agg_args)
    if aggregated.returncode:
        print(aggregated.stderr or aggregated.stdout or "Result aggregation failed", file=sys.stderr)
        return aggregated.returncode
    headers, rows = _read_csv(agg_csv)

    sections: list[str] = []
    sub = f"_{len(rows)} runs_" + (f", pred_len={args.pred_len}" if args.pred_len else "")
    sections.append(f"# Benchmark report — `{ds}`\n\n{sub}")

    if not rows:
        sections.append("_No results found. Run experiments first "
                        "(`uv run tsf run <config>`), then re-generate this report._")
        out_md.write_text("\n\n".join(sections) + "\n")
        print(f"✓ Wrote {out_md} (empty — no results for '{ds}')")
        return 0

    # Compare only fingerprinted protocols and matching seed coverage.
    from tsflab.experiments.infra.comparison import compare_rows
    import json
    from tsflab.experiments.infra.comparison import protocol_fingerprint
    from tsflab.experiments.infra.storage import canonical_hash
    states = {}
    planned = []
    for path in (Path(args.work_dir) / "_runs").glob("*/manifest.json"):
        state = json.loads(path.read_text())
        states[state["run_id"]] = state["status"]
        snapshot = state["config"]
        name = snapshot["dataset"].get("alias") or snapshot["dataset"]["name"]
        if name == ds and (not args.pred_len or str(snapshot["task"]["pred_len"]) == args.pred_len):
            planned.append({"model": snapshot["model"]["name"], "model_variant": canonical_hash(snapshot["model"]),
                            "seed": snapshot["experiment"]["random_seed"],
                            "protocol_sha256": protocol_fingerprint(snapshot, state["data"])})
    completed = [row for row in rows if states.get(row.get("run_id")) == "succeeded"]
    comparison = compare_rows(completed, planned)
    if len(completed) != len(rows):
        sections.append(f"{len(rows) - len(completed)} rows have no confirmed successful run manifest and are excluded from ranking.")
    lb = []
    if comparison["unverified_runs"]:
        sections.append(f"{comparison['unverified_runs']} legacy runs lack protocol/variant fingerprints. They remain in the results table but are not ranked.")
    for cohort in comparison["cohorts"]:
        lb_rows = []
        rank = 0
        for entry in cohort["leaderboard"][:args.top]:
            if entry["rankable"]:
                rank += 1
            lb_rows.append({"rank": rank if entry["rankable"] else "incomplete",
                           "model": entry["model"], "variant": entry["variant"],
                           "mse": _fmt(entry["mse"]), "mse_std": _fmt(entry["mse_std"]),
                           "mae": _fmt(entry["mae"]), "seeds": ", ".join(entry["seeds"]),
                           "missing": ", ".join(entry["missing_seeds"]), "duplicates": ", ".join(entry["duplicate_seeds"])})
        lb.extend(lb_rows)
        sections.append(f"## Protocol {cohort['protocol'][:12]}\n\n" + _md_table(
            ["rank", "model", "variant", "mse", "mse_std", "mae", "seeds", "missing", "duplicates"], lb_rows))

    # 3. Bubble chart (skipped gracefully if the size field is absent).
    if not args.no_plot and "total_params" in headers and len(comparison["cohorts"]) == 1 and not comparison["unverified_runs"] and len(completed) == len(rows):
        plotted = _run_tool("plot_bubble.py", ["--csv", str(agg_csv), "--x", "mse", "--y", "mae",
                                     "--size", "total_params", "--size-scale", "log",
                                     "--color-by", "model", "--label-by", "model",
                                     "--out", str(bubble_svg)])
        if plotted.returncode:
            print(plotted.stderr or plotted.stdout or "Plot generation failed", file=sys.stderr)
            return plotted.returncode
        if bubble_svg.exists():
            sections.append(f"## Performance bubble chart\n\n"
                            f"x = MSE, y = MAE, size = total_params.\n\n![bubble]({bubble_svg.name})")
    elif not args.no_plot:
        sections.append("## Performance bubble chart\n\n"
                        "_Skipped: missing profile data or multiple/unverified protocols. "
                        "Run with profiling to enable the chart._")

    # 4. Full results table (key columns, capped).
    key_cols = [c for c in ["model", "seq_len", "pred_len", "seed", "mse", "mae", "rmse", "mape"]
                if c in headers]
    metric_cols = {"mse", "mae", "rmse", "mape"}
    cap = args.top * 2
    tbl_rows = [{c: (_fmt(r[c]) if c in metric_cols else r.get(c, "")) for c in key_cols}
                for r in rows[:cap]]
    sections.append(f"## Results ({len(rows)} runs, showing up to {len(tbl_rows)})\n\n"
                    + _md_table(key_cols, tbl_rows))

    out_md.write_text("\n\n".join(sections) + "\n")
    print(f"✓ Wrote {out_md}  ({len(rows)} runs, {len(lb)} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
