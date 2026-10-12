#!/usr/bin/env python3
"""Rank completed model runs by horizon, seed, MSE, and MAE."""

import argparse

from tsflab.cli.args import add_name, add_option, resolve_name
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tsf result rank",
        description="Rank models by pred_len/seed for MSE and MAE."
    )
    add_option(parser, "--records", "input_root", "--input-root", type=Path, default=Path("work_dirs"),
               help="run-records root with <dataset>/<model>/ performance.csv files (default work_dirs)")
    add_name(parser, "dataset", flag="--dataset", help="dataset name to rank (default ETTh1)")
    parser.add_argument(
        "--out-mse",
        type=Path,
        default=None,
        help="Output wide MSE rankings CSV (model names)",
    )
    parser.add_argument(
        "--out-mae",
        type=Path,
        default=None,
        help="Output wide MAE rankings CSV (model names)",
    )
    parser.add_argument(
        "--out-long",
        type=Path,
        default=None,
        help="Output long rankings CSV (for plotting)",
    )
    # --- TFB-style fairness policy (opt-in; defaults preserve prior behavior) ---
    parser.add_argument(
        "--null-threshold",
        type=float,
        default=None,
        help=(
            "TFB fairness: exclude any model that is NaN/missing on more than this "
            "fraction of the (pred_len, seed) cells for the dataset. "
            "Unset (default) disables exclusion and preserves prior behavior. "
            "Typical value: 0.3."
        ),
    )
    parser.add_argument(
        "--aggregate",
        choices=["mean", "median", "max"],
        default="mean",
        help=(
            "TFB fairness: how to collapse multiple metric values within the same "
            "(model, pred_len, seed) cell when duplicates exist (default: mean). "
            "With no duplicate rows this is a no-op and matches prior behavior."
        ),
    )
    parser.add_argument(
        "--fill-nan-with-mean",
        action="store_true",
        help=(
            "TFB fairness: after excluding models over --null-threshold, fill any "
            "remaining NaN metric cells with that metric's column mean (computed per "
            "metric over surviving rows) before ranking. Off by default."
        ),
    )
    args = parser.parse_args()
    resolve_name(parser, args, "dataset", default="ETTh1")
    return args


def read_performance_files(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("**/performance.csv"))
    if not files:
        raise FileNotFoundError(f"No performance.csv found under {root}")

    frames = []
    for file in files:
        df = pd.read_csv(file)
        required = {"model", "pred_len", "seed", "mse", "mae"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns {sorted(missing)} in {file}")
        if "dataset" not in df.columns:
            dataset = file.parent.parent.name
            df["dataset"] = dataset
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def _apply_fairness_policy(
    base: pd.DataFrame,
    *,
    aggregate: str,
    null_threshold: float | None,
    fill_nan_with_mean: bool,
) -> pd.DataFrame:
    """Apply the TFB-style fairness policy to a (dataset, model, pred_len, seed,
    mse, mae) frame.

    Steps (each opt-in / no-op under defaults):
      1. Coerce mse/mae to numeric so NaN/missing/non-numeric become NaN.
      2. Collapse duplicate (model, pred_len, seed) rows for each metric using
         --aggregate (mean/median/max). With no duplicates this is identity.
      3. If --null-threshold is set, drop any model whose fraction of NaN/missing
         cells (over the full pred_len x seed grid present in the data) exceeds the
         threshold. A model absent from a cell counts as missing for that cell.
         Dropped models are logged with their null fraction.
      4. If --fill-nan-with-mean is set, fill remaining NaN metric cells with that
         metric's column mean over surviving rows, then re-densify the grid.
    """
    base = base.copy()
    for metric in ("mse", "mae"):
        base[metric] = pd.to_numeric(base[metric], errors="coerce")

    # Step 2: collapse duplicates within a cell per metric.
    grouped = base.groupby(["dataset", "model", "pred_len", "seed"], dropna=False)
    base = grouped.agg({"mse": aggregate, "mae": aggregate}).reset_index()

    # Build the full grid of (pred_len, seed) cells observed for this dataset.
    cells = base[["pred_len", "seed"]].drop_duplicates()
    n_cells = len(cells)
    models = sorted(base["model"].unique())
    dataset_name = base["dataset"].iloc[0]

    # Densify: every model x every cell, so absent rows become explicit NaN.
    full_index = pd.MultiIndex.from_product(
        [[dataset_name], models, list(cells.itertuples(index=False, name=None))],
        names=["dataset", "model", "_cell"],
    )
    base = base.set_index(["dataset", "model"])
    base["_cell"] = list(zip(base["pred_len"], base["seed"]))
    base = base.set_index("_cell", append=True)
    dense = base.reindex(full_index)
    dense = dense.reset_index()
    # _cell carries the (pred_len, seed) tuple for every densified row, so these are
    # always present even where the metrics are NaN; restore them as ints.
    dense["pred_len"] = dense["_cell"].map(lambda c: int(c[0]))
    dense["seed"] = dense["_cell"].map(lambda c: int(c[1]))
    dense = dense.drop(columns="_cell")

    # Step 3: null-threshold exclusion. A cell is "missing" for a model if either
    # metric is NaN (covers absent rows and NaN metric values alike).
    if null_threshold is not None and n_cells > 0:
        missing_mask = dense[["mse", "mae"]].isna().any(axis=1)
        null_frac = (
            missing_mask.groupby(dense["model"]).sum() / float(n_cells)
        ).sort_values(ascending=False)
        dropped = null_frac[null_frac > null_threshold]
        if len(dropped) > 0:
            print(
                f"[fairness] null-threshold={null_threshold}: excluding "
                f"{len(dropped)} model(s) over threshold "
                f"(of {len(models)} total, {n_cells} cells each):"
            )
            for model, frac in dropped.items():
                missing_n = int(round(frac * n_cells))
                print(
                    f"  - {model}: {missing_n}/{n_cells} cells missing "
                    f"(null_frac={frac:.3f} > {null_threshold})"
                )
        else:
            print(
                f"[fairness] null-threshold={null_threshold}: no models exceed "
                f"threshold; none excluded."
            )
        keep_models = null_frac[null_frac <= null_threshold].index
        dense = dense[dense["model"].isin(keep_models)].copy()

    # Step 4: fill remaining NaNs with the per-metric column mean before ranking.
    if fill_nan_with_mean:
        for metric in ("mse", "mae"):
            col_mean = dense[metric].mean()
            n_filled = int(dense[metric].isna().sum())
            if n_filled > 0 and pd.notna(col_mean):
                print(
                    f"[fairness] fill-nan-with-mean: filling {n_filled} NaN "
                    f"{metric} cell(s) with column mean={col_mean:.6f}"
                )
                dense[metric] = dense[metric].fillna(col_mean)
    else:
        # Without filling, rows where a metric is NaN cannot be ranked; drop them
        # per metric downstream via dropna in build_metric tables. Keep them here so
        # each metric is handled independently.
        pass

    return dense.reset_index(drop=True)


def build_rankings(
    df: pd.DataFrame,
    dataset: str,
    *,
    aggregate: str = "mean",
    null_threshold: float | None = None,
    fill_nan_with_mean: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = df[["dataset", "model", "pred_len", "seed", "mse", "mae"]].copy()
    base = base[base["dataset"] == dataset].copy()
    if base.empty:
        raise ValueError(f"No rows found for dataset: {dataset}")

    base = _apply_fairness_policy(
        base,
        aggregate=aggregate,
        null_threshold=null_threshold,
        fill_nan_with_mean=fill_nan_with_mean,
    )

    long_frames = []
    for metric in ["mse", "mae"]:
        metric_df = base[["dataset", "model", "pred_len", "seed", metric]].copy()
        metric_df = metric_df.rename(columns={metric: "value"})
        # NaN values cannot be ranked; drop them so they neither get a rank nor
        # pollute the leaderboard (relevant only when --fill-nan-with-mean is off).
        metric_df = metric_df[metric_df["value"].notna()].copy()
        metric_df["metric"] = metric
        metric_df["rank"] = (
            metric_df.groupby(["dataset", "pred_len", "seed", "metric"])["value"]
            .rank(method="min", ascending=True)
            .astype("Int64")
        )
        long_frames.append(metric_df)

    long_df = pd.concat(long_frames, ignore_index=True)

    def build_metric_table(metric: str) -> pd.DataFrame:
        metric_df = long_df[long_df["metric"] == metric].copy()
        metric_df = metric_df.sort_values(
            ["dataset", "pred_len", "seed", "rank", "model"]
        )
        metric_df["setting"] = (
            "pl"
            + metric_df["pred_len"].astype(str)
            + "_seed"
            + metric_df["seed"].astype(str)
        )

        table = metric_df.pivot_table(
            index="rank",
            columns="setting",
            values="model",
            aggfunc="first",
        ).sort_index()

        def sort_key(col: str) -> tuple[int, int]:
            parts = col.split("_")
            pred = int(parts[0].replace("pl", ""))
            seed = int(parts[1].replace("seed", ""))
            return pred, seed

        sorted_cols = sorted(table.columns, key=sort_key)
        table = table[sorted_cols].reset_index()
        return table

    mse_table = build_metric_table("mse")
    mae_table = build_metric_table("mae")

    return mse_table, mae_table, long_df


def main() -> None:
    args = parse_args()
    df = read_performance_files(args.input_root)
    mse_table, mae_table, long_df = build_rankings(
        df,
        args.dataset,
        aggregate=args.aggregate,
        null_threshold=args.null_threshold,
        fill_nan_with_mean=args.fill_nan_with_mean,
    )

    if args.out_mse is None:
        args.out_mse = Path("work_dirs") / args.dataset / "model_rankings_mse.csv"
    if args.out_mae is None:
        args.out_mae = Path("work_dirs") / args.dataset / "model_rankings_mae.csv"
    if args.out_long is None:
        args.out_long = Path("work_dirs") / args.dataset / "model_rankings_long.csv"

    args.out_mse.parent.mkdir(parents=True, exist_ok=True)
    args.out_mae.parent.mkdir(parents=True, exist_ok=True)
    args.out_long.parent.mkdir(parents=True, exist_ok=True)

    mse_table.to_csv(args.out_mse, index=False)
    mae_table.to_csv(args.out_mae, index=False)
    long_df.to_csv(args.out_long, index=False)


if __name__ == "__main__":
    main()
