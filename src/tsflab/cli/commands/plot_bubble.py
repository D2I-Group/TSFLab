"""Plot a bubble chart from a CSV file."""

from __future__ import annotations

import argparse
import os
from typing import cast

import matplotlib
import numpy as np
import pandas as pd

from tsflab.cli.args import add_option


def _apply_size_scale(values, scale: str):
    if scale == "linear":
        return values
    if scale == "sqrt":
        return values.pow(0.5)
    if scale == "log":
        return values.apply(lambda v: None if v <= 0 else np.log10(v))
    raise ValueError(f"Unsupported size scale: {scale}")


def _to_numeric(series):
    extracted = series.astype(str).str.extract(r"([-+]?[0-9]*\.?[0-9]+)")[0]
    return pd.to_numeric(extracted, errors="coerce")


def _normalize_sizes(values, min_size: float, max_size: float) -> list[float]:
    valid = values.dropna()
    if valid.empty:
        return []
    vmin = valid.min()
    vmax = valid.max()
    if vmax == vmin:
        return [max(min_size, (min_size + max_size) / 2) for _ in values]
    scaled = (values - vmin) / (vmax - vmin)
    sizes = min_size + scaled * (max_size - min_size)
    return sizes.fillna(min_size).tolist()


def _default_output_path(csv_path: str) -> str:
    base = os.path.splitext(os.path.basename(csv_path))[0]
    return os.path.join("work_dirs", "plots", f"bubble_{base}.svg")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="tsf result plot", description="Plot a bubble chart from CSV")
    parser.add_argument("--csv", required=True, type=str, help="Path to CSV file")
    parser.add_argument("--x", required=True, type=str, help="Field for x axis")
    parser.add_argument("--y", required=True, type=str, help="Field for y axis")
    parser.add_argument("--size", required=True, type=str, help="Field for bubble size")
    parser.add_argument(
        "--size-scale",
        type=str,
        default="linear",
        choices=("linear", "sqrt", "log"),
        help="Scale for bubble size values",
    )
    parser.add_argument(
        "--x-scale",
        type=str,
        default="linear",
        choices=("linear", "log"),
        help="Scale for x axis",
    )
    parser.add_argument(
        "--y-scale",
        type=str,
        default="linear",
        choices=("linear", "log"),
        help="Scale for y axis",
    )
    parser.add_argument(
        "--color-by",
        type=str,
        default="model",
        help="Field used to color groups",
    )
    parser.add_argument(
        "--label-by",
        type=str,
        default="model",
        help="Field used to annotate points",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Disable point labels",
    )
    parser.add_argument(
        "--legend",
        action="store_true",
        help="Show legend",
    )
    add_option(
        parser,
        "--out",
        "output",
        "--output",
        type=str,
        default=None,
        help="output image path (default: work_dirs/plots/bubble_<csv>.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title",
    )
    return parser.parse_args()


# (scale attribute, scale value, column attribute, keep-row predicate, message when nothing is left)
_SCALE_FILTERS = (
    ("x_scale", "log", "x", lambda v: v > 0, "No rows with positive x values for log scale"),
    ("y_scale", "log", "y", lambda v: v > 0, "No rows with positive y values for log scale"),
    ("size_scale", "log", "size", lambda v: v > 0, "No rows with positive size values for log scale"),
    ("size_scale", "sqrt", "size", lambda v: v >= 0, "No rows with non-negative size values for sqrt scale"),
)


def _load_data(args: argparse.Namespace) -> pd.DataFrame:
    df = cast(pd.DataFrame, pd.read_csv(args.csv))
    required = [args.x, args.y, args.size, args.color_by]
    if not args.no_labels and args.label_by not in required:
        required.append(args.label_by)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in CSV: {', '.join(missing)}")

    columns = [args.x, args.y, args.size, args.color_by]
    if not args.no_labels and args.label_by not in columns:
        columns.append(args.label_by)
    data: pd.DataFrame = cast(pd.DataFrame, df[columns].copy())
    data[args.x] = _to_numeric(data[args.x])
    data[args.y] = _to_numeric(data[args.y])
    data[args.size] = _to_numeric(data[args.size])

    subset = [args.x, args.y, args.size]
    data = data.dropna(subset=subset)
    if data.empty:
        raise SystemExit("No valid rows after filtering non-numeric values")
    return data


def _filter_scales(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    for scale_attr, scale, column_attr, keep, message in _SCALE_FILTERS:
        if getattr(args, scale_attr) == scale:
            column = getattr(args, column_attr)
            data = cast(pd.DataFrame, data[keep(data[column])])
            if data.empty:
                raise SystemExit(message)
    return data


def _colormap(plt, count: int):
    if count <= 20:
        return plt.get_cmap("tab20", count)
    if count <= 60:
        return plt.get_cmap("turbo", count)
    return plt.get_cmap("hsv", count)


def _style_axes(ax, data: pd.DataFrame, args: argparse.Namespace) -> None:
    ax.set_xlabel(args.x)
    ax.set_ylabel(args.y)
    ax.set_xscale(args.x_scale)
    ax.set_yscale(args.y_scale)
    if args.title:
        ax.set_title(args.title)
    else:
        ax.set_title(f"{args.x} vs {args.y} | size: {args.size} ({args.size_scale})")
    ax.grid(alpha=0.3, linestyle="--")
    if not args.no_labels:
        for _, row in data.iterrows():
            x_val = float(cast(float, row[args.x]))
            y_val = float(cast(float, row[args.y]))
            ax.annotate(
                str(row[args.label_by]),
                (x_val, y_val),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
                alpha=0.85,
            )
    if args.legend:
        ax.legend(title=args.color_by, loc="best", fontsize=8)


def main() -> None:
    args = _parse_args()

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _filter_scales(_load_data(args), args)

    scaled_size = _apply_size_scale(data[args.size], args.size_scale)
    sizes = _normalize_sizes(scaled_size, min_size=30.0, max_size=300.0)
    data = data.assign(_size=sizes)

    categories = sorted(data[args.color_by].dropna().unique())
    if not categories:
        raise SystemExit("No categories found for color grouping")

    cmap = _colormap(plt, len(categories))
    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, category in enumerate(categories):
        subset = data[data[args.color_by] == category]
        color = cmap(idx)
        ax.scatter(
            subset[args.x],
            subset[args.y],
            s=subset["_size"],
            alpha=0.85,
            edgecolors="#1a1a1a",
            linewidths=0.6,
            label=str(category),
            color=color,
        )

    _style_axes(ax, data, args)
    fig.tight_layout()

    output_path = args.output or _default_output_path(args.csv)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved bubble chart to {output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
