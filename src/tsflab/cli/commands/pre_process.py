"""Convert time-series CSV(s) to pre-windowed numpy split files.

Each output ``.npz`` file contains:

  x        float32  (N, seq_len, n_features)         input series
  y        float32  (N, label_len + pred_len, n_features)  decoder target
  x_mark   float32  (N, seq_len, 6)                  input timestamps
  y_mark   float32  (N, label_len + pred_len, 6)     output timestamps
  scaler_mean   float32  (n_features,)   only when --scale
  scaler_scale  float32  (n_features,)   only when --scale

Usage examples
--------------
# From a single CSV (auto-split 70/10/20):
  uv run tsf data prepare \\
      --input-csv dataset/ETTh1.csv \\
      --out dataset/ETTh1_npy \\
      --seq-len 96 --label-len 48 --horizon 96 --features M

# From a folder that already has train/val/test CSVs:
  uv run tsf data prepare \\
      --input-dir dataset/my_dataset \\
      --out dataset/my_dataset_npy \\
      --seq-len 96 --label-len 48 --horizon 96 --features S --target OT
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from tsflab.cli.args import add_option

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_features(df: pd.DataFrame, features: str, target: str) -> pd.DataFrame:
    if features in {"M", "MS"}:
        cols = [c for c in df.columns if c != "date"]
        return df[cols].copy()
    return df[[target]].copy()


def _build_stamp(df: pd.DataFrame) -> np.ndarray:
    """Return (N, 6) float32 timestamp array, or zeros if no 'date' column."""
    if "date" not in df.columns:
        return np.zeros((len(df), 6), dtype=np.float32)
    ts = pd.to_datetime(df["date"])
    return np.column_stack(
        [
            ts.dt.year,
            ts.dt.month,
            ts.dt.day,
            ts.dt.weekday,
            ts.dt.hour,
            ts.dt.minute,
        ]
    ).astype(np.float32)


def _make_windows(
    data: np.ndarray,
    stamp: np.ndarray,
    seq_len: int,
    label_len: int,
    pred_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Slide a window over *data* and *stamp*, returning stacked arrays."""
    n = len(data)
    n_windows = n - seq_len - pred_len + 1
    if n_windows <= 0:
        raise ValueError(f"Not enough rows ({n}) for seq_len={seq_len} + pred_len={pred_len}.")

    x_list, y_list, xm_list, ym_list = [], [], [], []
    for i in range(n_windows):
        x_list.append(data[i : i + seq_len])
        y_list.append(data[i + seq_len - label_len : i + seq_len + pred_len])
        xm_list.append(stamp[i : i + seq_len])
        ym_list.append(stamp[i + seq_len - label_len : i + seq_len + pred_len])

    return (
        np.stack(x_list, axis=0).astype(np.float32),
        np.stack(y_list, axis=0).astype(np.float32),
        np.stack(xm_list, axis=0).astype(np.float32),
        np.stack(ym_list, axis=0).astype(np.float32),
    )


def _save_splits(
    splits: dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> None:
    """Scale, window, and save each split as an .npz file."""
    # Fit scaler on train only
    scaler: StandardScaler | None = None
    if args.scale and len(splits.get("train", [])) > 0:
        train_data = _select_features(splits["train"], args.features, args.target).to_numpy()
        scaler = StandardScaler()
        scaler.fit(train_data)

    os.makedirs(args.output_dir, exist_ok=True)

    for split_name, df in splits.items():
        if len(df) == 0:
            print(f"  {split_name:5s}  skipped (ratio is 0)")
            continue
        raw = _select_features(df, args.features, args.target).to_numpy().astype(np.float32)
        data = scaler.transform(raw).astype(np.float32) if scaler else raw
        stamp = _build_stamp(df)

        x, y, x_mark, y_mark = _make_windows(data, stamp, args.seq_len, args.label_len, args.pred_len)

        save_kwargs: dict[str, np.ndarray] = dict(x=x, y=y, x_mark=x_mark, y_mark=y_mark)
        if scaler is not None:
            save_kwargs["scaler_mean"] = scaler.mean_.astype(np.float32)
            save_kwargs["scaler_scale"] = scaler.scale_.astype(np.float32)

        out_path = os.path.join(args.output_dir, f"{split_name}.npz")
        np.savez(out_path, **save_kwargs)
        print(f"  {split_name:5s}  {x.shape[0]:>6d} samples  x{list(x.shape[1:])}  y{list(y.shape[1:])}  →  {out_path}")


# ---------------------------------------------------------------------------
# Split strategies
# ---------------------------------------------------------------------------


def _process_single_csv(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input_csv)
    ratios = [float(r) for r in args.split_ratio.split(",")]
    if len(ratios) != 3:
        raise ValueError("--split-ratio must be three comma-separated values, e.g. 0.7,0.1,0.2")
    n = len(df)
    total = sum(ratios)
    n_train = int(n * ratios[0] / total)
    n_val = int(n * ratios[1] / total)
    splits = {
        "train": df.iloc[:n_train].reset_index(drop=True),
        "val": df.iloc[n_train : n_train + n_val].reset_index(drop=True),
        "test": df.iloc[n_train + n_val :].reset_index(drop=True),
    }
    print(f"Split sizes — train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])}")
    _save_splits(splits, args)


def _process_split_dir(args: argparse.Namespace) -> None:
    splits = {}
    for name, fname in [("train", "train.csv"), ("val", "val.csv"), ("test", "test.csv")]:
        path = os.path.join(args.input_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected file not found: {path}")
        splits[name] = pd.read_csv(path)
    _save_splits(splits, args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tsf data prepare",
        description="Convert time-series CSV(s) to pre-windowed .npz splits for use with the 'pre_processed' dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--input-csv",
        metavar="PATH",
        help="Single CSV file. Will be split automatically (see --split-ratio).",
    )
    src.add_argument(
        "--input-dir",
        metavar="DIR",
        help="Folder containing train.csv, val.csv, and test.csv.",
    )

    add_option(
        parser,
        "--out",
        "output_dir",
        "--output-dir",
        metavar="DIR",
        required=True,
        help="Where to write train.npz, val.npz, test.npz.",
    )
    parser.add_argument("--seq-len", type=int, required=True, help="Input sequence length.")
    parser.add_argument("--label-len", type=int, required=True, help="Decoder label length.")
    add_option(parser, "--horizon", "pred_len", "--pred-len", type=int, required=True, help="Prediction horizon.")
    parser.add_argument("--features", choices=["M", "S", "MS"], default="M", help="Feature mode (default: M).")
    parser.add_argument("--target", default="OT", help="Target column for S/MS mode (default: OT).")
    parser.add_argument(
        "--scale", action=argparse.BooleanOptionalAction, default=True, help="Apply StandardScaler (default: --scale)."
    )
    parser.add_argument(
        "--split-ratio",
        default="0.7,0.1,0.2",
        metavar="T,V,TE",
        help="Train/val/test ratio for --input-csv (default: 0.7,0.1,0.2).",
    )

    args = parser.parse_args()

    print(
        f"Settings: seq_len={args.seq_len}, label_len={args.label_len}, "
        f"pred_len={args.pred_len}, features={args.features}, "
        f"target={args.target}, scale={args.scale}"
    )

    if args.input_csv:
        _process_single_csv(args)
    else:
        _process_split_dir(args)

    print("Done.")


if __name__ == "__main__":
    main()
