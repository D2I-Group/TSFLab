"""Generate a tiny synthetic CSV for end-to-end smoke runs.

Writes ``dataset/smoke/smoke.csv`` with a ``date`` column (hourly) plus
``N`` numeric channels, the last named ``OT`` (the default target). Just
enough rows to form a handful of train/val/test windows.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def main() -> None:
    """Create the synthetic smoke dataset."""
    rows = 400
    n_channels = 6
    rng = np.random.default_rng(0)

    dates = pd.date_range("2020-01-01", periods=rows, freq="h")
    t = np.arange(rows)
    data = {}
    for c in range(n_channels - 1):
        # Daily + weekly seasonality plus noise.
        series = (
            np.sin(2 * np.pi * t / 24 + c) + 0.5 * np.sin(2 * np.pi * t / (24 * 7)) + 0.1 * rng.standard_normal(rows)
        )
        data[f"ch{c}"] = series
    data["OT"] = np.cos(2 * np.pi * t / 24) + 0.1 * rng.standard_normal(rows)

    df = pd.DataFrame({"date": dates, **data})
    out_dir = os.path.join("dataset", "smoke")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "smoke.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  shape={df.shape}")


if __name__ == "__main__":
    main()
