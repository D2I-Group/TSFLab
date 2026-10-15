"""Plot sample windows of a dataset split (``tsf data analyze --plot``)."""

from __future__ import annotations

import os
from collections.abc import Iterable

import matplotlib
import numpy as np


def _select_indices(dataset_len: int, num_samples: int, index: int | None, seed: int | None) -> list[int]:
    if dataset_len <= 0:
        return []
    if index is not None:
        if index < 0 or index >= dataset_len:
            raise ValueError(f"index {index} is out of range (0-{dataset_len - 1})")
        return [index]
    count = min(num_samples, dataset_len)
    if seed is None:
        return list(range(count))
    rng = np.random.default_rng(seed)
    return rng.choice(dataset_len, size=count, replace=False).tolist()


def _normalize_series(series: np.ndarray) -> np.ndarray:
    arr = np.asarray(series)
    if arr.ndim == 1:
        return arr[:, None]
    return arr


def _parse_channels(channels: str | None, total_channels: int) -> list[int]:
    if channels is None or channels == "all":
        return list(range(total_channels))
    tokens = [token.strip() for token in channels.split(",") if token.strip()]
    indices = [int(token) for token in tokens]
    for idx in indices:
        if idx < 0 or idx >= total_channels:
            raise ValueError(f"channel index {idx} is out of range (0-{total_channels - 1})")
    return indices


def _title_parts(config, split: str, sample_index: int, dataset) -> Iterable[str]:
    yield f"{config.dataset.alias or config.dataset.name}"
    yield f"{split}"
    yield f"idx={sample_index}"
    if hasattr(dataset, "period"):
        yield f"period={getattr(dataset, 'period')}"


def plot_samples(
    config,
    split: str,
    save_path,
    *,
    num_samples: int = 3,
    index: int | None = None,
    channels: str = "all",
    seed: int | None = None,
    show: bool = False,
) -> None:
    """Plot input and target windows of one split (``tsf data analyze --plot``) to ``save_path``."""
    from tsflab.cli.commands.dataset_characteristics import _build_dataset_for_inspect

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset, config = _build_dataset_for_inspect(config, split)
    indices = _select_indices(len(dataset), num_samples, index, seed)
    if not indices:
        raise RuntimeError("Dataset is empty; no samples to visualize")
    fig, axes = plt.subplots(len(indices), 1, figsize=(10, max(3, 3 * len(indices))), sharex=False)
    axes = [axes] if len(indices) == 1 else axes
    for ax, sample_index in zip(axes, indices):
        _plot_window(ax, dataset, sample_index, channels)
        ax.set_title(" | ".join(_title_parts(config, split, sample_index, dataset)))
    fig.tight_layout()
    save_dir = os.path.dirname(str(save_path))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _plot_window(ax, dataset, sample_index: int, channels: str) -> None:
    input_series, output_series, _, _ = dataset[sample_index]
    input_arr = _normalize_series(input_series)
    output_arr = _normalize_series(output_series)
    full_arr = np.concatenate([input_arr, output_arr[dataset.label_len :]], axis=0)
    t = np.arange(full_arr.shape[0])
    for ch in _parse_channels(channels, full_arr.shape[1]):
        ax.plot(t, full_arr[:, ch], label=f"ch{ch}")
    ax.axvline(dataset.seq_len - 1, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax.legend(ncol=4, fontsize=8)
