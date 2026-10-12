"""Profile a dataset for model selection: ``tsf data analyze <preset|--config TOML|--path FILE>``.

Loads the chronological train/val/test split of a dataset preset (same loaders and
split rules as training, but unscaled) or of a raw file/store, then writes a
structured JSON profile plus a short markdown summary, the TFB-style
characteristics table of one split (formerly ``tsf data inspect``), and with
``--plot`` an image of sample windows (formerly ``tsf data plot``). See
``tsflab.data.profile`` for the metrics and ``profile_rules.toml`` for the
profile-to-catalog mapping. Analysis only; it never trains or modifies data.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tsflab.data import profile as prof

DEFAULT_RATIO = (0.7, 0.1, 0.2)


def _full_array(ds) -> np.ndarray:
    raw = getattr(ds, "data", None)
    if raw is None:
        raw = getattr(ds, "values", None)
    if raw is None:
        raise RuntimeError("dataset exposes neither `.data` nor `.values`; use --path FILE")
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 3:  # (T, N, features): channel 0 is the target value
        arr = arr[..., 0]
    return arr[:, None] if arr.ndim == 1 else arr


def _extract_splits(datasets: dict[str, object]) -> dict[str, np.ndarray]:
    """Chronological train/val/test arrays from three split-flagged datasets.

    Window-indexed loaders (CauAir, real-time panels) share one full array; their
    split boundaries are the first window centre of val and test.
    """
    if hasattr(datasets["val"], "idx"):
        full = _full_array(datasets["val"])
        b1 = int(np.asarray(datasets["val"].idx).min())
        b2 = int(np.asarray(datasets["test"].idx).min())
        return {"train": full[:b1], "val": full[b1:b2], "test": full[b2:]}
    return {flag: _full_array(ds) for flag, ds in datasets.items()}


def _frequency_from_file(ds_path: Path) -> Any:
    import pandas as pd

    try:
        if ds_path.suffix == ".csv":
            head = pd.read_csv(ds_path, usecols=[0], nrows=500)
            return pd.to_datetime(head.iloc[:, 0], errors="coerce")
        if ds_path.suffix == ".parquet":
            return pd.read_parquet(ds_path).index[:500]
    except Exception:
        return None
    return None


def _freq_override(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    import pandas as pd

    step = pd.to_timedelta(pd.tseries.frequencies.to_offset(text)).total_seconds()
    return prof.infer_frequency(pd.date_range("2000-01-01", periods=8, freq=f"{int(step)}s"))


def preset_config_path(name_or_path: str) -> Path:
    """A preset name resolves to ``configs/datasets/<name>.toml``; a TOML path is used as given."""
    path = Path(name_or_path)
    if not path.suffix:
        path = Path("configs/datasets") / f"{name_or_path}.toml"
    if not path.is_file():
        raise SystemExit(f"unknown dataset preset {name_or_path!r} (looked for {path})")
    return path


def load_dataset_config(path: Path):
    """The dataset and task of a dataset preset or of a full run config (``extends`` resolved)."""
    from tsflab.cli.commands import dataset_characteristics as dc

    try:
        return dc._load_partial_config(str(path))
    except (RuntimeError, KeyError):  # a run config: [dataset] comes from its extends chain
        from tsflab.experiments.config import load_config

        resolved = load_config(str(path))[0].config
        dataset, task = resolved.dataset, resolved.task
        return dc._PartialConfig(
            dataset=dc._DatasetConfig(
                name=dataset.name, alias=dataset.alias, path=dataset.path or "",
                id=getattr(dataset, "id", None), params=dc._params_to_dict(dataset.params)),
            task=dc._TaskConfig(seq_len=task.seq_len, label_len=task.label_len,
                                pred_len=task.pred_len, features=task.features),
        )


def load_preset(name_or_path: str, freq: str | None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load train/val/test (unscaled) for a dataset preset or run config TOML."""
    from tsflab.cli.commands import dataset_characteristics as dc

    path = preset_config_path(name_or_path)
    partial = load_dataset_config(path)
    params = dict(partial.dataset.params)
    if "scale" in params:
        params["scale"] = False
    config = dc._PartialConfig(
        dataset=dataclasses.replace(partial.dataset, params=params), task=partial.task
    )
    datasets = {flag: dc._build_dataset(config, flag) for flag in ("train", "val", "test")}
    splits = _extract_splits(datasets)
    file_path = getattr(datasets["train"], "file_path", "")
    meta: dict[str, Any] = {
        "name": config.dataset.alias or config.dataset.name,
        "source": f"{path} -> {config.dataset.path}",
        "split_ratio": params.get("split_ratio"),
        "config": config,
    }
    freq_info = _freq_override(freq)
    if freq_info is None:
        stamps = _frequency_from_file(Path(file_path)) if file_path else None
        freq_info = prof.infer_frequency(stamps)
    meta["frequency"] = freq_info
    return splits, meta


def load_path(
    file: str, ratio: tuple[float, float, float], freq: str | None
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load a csv/txt/parquet/npy/npz file or a realtime store directory and split it."""
    import pandas as pd

    p = Path(file)
    stamps = None
    if p.is_dir():
        panels = sorted((p / "panel").glob("*.parquet")) or sorted(p.glob("*.parquet"))
        if not panels:
            raise SystemExit(f"{p} has no panel/*.parquet files")
        frame = pd.concat([pd.read_parquet(f) for f in panels]).sort_index()
        stamps, values = frame.index, frame.to_numpy(np.float64)
    elif p.suffix in {".csv", ".txt"}:
        frame = pd.read_csv(p, header=None if p.suffix == ".txt" else "infer")
        first = frame.iloc[:, 0]
        if not pd.api.types.is_numeric_dtype(first):
            stamps = pd.to_datetime(first, errors="coerce")
            frame = frame.iloc[:, 1:]
        values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    elif p.suffix == ".parquet":
        frame = pd.read_parquet(p)
        stamps, values = frame.index, frame.select_dtypes("number").to_numpy(np.float64)
    elif p.suffix in {".npy", ".npz"}:
        loaded = np.load(p, allow_pickle=False)
        values = np.asarray(loaded if p.suffix == ".npy" else loaded[loaded.files[0]], np.float64)
        values = values[..., 0] if values.ndim == 3 else values
    else:
        raise SystemExit(f"unsupported file type {p.suffix!r}; use csv, txt, parquet, npy, npz, or a store directory")
    if values.ndim == 1:
        values = values[:, None]
    from tsflab.data.protocol import split_borders

    splits = {flag: values[start:end] for flag, (start, end) in split_borders(len(values), ratio).items()}
    freq_info = _freq_override(freq) or prof.infer_frequency(stamps)
    return splits, {"name": p.stem or p.name, "source": str(p), "split_ratio": list(ratio), "frequency": freq_info}


def analyze(splits: dict[str, np.ndarray], meta: dict[str, Any], max_channels: int) -> dict[str, Any]:
    """Build the full report: profile plus catalog recommendations."""
    profile = prof.build_profile(
        splits,
        name=meta["name"],
        source=meta["source"],
        frequency=meta["frequency"],
        max_channels=max_channels,
        split_ratio=tuple(meta["split_ratio"]) if meta.get("split_ratio") else None,
    )
    profile["recommendations"] = prof.evaluate_rules(profile)
    return profile


def tfb_characteristics(splits: dict[str, np.ndarray], split: str, period: int | None,
                        per_channel: bool) -> list[dict[str, Any]]:
    """TFB-style characteristics (trend, seasonality, stationarity, shifting, transition,
    correlation) of one unscaled split: a dataset row, plus one row per channel on request."""
    from tsflab.cli.commands import dataset_characteristics as dc

    series = splits[split]
    if series.shape[0] < 4:
        raise SystemExit(f"split {split!r} has only {series.shape[0]} time steps; too short to analyse")
    if series.shape[1] > dc._MAX_CHANNELS:
        series = series[:, np.linspace(0, series.shape[1] - 1, dc._MAX_CHANNELS).astype(np.int64)]
    rows = dc.compute_characteristics(series, period, per_channel)
    return [{column: getattr(row, column) for column in dc._COLUMNS} for row in rows]


def _characteristics_markdown(rows: list[dict[str, Any]], split: str) -> str:
    from tsflab.cli.commands import dataset_characteristics as dc

    table = [[dc._fmt(row[column]) for column in dc._COLUMNS] for row in rows]
    lines = [f"## TFB characteristics ({split} split)", "",
             "| " + " | ".join(dc._COLUMNS) + " |", "|" + "---|" * len(dc._COLUMNS)]
    lines += ["| " + " | ".join(cells) + " |" for cells in table]
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    hidden = argparse.SUPPRESS
    parser = argparse.ArgumentParser(
        prog="tsf data analyze",
        description="Profile a dataset (train-only statistics, train/val/test shift, TFB "
                    "characteristics) and map it to catalog facts; --plot also draws sample windows.",
    )
    parser.add_argument("preset", nargs="?", metavar="dataset", help="dataset preset name (configs/datasets/<name>.toml)")
    parser.add_argument("--config", help="dataset preset or run config TOML (overrides <preset>)")
    parser.add_argument("--dataset", dest="dataset_alias", help=hidden)
    parser.add_argument("--path", help="raw csv/txt/parquet/npy/npz file or realtime store directory")
    parser.add_argument("--split-ratio", type=float, nargs=3, default=DEFAULT_RATIO,
                        metavar=("TRAIN", "VAL", "TEST"), help="chronological split for --path")
    parser.add_argument("--freq", help="sampling interval override such as 10min or 1h")
    parser.add_argument("--max-channels", type=int, default=128,
                        help="channels sampled for spectral statistics (default 128)")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train",
                        help="split for the TFB characteristics and the sample plot (default train)")
    parser.add_argument("--period", type=int, default=None,
                        help="seasonal period for the TFB characteristics (default: dominant FFT period)")
    parser.add_argument("--per-channel", action="store_true", help="one TFB characteristics row per channel")
    parser.add_argument("--plot", action="store_true", help="also plot sample windows (preset or --config)")
    parser.add_argument("--num-samples", type=int, default=3, help="windows to plot (default 3)")
    parser.add_argument("--index", type=int, default=None, help="plot this window index only")
    parser.add_argument("--channels", default="all", help="channels to plot: comma-separated indices or all")
    parser.add_argument("--seed", type=int, default=None, help="random seed for the plotted windows")
    parser.add_argument("--out", help="output directory (default work_dirs/profiles/<name>)")
    parser.add_argument("--json", action="store_true", help="print the JSON profile instead of markdown")
    parser.add_argument("--write-card", action="store_true",
                        help="record the fired data characteristics in the preset's dataset card")
    # Spellings of the retired `data inspect` and `data plot`: CSV and image paths, a plot window.
    parser.add_argument("--characteristics-csv", help=hidden)
    parser.add_argument("--plot-file", "--save", dest="plot_file", help=hidden)
    parser.add_argument("--show", action="store_true", help=hidden)
    return parser


def _validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    args.preset = args.config or args.preset or args.dataset_alias
    if bool(args.preset) == bool(args.path):
        parser.error("give exactly one of <preset> (or --config TOML) or --path FILE")
    if args.max_channels < 2:
        parser.error("--max-channels must be at least 2")
    if args.write_card and not args.preset:
        parser.error("--write-card needs a preset")
    if (args.plot or args.plot_file or args.show) and not args.preset:
        parser.error("--plot needs a preset or --config (sample windows come from its loader)")


def _write_card(preset: str, report: dict[str, Any]) -> tuple[Path, list[str]]:
    from datetime import date

    from tsflab.catalog.cards.datasets import write_characteristics
    from tsflab.core.paths import repository_root

    config = Path(preset) if Path(preset).suffix else Path("configs/datasets") / f"{preset}.toml"
    fired = [str(rule["id"]) for rule in report["recommendations"]["fired"]]
    basis = (f"measured by tsf data analyze ({prof.SCHEMA}, decision-safe train/val metrics, "
             f"thresholds in profile_rules.toml) on {date.today().isoformat()}")
    return write_characteristics(repository_root(), config.as_posix(), fired, basis), fired


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate(parser, args)
    splits, meta = (
        load_preset(args.preset, args.freq)
        if args.preset
        else load_path(args.path, tuple(args.split_ratio), args.freq)
    )
    config = meta.pop("config", None)
    report = analyze(splits, meta, args.max_channels)
    rows = tfb_characteristics(splits, args.split, args.period, args.per_channel)
    report["tfb_characteristics"] = {"split": args.split, "rows": rows}
    markdown = prof.render_markdown(report, report["recommendations"]) + "\n" + \
        _characteristics_markdown(rows, args.split)
    out = Path(args.out or Path("work_dirs/profiles") / meta["name"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "profile.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "profile.md").write_text(markdown, encoding="utf-8")
    from tsflab.cli.commands import dataset_characteristics as dc

    csv_path = Path(args.characteristics_csv or out / f"characteristics_{args.split}.csv")
    dc.write_rows_csv(str(csv_path), rows)
    note = sys.stderr if args.json else sys.stdout
    print(json.dumps(report, indent=2) if args.json else markdown)
    outputs = [out / "profile.json", out / "profile.md", csv_path]
    if args.plot or args.plot_file or args.show:
        from tsflab.cli.commands.visual_data import plot_samples

        image = Path(args.plot_file or out / f"samples_{args.split}.png")
        plot_samples(config, args.split, image, num_samples=args.num_samples, index=args.index,
                     channels=args.channels, seed=args.seed, show=args.show)
        outputs.append(image)
    print("Output: " + ", ".join(str(path) for path in outputs), file=note)
    if args.write_card:
        card, fired = _write_card(args.preset, report)
        print(f"Characteristics written to {card}: {', '.join(fired) or 'none fired'}", file=note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
