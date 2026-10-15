"""Forecast a round with any catalog model through the standard runner.

The round's history (up to the cutoff) is exported in the index-windowed
``cauair`` layout that both ``cauair_ts`` (time-series models, nodes as
channels) and ``cauair_st`` (spatiotemporal models) already consume. Future
rows up to the last target are appended with known calendar covariates and
zero values, and the single test window is centred on the cutoff. The runner
trains and early-stops on the history exactly as for any benchmark run; the
best checkpoint then forecasts the test window, and only the target steps are
submitted. Needs the full TSFLab runtime (``torch``).
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tsflab.core.realtime import ForecastSubmission, RoundSpec
from tsflab.data.calendar import node_calendar
from tsflab.realtime.store import PanelStore
from tsflab.realtime.tracks import TrackSpec

# Constructor parameters that always equal the number of channels (or nodes).
_WIDTH_PARAMS = ("enc_in", "dec_in", "c_out", "num_nodes", "n_vars", "node_num", "num_node")


def _naive(ts: str) -> pd.Timestamp:
    stamp = pd.Timestamp(ts)
    return stamp.tz_localize(None) if stamp.tzinfo is not None else stamp


def export_bundle(
    spec: RoundSpec, store: PanelStore, track: TrackSpec, directory: Path, val_fraction: float = 0.1
) -> dict:
    """Write ``his.npz`` + window indices for the round; return layout facts."""
    offset = pd.tseries.frequencies.to_offset(spec.freq)
    cutoff = _naive(spec.cutoff)
    days = track.history_days or 365
    history = store.read(start=cutoff - pd.Timedelta(days=days), end=cutoff)
    history = history.reindex(columns=spec.channels)
    grid = pd.date_range(history.index.min(), cutoff, freq=offset)
    history = history.reindex(grid).ffill().bfill().fillna(0.0)
    future = pd.date_range(cutoff, _naive(spec.target_timestamps[-1]), freq=offset)[1:]
    lead = len(future)
    index = grid.append(future)
    values = np.concatenate([history.to_numpy(np.float32), np.zeros((lead, len(spec.channels)), np.float32)])
    data = np.concatenate([values[..., None], node_calendar(index, len(spec.channels))], axis=-1)
    hist_len = len(grid)
    mean = np.array([history.to_numpy().mean(), 0.0, 0.0], np.float32)
    std = np.array([history.to_numpy().std() or 1.0, 1.0, 1.0], np.float32)
    centers = np.arange(spec.seq_len - 1, hist_len - lead)
    if len(centers) < 10:
        raise ValueError("not enough history for training windows; increase history_days")
    split = int(len(centers) * (1 - val_fraction))
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "his.npz", data=data, mean=mean, std=std)
    np.save(directory / "idx_train.npy", centers[:split])
    np.save(directory / "idx_val.npy", centers[split:])
    np.save(directory / "idx_test.npy", np.array([hist_len - 1]))
    return {"lead": lead, "input_dim": data.shape[-1]}


def _run_config(spec: RoundSpec, model: str, bundle: Path, layout: dict, work_dir: Path, overrides: dict) -> Path:
    from tsflab.catalog.registry.models import MODEL_CATALOG

    catalog = MODEL_CATALOG.get(model)
    mode = (
        "spatiotemporal"
        if spec.mode == "spatiotemporal" and "spatiotemporal" in catalog.capabilities
        else "time_series"
    )
    params = {k: len(spec.channels) for k in _WIDTH_PARAMS if k in catalog.params_schema.model_fields}
    params.update(overrides.pop("model.params", {}))
    lines = [
        f'extends = ["tsflab://configs/base.toml", "tsflab://configs/models/{model}.toml"]',
        "[experiment]",
        f'work_dir = "{work_dir.as_posix()}"',
        f'description = "realtime {spec.track} {spec.round_id}"',
        "[dataset]",
        f'name = "{"cauair_st" if mode == "spatiotemporal" else "cauair_ts"}"',
        f'alias = "{spec.track}"',
        f'path = "{bundle.as_posix()}"',
        "[dataset.params]",
        f"input_dim = {layout['input_dim'] if mode == 'spatiotemporal' else 1}",
        "scale = true",
        "[task]",
        f'mode = "{mode}"',
        f"seq_len = {spec.seq_len}",
        "label_len = 0",
        f"pred_len = {layout['lead']}",
        'features = "M"',
        "[model.params]",
        *(f"{k} = {json.dumps(v)}" for k, v in params.items()),
    ]
    for section, values in overrides.items():
        lines.append(f"[{section}]")
        lines.extend(f"{k} = {json.dumps(v)}" for k, v in values.items())
    path = work_dir / f"{model}.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def forecast_with_model(
    spec: RoundSpec,
    store: PanelStore,
    track: TrackSpec,
    model: str,
    work_root: Path,
    overrides: dict | None = None,
    submitter: str | None = None,
) -> ForecastSubmission:
    import torch

    from tsflab.experiments.config.loader import load_config
    from tsflab.experiments.runner.evaluator import _point_reduce, _resolve_output_kind
    from tsflab.experiments.runner.model_io import call_forecaster, make_decoder_input
    from tsflab.experiments.runner.run_one import _build_device, _build_loaders, _build_model, run_one

    work_dir = work_root / spec.track / spec.round_id
    bundle = work_dir / "bundle"
    layout = export_bundle(spec, store, track, bundle)
    config_path = _run_config(spec, model, bundle, layout, work_dir, dict(overrides or {}))
    loaded = load_config(str(config_path))[0]
    result = run_one(loaded.config, loaded.raw, loaded.sweep_keys)

    config = loaded.config
    device = _build_device(config.experiment.runtime)
    train_set, _, _, _, test_set, test_loader, adj_norm = _build_loaders(config)
    net, _ = _build_model(config, train_set, adj_norm, device)
    net.load_state_dict(torch.load(result.checkpoint_path, map_location=device, weights_only=True))
    net.eval()
    output_type, _ = _resolve_output_kind(net)
    with torch.no_grad():
        batch_x, batch_y, x_mark, y_mark = next(iter(test_loader))
        batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
        x_mark = x_mark.float().to(device) if x_mark is not None else None
        y_mark = y_mark.float().to(device) if y_mark is not None else None
        dec = make_decoder_input(batch_y, config.task.label_len, config.task.pred_len, device)
        out = call_forecaster(net, batch_x, x_mark, dec, y_mark).detach().cpu().numpy()
    if out.ndim == 4:
        out = _point_reduce(out, output_type, list(getattr(net, "quantile_levels", []) or []))
    path = out[0, -config.task.pred_len :, :]
    path = test_set.inverse_transform(path)
    predictions = path[-spec.horizon :]
    return ForecastSubmission(
        track=spec.track,
        round_id=spec.round_id,
        model=model,
        submitter=submitter,
        submitted_at=datetime.now(UTC).isoformat(timespec="seconds"),
        predictions=[[float(v) for v in row] for row in predictions],
        code_revision=_git_revision(),
    )


def _git_revision() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
