"""Runtime specification for ARKAN."""

from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.arkan.model import (
    Model,
    autocorrelation_sums,
    lagged_moments,
    solve_least_squares,
    solve_yule_walker,
)


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    lag: int = Field(default=20, ge=1)
    hidden: int = Field(default=50, ge=1)
    grid: int = Field(default=3, ge=1)
    spline_order: int = Field(default=3, ge=0)
    grid_bound: float = Field(default=1.0, gt=0)
    ar_estimator: Literal["least_squares", "yule_walker"] = "least_squares"


def build_model(cfg, params):
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **params)


def _training_series(train_loader, enc_in: int):
    """Return the ``[T, C]`` training series behind the loader, or ``None``."""
    data = getattr(getattr(train_loader, "dataset", None), "data", None)
    if data is None:
        return None
    series = torch.as_tensor(data).float()
    if series.ndim != 2 or series.shape[1] != enc_in:
        return None
    return series


def training_setup(model, train_loader, *, pred_len, features):
    """Pre-fit the AR memory on the training split before the KAN is trained (Sec. III-B).

    Loaders without a series-backed dataset fall back to the lag windows inside
    every training history window.
    """
    del pred_len, features
    series = _training_series(train_loader, model.enc_in)
    if series is not None and series.shape[0] > model.lag:
        model.fit_ar(series)
        return
    _fit_ar_from_windows(model, train_loader)


def _add(total, part):
    return part if total is None else total + part


def _fit_ar_from_windows(model, train_loader) -> None:
    """Accumulate Yule-Walker sums/counts or least-squares moments over the history windows."""
    yule_walker = model.ar_estimator == "yule_walker"
    statistics = autocorrelation_sums if yule_walker else lagged_moments
    first = second = None
    for batch_x, *_ in train_loader:
        part = statistics(batch_x.double(), model.lag)
        first = _add(first, part[0])
        second = _add(second, part[1])
    if first is not None:
        solve = solve_yule_walker if yule_walker else solve_least_squares
        model.set_ar_weights(solve(first, second))


def training_objective(model, batch: TrainingBatch, criterion):
    """One-step-ahead training (Eq. 7 form): every horizon step is predicted from observed lags."""
    future = batch.y[:, -batch.pred_len :, :]
    predictions = model.teacher_forced(batch.x, future)
    return None, criterion(batch.align(predictions), batch.target)


SPEC = ModelSpec(
    name="ARKAN",
    module="tsflab.models.arkan",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/ARKAN.toml",
    model_card="src/tsflab/models/arkan/README.md",
    smoke_config="configs/runs/smoke_arkan.toml",
    capabilities=frozenset({"time-series"}),
    components=("bspline_basis",),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
    training_objective=training_objective,
    training_setup=training_setup,
)
