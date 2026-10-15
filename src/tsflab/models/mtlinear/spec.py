"""Runtime specification for MTLinear."""

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.mtlinear.model import LAYER_TYPES, Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    layer_type: str = "DLinear"
    kernel_size: int = Field(default=25, ge=1)
    cluster_dist: float = Field(default=0.5, ge=0)
    max_clusters: int = Field(default=8, ge=1)
    penalty_param: float = Field(default=2.0, ge=0)
    use_horizon_penalty: bool = True
    use_variates_penalty: bool = True

    @field_validator("layer_type")
    @classmethod
    def _known_layer(cls, value: str) -> str:
        if value not in LAYER_TYPES:
            raise ValueError(f"layer_type must be one of {LAYER_TYPES}")
        return value

    @field_validator("kernel_size")
    @classmethod
    def _odd_kernel(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        return value


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        layer_type=params["layer_type"],
        kernel_size=params["kernel_size"],
        cluster_dist=params["cluster_dist"],
        max_clusters=params["max_clusters"],
        penalty_param=params["penalty_param"],
        use_horizon_penalty=params["use_horizon_penalty"],
        use_variates_penalty=params["use_variates_penalty"],
    )


def _full_training_series(train_loader, enc_in: int):
    """Return the full ``(T, C)`` training series behind the loader, or ``None``."""
    data = getattr(getattr(train_loader, "dataset", None), "data", None)
    if data is None:
        return None
    series = torch.as_tensor(data).float()
    if series.ndim != 2 or series.shape[1] != enc_in or series.shape[0] < 2:
        return None
    return series


def training_setup(model, train_loader, *, pred_len, features):
    """Fit the variate grouping on the training split (MTLinear Sec. 4.2).

    The official code clusters on the whole scaled training series, so the loader's
    dataset series is used in full. Loaders without a series-backed dataset fall back
    to the newest step of every stride-1 window, which covers the training series
    minus the first lookback.
    """
    series = _full_training_series(train_loader, model.enc_in)
    if series is None:
        steps = [batch_x[:, -1, :].float() for batch_x, *_ in train_loader]
        series = torch.cat(steps)
    model.fit_clusters(series.to(model.group_of.device))


def training_objective(model, batch: TrainingBatch, criterion):
    """Eq. (7)-(8) replaces the observation loss with the penalized squared error."""
    forecast = batch.forecast(model)
    if not (model.use_horizon_penalty or model.use_variates_penalty):
        return forecast, criterion(batch.align(forecast), batch.target)
    return forecast, model.penalized_loss(batch.align(forecast), batch.target)


SPEC = ModelSpec(
    name="MTLinear",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_mtlinear.toml",
    capabilities=frozenset({"time-series"}),
    components=("channel_wise_linear", "dlinear", "last_value_center", "revin"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
    training_objective=training_objective,
    training_setup=training_setup,
)
