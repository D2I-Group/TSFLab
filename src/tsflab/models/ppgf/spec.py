"""Runtime specification for PPGF."""

import torch
from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.ppgf.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    num_groups: int = Field(default=4, ge=2)
    d_model: int = Field(default=32, ge=1)
    conv_kernel: int = Field(default=2, ge=1)
    n_heads: int = Field(default=8, ge=1)
    d_k: int = Field(default=64, ge=1)
    d_ff: int = Field(default=2048, ge=1)
    e_layers: int = Field(default=1, ge=1)
    grn_hidden: int = Field(default=416, ge=1)
    hidden_dim: int = Field(default=300, ge=1)
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    pos_dropout: float = Field(default=0.3, ge=0.0, lt=1.0)
    lambda_conf: float = Field(default=1.0, ge=0.0)
    lambda_cls: float = Field(default=1.0, ge=0.0)
    lambda_reg: float = Field(default=5.0, ge=0.0)


_FIELDS = tuple(ModelParameterConfig.model_fields)


def build_model(cfg, params):
    values = {name: params[name] for name in _FIELDS if name in params}
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **values)


def _full_training_series(train_loader, enc_in: int):
    """Return the full ``(T, C)`` scaled training series behind the loader, or ``None``."""
    data = getattr(getattr(train_loader, "dataset", None), "data", None)
    if data is None:
        return None
    series = torch.as_tensor(data).float()
    if series.ndim != 2 or series.shape[1] != enc_in or series.shape[0] < 2:
        return None
    return series


def training_setup(model, train_loader, *, pred_len, features):
    """Fit the per-channel group edges of Eqs. (1)-(2) on the training split.

    The paper sorts the whole training series; loaders without a series-backed
    dataset fall back to the newest history step of every stride-1 window.
    """
    del pred_len, features
    series = _full_training_series(train_loader, model.enc_in)
    if series is None:
        series = torch.cat([batch_x[:, -1, :].float() for batch_x, *_ in train_loader])
    model.fit_groups(series.to(model.edges.device))


def training_objective(model, batch: TrainingBatch, criterion):
    """Eq. (11) replaces the observation loss: confidence, classification and relative terms."""
    del criterion
    parts = model.heads(batch.x)
    forecast = model.predict(parts)
    aligned = {name: batch.align(value) for name, value in parts.items()}
    edges = model.edges[-1:] if batch.features == "MS" else model.edges
    return forecast, model.pattern_loss(aligned, batch.target, edges)


SPEC = ModelSpec(
    name="PPGF",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_ppgf.toml",
    capabilities=frozenset({"time-series"}),
    components=("embed",),
    contract_task={"seq_len": 32, "pred_len": 12, "label_len": 0},
    training_objective=training_objective,
    training_setup=training_setup,
)
