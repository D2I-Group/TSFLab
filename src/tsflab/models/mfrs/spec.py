"""Runtime specification for MFRS."""

from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.mfrs.model import Model, stamp_minutes


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    d_model: int = Field(default=512, ge=1)
    n_heads: int = Field(default=8, ge=1)
    e_layers: int = Field(default=2, ge=1)
    d_ff: int = Field(default=512, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    activation: Literal["gelu", "relu"] = "gelu"
    waveform: Literal["sine", "sawtooth", "rectangle", "pulse"] = "sine"
    periods: list[int] = Field(default_factory=list)
    extract: bool = True
    max_period: int = Field(default=1000, ge=4)
    harmonics: int = Field(default=8, ge=0)
    share_embedding: bool = True

    @model_validator(mode="after")
    def _check(self) -> "ModelParameterConfig":
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if any(period < 2 for period in self.periods):
            raise ValueError("periods must be integers of at least 2")
        return self


_FIELDS = tuple(ModelParameterConfig.model_fields)


def build_model(cfg, params):
    values = {name: params[name] for name in _FIELDS if name in params}
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **values)


def _dataset_series(train_loader, enc_in: int) -> torch.Tensor | None:
    """The loader dataset's ``[T, C]`` split when it has at least 4 steps, else ``None``."""
    data = getattr(getattr(train_loader, "dataset", None), "data", None)
    if data is None:
        return None
    series = torch.as_tensor(data).float()
    if series.ndim == 2 and series.shape[1] == enc_in and series.shape[0] >= 4:
        return series
    return None


def _training_series(train_loader, enc_in: int) -> torch.Tensor:
    """The ``[T, C]`` training split behind the loader, else rebuilt from time-ordered windows."""
    series = _dataset_series(train_loader, enc_in)
    if series is not None:
        return series
    starts, rows = [], []
    for batch in train_loader:
        x, x_mark = torch.as_tensor(batch[0]).float(), batch[2]
        if x_mark is None or not bool((torch.as_tensor(x_mark)[..., 0] > 0).all()):
            raise ValueError(
                "MFRS needs a series-backed training dataset or dated windows to fit its patterns"
            )
        starts.append(stamp_minutes(torch.as_tensor(x_mark)[:, 0]))
        rows.append(x[:, 0])
    order = torch.cat(starts).argsort()
    return torch.cat(rows)[order]


def training_setup(model: Model, train_loader, *, pred_len, features):
    """Sec. 3.2: extract base patterns from the training split before training."""
    del pred_len, features
    model.fit_reference_series(_training_series(train_loader, model.enc_in))


def training_objective(model: Model, batch: TrainingBatch, criterion):
    """MFRS trains with the ordinary forecasting loss; this hook only pairs with ``training_setup``."""
    forecast = batch.forecast(model)
    return forecast, criterion(batch.align(forecast), batch.target)


SPEC = ModelSpec(
    name="MFRS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_mfrs.toml",
    capabilities=frozenset({"time-series"}),
    components=("marks", "self_attention_family"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
    training_objective=training_objective,
    training_setup=training_setup,
)
