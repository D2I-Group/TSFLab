"""Runtime specification for MTSUNMixers."""

import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.mtsunmixers.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    d_model: int = Field(default=128, ge=1)
    d_state: int = Field(default=16, ge=1)
    d_conv: int = Field(default=4, ge=1)
    expand: int = Field(default=2, ge=1)
    e_layers: int = Field(default=1, ge=1)
    patch_len: int = Field(default=16, ge=1)
    time_bases: int = Field(default=16, ge=1)
    channel_bases: int = Field(default=16, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    recon_weight: float = Field(default=1.0, ge=0.0)
    use_norm: bool = True


_FIELDS = tuple(ModelParameterConfig.model_fields)


def build_model(cfg, params):
    values = {name: params[name] for name in _FIELDS if name in params}
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **values)


def training_objective(model: Model, batch: TrainingBatch, criterion):
    """Eq. (21): forecasting loss plus ``recon_weight`` times the L1 reconstruction of the history."""
    forecast, history = model.forecast_and_reconstruct(batch.x)
    loss = criterion(batch.align(forecast), batch.target)
    return forecast, loss + model.recon_weight * F.l1_loss(history, batch.x)


SPEC = ModelSpec(
    name="MTSUNMixers",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_mtsunmixers.toml",
    capabilities=frozenset({"time-series"}),
    components=("mamba",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
    training_objective=training_objective,
)
