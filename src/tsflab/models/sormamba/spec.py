"""Runtime specification for SORMamba."""

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.sormamba.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    d_model: int = Field(default=128, ge=1)
    d_state: int = Field(default=16, ge=1)
    d_ff: int = Field(default=128, ge=1)
    e_layers: int = Field(default=2, ge=1)
    expand: int = Field(default=1, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    reg_lambda: float = Field(default=0.01, ge=0.0)
    mlp_residual: bool = False
    use_norm: bool = True


_FIELDS = tuple(ModelParameterConfig.model_fields)


def build_model(cfg, params):
    values = {name: params[name] for name in _FIELDS if name in params}
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **values)


def training_objective(model: Model, batch: TrainingBatch, criterion):
    """Eq. (4): forecasting loss plus ``lambda`` times the per-layer order regularizer."""
    forecast, regularizer = model.forecast_with_regularizer(batch.x)
    loss = criterion(batch.align(forecast), batch.target)
    return forecast, loss + model.reg_lambda * regularizer


SPEC = ModelSpec(
    name="SORMamba",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_sormamba.toml",
    capabilities=frozenset({"time-series"}),
    components=("mamba",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
    training_objective=training_objective,
)
