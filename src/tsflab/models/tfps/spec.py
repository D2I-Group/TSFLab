"""Runtime specification for TFPS."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.tfps.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    patch_len: int = Field(default=16, gt=0)
    stride: int = Field(default=8, gt=0)
    e_layers: int = Field(default=3, gt=0)
    n_heads: int = Field(default=4, gt=0)
    d_model: int = Field(default=16, gt=0)
    d_ff: int = Field(default=128, gt=0)
    dropout: float = Field(default=0.3, ge=0.0, lt=1.0)
    head_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    t_experts: int = Field(default=4, gt=0)
    t_top_k: int = Field(default=1, gt=0)
    f_experts: int = Field(default=4, gt=0)
    f_top_k: int = Field(default=1, gt=0)
    eta: float = Field(default=5.0, ge=0.0)
    beta: float = Field(default=0.1, ge=0.0)
    lambda_time: float = Field(default=1.0, ge=0.0)
    lambda_freq: float = Field(default=1.0, ge=0.0)
    individual: bool = True

    @model_validator(mode="after")
    def _contract(self) -> "ModelParameterConfig":
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.t_top_k > self.t_experts or self.f_top_k > self.f_experts:
            raise ValueError("top_k must not exceed the number of experts")
        if self.enc_in * self.d_model < max(self.t_experts, self.f_experts):
            raise ValueError("enc_in * d_model must be at least the number of experts")
        return self


def build_model(cfg, params):
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **params)


def training_objective(model: Model, batch: TrainingBatch, criterion):
    """Eq. (12): configured criterion plus the time and frequency clustering losses."""
    forecast, s_time, s_freq = model.forecast_with_affinity(batch.x)
    loss = criterion(batch.align(forecast), batch.target) + model.clustering_loss(s_time, s_freq)
    return forecast, loss


SPEC = ModelSpec(
    name="TFPS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_tfps.toml",
    capabilities=frozenset({"time-series"}),
    components=("flatten_forecast_head", "positional_encoding", "revin"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
    training_objective=training_objective,
)
