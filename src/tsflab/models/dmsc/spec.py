"""Runtime specification for DMSC."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.dmsc.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    e_layers: int = Field(default=3, gt=0)
    d_model: int = Field(default=128, gt=0)
    d_ff: int = Field(default=256, gt=0)
    patch_min: int = Field(default=8, gt=0)
    patch_max: int = Field(default=64, gt=0)
    patch_decay: int = Field(default=2, ge=1)
    num_global: int = Field(default=2, gt=0)
    num_local: int = Field(default=8, gt=0)
    top_k: int = Field(default=2, gt=0)
    balance_coeff: float = Field(default=0.2, ge=0.0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    use_norm: bool = True

    @model_validator(mode="after")
    def _patch_and_expert_bounds(self) -> "ModelParameterConfig":
        if self.patch_min > self.patch_max:
            raise ValueError("DMSC needs 1 <= patch_min <= patch_max <= seq_len")
        if self.top_k > self.num_local:
            raise ValueError("DMSC needs patch_decay >= 1 and top_k <= num_local")
        return self


_FIELDS = tuple(ModelParameterConfig.model_fields)


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        **{name: params[name] for name in _FIELDS if name in params},
    )


def training_objective(model, batch: TrainingBatch, criterion):
    """Eq. (23): the configured forecast loss plus the ASR-MoE balance loss of Eq. (22)."""
    forecast = batch.forecast(model)
    loss = criterion(batch.align(forecast), batch.target)
    balance = model.last_balance_loss
    if balance is not None:
        loss = loss + balance
    return forecast, loss


SPEC = ModelSpec(
    name="DMSC",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_dmsc.toml",
    capabilities=frozenset({"time-series"}),
    components=("revin", "topk_expert_router"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
    training_objective=training_objective,
)
