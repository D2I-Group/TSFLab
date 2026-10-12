"""Runtime specification for DiPELinear."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.dipelinear.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    num_experts: int = Field(default=1, gt=0)
    use_revin: bool = True
    use_time_w: bool = True
    use_freq_w: bool = True
    loss_alpha: float = Field(default=0.9, ge=0, le=1)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    temperature_start: float = Field(default=30.0, gt=0)
    temperature_end: float = Field(default=1.0, gt=0)
    anneal_epochs: int = Field(default=10, gt=0)

    @model_validator(mode="after")
    def _schedule(self) -> "ModelParameterConfig":
        if self.temperature_end > self.temperature_start:
            raise ValueError("need temperature_end <= temperature_start")
        return self


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        num_experts=params["num_experts"],
        use_revin=params["use_revin"],
        use_time_w=params["use_time_w"],
        use_freq_w=params["use_freq_w"],
        loss_alpha=params["loss_alpha"],
        dropout=params["dropout"],
        temperature_start=params["temperature_start"],
        temperature_end=params["temperature_end"],
        anneal_epochs=params["anneal_epochs"],
    )


def training_setup(model, train_loader, *, pred_len, features):
    """Record the epoch length so the router temperature can anneal per epoch."""
    del pred_len, features
    model.steps_per_epoch.fill_(len(train_loader))


def training_objective(model, batch, criterion):
    del criterion  # SFALoss replaces the configured observation loss
    forecast = batch.forecast(model)
    return forecast, model.training_objective(batch.align(forecast), batch.target)


SPEC = ModelSpec(
    name="DiPELinear",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_dipelinear.toml",
    capabilities=frozenset({"time-series"}),
    components=("fft_extrapolation_conv", "weight_set_router"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
    training_objective=training_objective,
    training_setup=training_setup,
)
