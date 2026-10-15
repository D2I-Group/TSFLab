"""Runtime specification for APTF."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.aptf.model import Model, loss_order


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    num_buckets: int = Field(default=9, ge=2)
    stage_epochs: int = Field(default=2, ge=1)
    amortization: bool = True
    patch_len: int = Field(default=16, ge=1)
    stride: int = Field(default=8, ge=1)
    e_layers: int = Field(default=3, ge=1)
    d_model: int = Field(default=128, ge=1)
    n_heads: int = Field(default=16, ge=1)
    d_ff: int = Field(default=256, ge=1)
    dropout: float = Field(default=0.2, ge=0, lt=1)
    head_dropout: float = Field(default=0.0, ge=0, lt=1)
    revin: bool = True

    @model_validator(mode="after")
    def _heads_divide_width(self) -> "ModelParameterConfig":
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self


def build_model(cfg, params):
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **params)


def training_setup(model, train_loader, *, pred_len, features):
    """Record the number of training batches per epoch; stages are counted in epochs (Eq. 2)."""
    del pred_len, features
    model.steps_per_epoch.fill_(len(train_loader))
    model.train_step.zero_()


def training_objective(model, batch: TrainingBatch, criterion):
    """Algorithm 2: amortized hierarchical predictability-aware loss.

    The source loss is bucketed by the amortization model's per-sample errors and
    the amortization loss by the source model's; both models have disjoint
    parameters, so their sum trains each on its own term. Without amortization
    the source model buckets its own errors.
    """
    stage = model.stage(model.current_epoch())
    model.train_step.add_(1)
    target = batch.target
    forecast = batch.forecast(model)
    source = batch.align(forecast)
    if model.amortizer is None:
        return forecast, model.hierarchical_loss(source, target, loss_order(source, target), stage, criterion)
    amortized = batch.align(model.amortizer(batch.x))
    source_loss = model.hierarchical_loss(source, target, loss_order(amortized, target), stage, criterion)
    amortizer_loss = model.hierarchical_loss(amortized, target, loss_order(source, target), stage, criterion)
    return forecast, source_loss + amortizer_loss


SPEC = ModelSpec(
    name="APTF",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_aptf.toml",
    capabilities=frozenset({"time-series"}),
    components=("patchtst",),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
    training_objective=training_objective,
    training_setup=training_setup,
)
