"""Model specification for TimeFilter."""

from __future__ import annotations

from tsflab.experiments.runner.objective import TrainingBatch

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.timefilter.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    d_ff: int = 128
    e_layers: int = 2
    patch_len: int = 16
    dropout: float = 0.1
    top_p: float = 0.5
    pos: bool = True
    num_experts: int = 4


def build_model(cfg, params):
    """Construct TimeFilter from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        label_len=cfg.task.label_len,
        features=cfg.task.features,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        d_ff=params["d_ff"],
        e_layers=params["e_layers"],
        patch_len=params["patch_len"],
        dropout=params["dropout"],
        top_p=params["top_p"],
        pos=bool(params["pos"]),
        num_experts=params["num_experts"],
    )


MOE_LOSS_WEIGHT = 0.05  # alpha in the official training loop


def training_objective(model, batch: TrainingBatch, criterion):
    """Forecast criterion plus the weighted router load-balancing loss."""
    forecast = batch.forecast(model)
    loss = criterion(batch.align(forecast), batch.target)
    moe_loss = model.last_moe_loss
    if moe_loss is not None:
        loss = loss + MOE_LOSS_WEIGHT * moe_loss
    return forecast, loss


SPEC = ModelSpec(
    name="TimeFilter",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
    training_objective=training_objective,
)
