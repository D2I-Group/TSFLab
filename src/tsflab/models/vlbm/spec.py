"""Runtime specification for VLBM."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.vlbm.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    hidden: int = Field(default=256, gt=0)
    num_blocks: int = Field(default=2, ge=1)
    top_k: int = Field(default=8, ge=1)
    num_basis: int = Field(default=800, ge=1)
    interval: float = Field(default=5.0, gt=0.0, le=1440.0)
    use_norm: bool = True
    kl_weight: float = Field(default=0.1, ge=0.0)


def build_model(cfg, params):
    # The runner injects ``adj_mx``/``num_nodes`` for graph datasets; the official
    # model always uses the identity mask, so they are not model parameters.
    params = dict(params)
    num_nodes = params.pop("num_nodes", None)
    params.pop("adj_mx", None)
    options = ModelParameterConfig(**params).model_dump()
    if num_nodes is not None and int(num_nodes) != options["enc_in"]:
        raise ValueError(f"VLBM enc_in={options['enc_in']} does not match num_nodes={num_nodes}")
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **options)


def training_objective(model, batch: TrainingBatch, criterion):
    """Eq. 33: configured forecast loss on the prior-path forecast plus ``kl_weight * KL(q || p)``.

    The future-aware posterior embeds the whole future window of every variable
    and the calendar slot of its last step.
    """
    future = batch.y[:, -batch.pred_len :, :]
    if future.shape[-1] != model.enc_in:
        raise ValueError("VLBM's posterior needs the future of every variable")
    future_marks = None if batch.y_mark is None else batch.y_mark[:, -batch.pred_len :]
    out = model.variational_forward(batch.x, batch.x_mark, future, future_marks)
    loss = criterion(batch.align(out.forecast), batch.target) + model.kl_weight * out.kl()
    return out.forecast, loss


SPEC = ModelSpec(
    name="VLBM",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_vlbm.toml",
    capabilities=frozenset({"time-series", "spatiotemporal"}),
    components=("marks", "revin"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
    training_objective=training_objective,
)
