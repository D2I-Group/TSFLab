"""Runtime specification for GRAMODE."""

import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.gramode.model import Model, fit_from_loader


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``.

    ``num_nodes`` and the road-graph ``adj_mx`` are injected by the runner from the
    dataset; the DTW graph is built from the training split before the first epoch.
    """

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    input_dim: int = Field(default=1, ge=1)
    hidden_dim: int = Field(default=64, ge=1)
    tcn_mid_dim: int = Field(default=32, ge=1)
    kernel_size: int = Field(default=2, ge=1)
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    channels: int = Field(default=3, ge=1)
    layers: int = Field(default=2, ge=1)
    local_chunks: int = Field(default=4, ge=1)
    local_heads: int = Field(default=32, ge=1)
    output_heads: int = Field(default=12, ge=1)
    graph_alpha: float = Field(default=0.8, gt=0.0, lt=1.0)
    ode_beta: float = 0.6
    steps_per_day: int = Field(default=288, ge=1)
    semantic_sigma: float = Field(default=0.1, gt=0.0)
    semantic_threshold: float = Field(default=0.6, gt=0.0, lt=1.0)
    huber_delta: float = Field(default=1.0, gt=0.0)


_FIELDS = tuple(name for name in ModelParameterConfig.model_fields if name != "enc_in")


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        num_nodes=params.get("num_nodes", params["enc_in"]),
        adj_mx=params.get("adj_mx"),
        **{name: params[name] for name in _FIELDS if name in params},
    )


def training_setup(model, train_loader, *, pred_len, features):
    """Build the DTW graph (Eq. 2) from the training split before the first epoch."""
    del pred_len, features
    fit_from_loader(model, train_loader)


def training_objective(model, batch: TrainingBatch, criterion):
    """Huber loss (Eq. 24) on the forecast; the official code uses ``SmoothL1Loss`` (delta 1)."""
    del criterion
    forecast = batch.forecast(model)
    loss = F.huber_loss(batch.align(forecast), batch.target, delta=model.huber_delta)
    return forecast, loss


SPEC = ModelSpec(
    name="GRAMODE",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_gramode.toml",
    capabilities=frozenset({"spatiotemporal"}),
    components=("adj_norm", "marks"),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
    training_objective=training_objective,
    training_setup=training_setup,
)
