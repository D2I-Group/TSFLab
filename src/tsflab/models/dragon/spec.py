"""Runtime specification for DRAGON."""

from __future__ import annotations

from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.dragon.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    d_model: int = Field(default=16, gt=0)
    d_ff: int = Field(default=32, gt=0)
    e_layers: int = Field(default=2, ge=1)
    top_k: int = Field(default=5, ge=1)
    num_kernels: int = Field(default=6, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    k: int = Field(default=4, ge=2)
    alphabet_sizes: list[Annotated[int, Field(ge=1)]] = Field(default_factory=lambda: [20, 25, 30], min_length=1)
    d_graph: int = Field(default=16, gt=0)
    graph_layers: int = Field(default=2, ge=2)
    node_feat_size: int = Field(default=4, ge=1)
    graph_heads: int = Field(default=16, ge=1)
    graph_dropout: float = Field(default=0.4, ge=0.0, lt=1.0)
    use_gdc: bool = True
    gdc_topk: int = Field(default=32, ge=1)
    ppr_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _contract(self) -> "ModelParameterConfig":
        if self.d_model % 2:
            raise ValueError("d_model must be even (sinusoidal positions)")
        return self


def build_model(cfg, params):
    options = ModelParameterConfig(**params).model_dump()
    options["alphabet_sizes"] = tuple(options["alphabet_sizes"])
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **options)


def training_setup(model, train_loader, *, pred_len, features):
    """Build every MdBG from the whole scaled training split, as the official ``dBG_Dataset`` does."""
    del pred_len, features
    data = getattr(getattr(train_loader, "dataset", None), "data", None)
    if data is None:
        raise ValueError("DRAGON needs the training split series (train_loader.dataset.data)")
    series = np.asarray(data, dtype=np.float64)
    if series.ndim != 2 or series.shape[1] != model.enc_in:
        raise ValueError(
            f"DRAGON expects a [T, {model.enc_in}] training series, got {series.shape}"
        )
    model.fit_graphs(series)


def training_objective(model, batch: TrainingBatch, criterion):
    """The configured forecast loss; declared so that ``training_setup`` can build the graphs."""
    forecast = batch.forecast(model)
    return forecast, criterion(batch.align(forecast), batch.target)


SPEC = ModelSpec(
    name="DRAGON",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_dragon.toml",
    capabilities=frozenset({"time-series"}),
    components=("dominant_periods", "embed", "inception_block", "marks", "revin"),
    contract_task={"seq_len": 12, "pred_len": 96, "label_len": 6},
    training_objective=training_objective,
    training_setup=training_setup,
)
