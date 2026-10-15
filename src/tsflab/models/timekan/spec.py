"""Model specification for TimeKAN."""

from __future__ import annotations

from pydantic import BaseModel, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.timekan.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int
    c_out: int | None = None
    d_model: int = 16
    e_layers: int = 1
    down_sampling_window: int = 2
    down_sampling_layers: int = 1
    begin_order: int = 0
    moving_avg: int = 25
    dropout: float = 0.1
    embed: str = "timeF"
    freq: str = "h"
    use_norm: int = 1

    @model_validator(mode="after")
    def check_head(self):
        if self.c_out not in (None, self.enc_in):
            raise ValueError("TimeKAN's variate-independent head requires c_out == enc_in")
        return self


def build_model(cfg, params):
    """Construct TimeKAN from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        label_len=cfg.task.label_len,
        features=cfg.task.features,
        enc_in=params["enc_in"],
        c_out=params["c_out"],
        d_model=params["d_model"],
        e_layers=params["e_layers"],
        down_sampling_window=params["down_sampling_window"],
        down_sampling_layers=params["down_sampling_layers"],
        begin_order=params["begin_order"],
        moving_avg=params["moving_avg"],
        dropout=params["dropout"],
        embed=params["embed"],
        freq=params["freq"],
        use_norm=params["use_norm"],
    )


SPEC = ModelSpec(
    name="TimeKAN",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
