"""Model specification for FITS."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.fits.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    individual: bool = False
    cut_freq: int = 24


def build_model(cfg, params):
    """Construct FITS from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        individual=bool(params["individual"]),
        cut_freq=params["cut_freq"],
    )


SPEC = ModelSpec(
    name="FITS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
