"""Model specification for RPMixer."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.rpmixer.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    random_dim: int = Field(default=4, gt=0)
    e_layers: int = Field(default=3, gt=0)


def build_model(cfg, params):
    """Construct RPMixer from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        random_dim=params["random_dim"],
        e_layers=params["e_layers"],
    )


SPEC = ModelSpec(
    name="RPMixer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["spatiotemporal"]),
    components=("channel_wise_linear",),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
