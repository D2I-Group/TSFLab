"""Model specification for TimePerceiver."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.timeperceiver.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 32
    n_heads: int = 2
    patch_len: int = 16
    dropout: float = 0.1
    num_latents: int = 8
    latent_dim: int = 128
    latent_d_ff: int = 256
    num_latent_blocks: int = 1
    query_share: bool = True


def build_model(cfg, params):
    """Construct TimePerceiver from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        label_len=cfg.task.label_len,
        features=cfg.task.features,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        patch_len=params["patch_len"],
        dropout=params["dropout"],
        num_latents=params["num_latents"],
        latent_dim=params["latent_dim"],
        latent_d_ff=params["latent_d_ff"],
        num_latent_blocks=params["num_latent_blocks"],
        query_share=bool(params["query_share"]),
    )


SPEC = ModelSpec(
    name="TimePerceiver",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
