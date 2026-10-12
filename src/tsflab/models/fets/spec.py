"""Model specification for FeTS."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.fets.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 32
    patch_len: int = 16
    stride: int = 8
    fourier_order: int = 2
    polynomial_order: int = 2
    kernel_size: int = 3
    dropout: float = 0.0
    use_revin: bool = True


def build_model(cfg, params):
    """Construct FeTS from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        patch_len=params["patch_len"],
        stride=params["stride"],
        fourier_order=params["fourier_order"],
        polynomial_order=params["polynomial_order"],
        kernel_size=params["kernel_size"],
        dropout=params["dropout"],
        use_revin=bool(params["use_revin"]),
    )


SPEC = ModelSpec(
    name="FeTS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
