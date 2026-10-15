"""Model specification for Pathformer."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.pathformer.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(ge=1)
    layer_nums: int = Field(default=2, ge=1)
    k: int = 2
    num_experts: int = Field(default=4, ge=1)
    # Flat list of length layer_nums * num_experts, reshaped to per-layer
    # patch sizes. Each value must divide seq_len evenly.
    patch_size_list: list[int] = [16, 12, 8, 6, 16, 12, 8, 6]
    d_model: int = 16
    d_ff: int = 64
    residual_connection: int = 1
    revin: bool = True
    n_heads: int = Field(default=4, ge=1)
    dropout: float = 0.1

    @model_validator(mode="after")
    def _architecture_contract(self):
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        sizes = self.patch_size_list or [16, 12, 8, 6] * self.layer_nums
        if len(sizes) != self.layer_nums * self.num_experts or any(s < 1 for s in sizes):
            raise ValueError("patch_size_list must contain layer_nums * num_experts positive values")
        return self


def build_model(cfg, params):
    """Construct Pathformer from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        features=cfg.task.features,
        enc_in=params["enc_in"],
        layer_nums=params["layer_nums"],
        k=params["k"],
        num_experts=params["num_experts"],
        patch_size_list=params["patch_size_list"],
        d_model=params["d_model"],
        d_ff=params["d_ff"],
        residual_connection=params["residual_connection"],
        revin=bool(params["revin"]),
        n_heads=params["n_heads"],
        dropout=params["dropout"],
    )


SPEC = ModelSpec(
    name="Pathformer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
