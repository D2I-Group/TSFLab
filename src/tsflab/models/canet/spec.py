"""Runtime specification for CANet."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.canet.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    patch_sizes: tuple[int, ...] = (8, 64)
    embed_dim: int = Field(default=32, gt=0)
    output_features: int = Field(default=1024, gt=0)
    dropout: float = Field(default=0.5, ge=0.0, lt=1.0)
    blend_ratio: float = Field(default=0.1, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _architecture_contract(self) -> "ModelParameterConfig":
        if not self.patch_sizes:
            raise ValueError("patch_sizes must be non-empty")
        if any(size <= 0 for size in self.patch_sizes):
            raise ValueError("every patch size must be positive")
        return self


def build_model(cfg, params):
    """Construct CANet from a validated run configuration."""
    patch_sizes = tuple(params["patch_sizes"])
    for size in patch_sizes:
        if size > cfg.task.seq_len:
            raise ValueError("every patch size must not exceed seq_len")
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        patch_sizes=patch_sizes,
        embed_dim=params["embed_dim"],
        output_features=params["output_features"],
        dropout=params["dropout"],
        blend_ratio=params["blend_ratio"],
    )


SPEC = ModelSpec(
    name="CANet",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_canet.toml",
    capabilities=frozenset({"time-series"}),
    components=("adain_style_norm", "positional_encoding"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
