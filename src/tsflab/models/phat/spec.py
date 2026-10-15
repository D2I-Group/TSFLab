"""Model specification for PHAT."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.phat.model import Model


class ModelParameterConfig(BaseModel):
    """Validated PHAT parameters supplied via ``model.params``."""

    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    n_heads: int = Field(default=8, gt=0)
    d_layers: int = Field(default=1, gt=0)
    attn_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    ffn_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    ffn_expand_ratio: float = Field(default=2.66667, gt=0.0)
    period_topk: int = Field(default=1, gt=0)
    period_list: list[int] | None = None

    @model_validator(mode="after")
    def _architecture_contract(self) -> ModelParameterConfig:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.period_list is not None and any(period <= 0 for period in self.period_list):
            raise ValueError("period_list entries must be positive")
        return self


def build_model(cfg, params):
    """Construct PHAT from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        d_layers=params["d_layers"],
        attn_dropout=params["attn_dropout"],
        ffn_dropout=params["ffn_dropout"],
        ffn_expand_ratio=params["ffn_expand_ratio"],
        period_topk=params["period_topk"],
        period_list=params.get("period_list"),
    )


SPEC = ModelSpec(
    name="PHAT",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
