"""Model specification for QuantilePatchTST."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.quantile_patchtst.model import Model

from typing import Optional

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    patch_len: int = 16
    stride: int = 8
    padding_patch: str = "end"
    e_layers: int = 3
    d_model: int = 512
    n_heads: int = 8
    d_k: Optional[int] = None
    d_v: Optional[int] = None
    d_ff: int = 2048
    activation: str = "gelu"
    norm: str = "BatchNorm"
    attn_dropout: float = 0.0
    ffn_dropout: float = 0.0
    res_dropout: float = 0.0
    proj_dropout: float = 0.0
    head_dropout: float = 0.0
    pre_norm: bool = False
    pe: str = "zeros"
    learn_pe: bool = False
    head_type: str = "flatten"
    individual: bool = False
    revin: bool = True
    affine: bool = False
    subtract_last: bool = False
    quantile_levels: list[float] | None = None


def build_model(cfg, params):
    """Construct QuantilePatchTST from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        features=cfg.task.features,
        patch_len=params["patch_len"],
        stride=params["stride"],
        padding_patch=params["padding_patch"],
        e_layers=params["e_layers"],
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        d_k=params.get("d_k"),
        d_v=params.get("d_v"),
        d_ff=params["d_ff"],
        activation=params["activation"],
        norm=params["norm"],
        attn_dropout=params["attn_dropout"],
        res_dropout=params["res_dropout"],
        ffn_dropout=params["ffn_dropout"],
        proj_dropout=params["proj_dropout"],
        head_dropout=params["head_dropout"],
        pre_norm=bool(params["pre_norm"]),
        pe=params["pe"],
        learn_pe=bool(params["learn_pe"]),
        head_type=params["head_type"],
        individual=bool(params["individual"]),
        revin=bool(params["revin"]),
        affine=bool(params["affine"]),
        subtract_last=bool(params["subtract_last"]),
        quantile_levels=params.get("quantile_levels") or list(cfg.evaluation.quantile_levels),
    )


SPEC = ModelSpec(
    name="QuantilePatchTST",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["quantile-output", "time-series"]),
    components=("patchtst", "quantile_head"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
